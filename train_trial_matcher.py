#!/usr/bin/env python3
"""
Refactored training driver for Clinical Trial Matchmaking.
Features:
 - Loads heavy models once
 - Caches per-example processed features (entities, UMLS, profile)
 - Offline mode to avoid live UMLS calls (--offline)
 - Corrected predict() call, added feature_contributions handling
 - Produces outputs: metrics.json, detailed_results.json, training_examples.csv, model.pkl
"""

import argparse
import hashlib
import json
import logging
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Import your existing helper functions / classes
from test_trial_matching import (
    extract_biobert_entities,
    map_to_umls_realtime,
    create_structured_profile,
    convert_to_fhir_format
)
from umls_client import UMLSClient
from trial_match_model import TrialMatchModel

import pandas as pd
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
BASE_DIR = Path("c:/AyuSynapse")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Cache helpers


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_path_for_text(text_hash: str) -> Path:
    return CACHE_DIR / f"{text_hash}.pkl"


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Full preprocessing pipeline for a single patient text (reuses external helpers)


def full_preprocess(patient_text: str, umls_client: UMLSClient, offline: bool = False):
    """
    Returns dict {
      'entities': ...,
      'umls_concepts': ...,
      'profile': ...,
      'fhir': ...
    }
    Uses cached UMLS results when offline=True or on failure.
    """
    # 1. Entities (BioBERT)
    entities = extract_biobert_entities(patient_text)

    # 2. UMLS mapping (with retry / fallback)
    umls_concepts = []
    try:
        if offline:
            raise Exception("Offline mode: skipping UMLS API")
        umls_concepts = map_to_umls_realtime(entities, umls_client)
    except Exception as e:
        logging.warning(
            f"UMLS mapping failed or offline mode: {str(e)} — using placeholder CUIs")
        # fallback: mark each entity with UNKNOWN_... (keeps structure)
        umls_concepts = [f"UNKNOWN::{ent}" for ent in entities]

    # 3. Structured profile
    profile = create_structured_profile(patient_text, entities, umls_concepts)

    # 4. FHIR conversion if available
    try:
        fhir = convert_to_fhir_format(profile)
    except Exception:
        fhir = {}

    return {
        "entities": entities,
        "umls_concepts": umls_concepts,
        "profile": profile,
        "fhir": fhir
    }


class Trainer:
    def __init__(self, umls_client: UMLSClient, offline: bool = False):
        self.umls_client = umls_client
        self.offline = offline
        # instantiate model (model class handles its own ml model and vectorizers)
        self.model = TrialMatchModel(umls_client=self.umls_client)

    def preprocess_and_cache(self, df: pd.DataFrame, force_recompute: bool = False):
        """
        For each row in df (expects patient_text, eligibility_text, label),
        compute or load cached processed data and features.
        Returns a list of feature dicts and labels.
        """
        features_list = []
        labels = []
        rows = list(df.itertuples(index=False))
        for i, row in enumerate(tqdm(rows, desc="Preprocessing & caching")):
            patient_text = row.patient_text
            eligibility_text = row.eligibility_text
            label = row.label

            key = _hash_text(patient_text)
            cache_file = cache_path_for_text(key)

            if cache_file.exists() and not force_recompute:
                processed = load_pickle(cache_file)
                logging.info(f"✅ Loaded cached features for example #{i+1}")
            else:
                logging.info(
                    f"🔁 Computing features (cache miss) for example #{i+1}")
                processed = full_preprocess(
                    patient_text, self.umls_client, offline=self.offline)
                save_pickle(processed, cache_file)

            # Build trial_data (include umls_concepts)
            trial_data = {
                "eligibility_criteria": eligibility_text,
                "conditions": processed["profile"].get("structured_data", {}).get("conditions", []),
                "umls_concepts": processed.get("umls_concepts", [])
            }

            # Extract numerical features using model.extract_features
            # Note: model.extract_features expects processed_data similar structure
            processed_data_for_model = {
                "profile": processed["profile"],
                "entities": processed["entities"],
                "umls_concepts": processed["umls_concepts"]
            }
            feat_dict = self.model.extract_features(
                processed_data_for_model, trial_data)

            features_list.append(feat_dict)
            labels.append(label)

        return features_list, labels

    def train(self, df: pd.DataFrame, force_recompute: bool = False):
        """Main training routine: preprocess -> train model -> evaluate -> save outputs."""
        # 1. Preprocess and cache (heavy step)
        X_dicts, y = self.preprocess_and_cache(
            df, force_recompute=force_recompute)

        # If no features, abort
        if len(X_dicts) == 0:
            raise RuntimeError("No features extracted. Check dataset.")

        # Convert list of dicts to numpy array (respect order of feature names)
        feature_names = sorted(X_dicts[0].keys())
        X = np.array([[d.get(fn, 0) for fn in feature_names] for d in X_dicts])
        y = np.array(y)

        logging.info(f"Feature matrix shape: {X.shape}, labels: {y.shape}")

        # 2. Train model
        logging.info("Training ML model...")
        self.model.train_model(X, y)

        # 3. Evaluate on training set (simple) and produce outputs
        probs = self.model.predict_probability(X)
        # threshold 0.5 for binary label - adapt for your labels
        preds = (probs >= 0.5).astype(int)
        # Note: your labels might be 0,1,2; adjust metrics accordingly if multiclass
        try:
            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds, average="weighted")
            roc = roc_auc_score(y, probs, multi_class="ovr") if len(
                np.unique(y)) > 1 else float("nan")
        except Exception:
            acc = float("nan")
            f1 = float("nan")
            roc = float("nan")

        logging.info(
            f"Training done — Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc}")

        # 4. Create detailed results (per-example)
        detailed_results = []
        for i, feat in enumerate(X_dicts):
            # regenerate processed for output convenience (load from cache)
            key = _hash_text(df.iloc[i].patient_text)
            processed = load_pickle(cache_path_for_text(key))

            trial_data = {
                "eligibility_criteria": df.iloc[i].eligibility_text,
                "conditions": processed["profile"].get("structured_data", {}).get("conditions", []),
                "umls_concepts": processed.get("umls_concepts", [])
            }

            # Use corrected predict signature
            pred = self.model.predict(df.iloc[i].patient_text, trial_data)
            # Ensure feature_contributions exists
            feature_contribs = pred.get("feature_contributions", {})

            detailed_results.append({
                "index": i,
                "patient_text": df.iloc[i].patient_text,
                "trial_eligibility_text": df.iloc[i].eligibility_text,
                "label": int(df.iloc[i].label),
                "pred_score": int(pred.get("score", 0)),
                "pred_probability": float(pred.get("probability", 0.0)),
                "feature_contributions": feature_contribs
            })

        # 5. Save outputs
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_dir = MODELS_DIR / timestamp
        out_dir.mkdir(exist_ok=True)

        metrics = {
            "timestamp": timestamp,
            "n_examples": int(len(df)),
            "accuracy": acc if not np.isnan(acc) else None,
            "f1": f1 if not np.isnan(f1) else None,
            "roc_auc": roc if not np.isnan(roc) else None
        }

        # Save files
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(out_dir / "detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        pd.DataFrame(detailed_results).to_csv(
            out_dir / "training_examples.csv", index=False)

        # Save trained model (model.save_model handles pickle of model and scalers)
        model_file = out_dir / "model.pkl"
        self.model.save_model(model_file)

        logging.info(f"Saved outputs to {out_dir}")

        return metrics, detailed_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="c:/AyuSynapse/Dataset/training_examples.csv",
                        help="CSV with patient_text, eligibility_text, label")
    parser.add_argument("--offline", action="store_true",
                        help="Offline mode: do not call UMLS API")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute of cached features")
    args = parser.parse_args()

    logging.info("Starting training sanity check...")

    # Load dataset
    df = pd.read_csv(args.data)
    logging.info(f"Loaded dataset shape: {df.shape}")

    # Basic column check
    required = {"patient_text", "eligibility_text", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Missing required columns: {required - set(df.columns)}")

    # Prepare UMLS client (if offline this client may still be used for caching)
    umls_client = UMLSClient()

    trainer = Trainer(umls_client=umls_client, offline=args.offline)
    metrics, detailed_results = trainer.train(df, force_recompute=args.force)

    print("\nTraining completed.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
