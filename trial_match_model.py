#!/usr/bin/env python3
"""
TrialMatchModel: XGBoost-based numeric classifier + pipeline glue.
Corrected and improved to:
 - expose train_model(X,y)
 - expose predict(patient_text, trial_data)  <-- corrected signature
 - return feature_contributions in predict dict
 - save/load model + scaler
 - get_feature_importance()
 - handle age None safely
"""

import numpy as np
import pickle
from pathlib import Path
import logging

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Optional imports from your existing helpers
from test_trial_matching import extract_biobert_entities, map_to_umls_realtime, create_structured_profile, convert_to_fhir_format

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class TrialMatchModel:
    def __init__(self, umls_client=None):
        self.umls_client = umls_client

        # scaler for numeric features
        self.feature_scaler = StandardScaler()

        # XGBoost params (tuned reasonably)
        self.ml_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )

        self.is_trained = False

        # Keep a canonical feature_names ordering used by trainer
        self.feature_names = ["umls_match",
                              "condition_match", "entity_count", "age_match"]

    # ---------------------------
    # Training / predict helpers
    # ---------------------------
    def train_model(self, X, y):
        """
        X: numpy array (n_samples x n_features)
        y: labels (n_samples,)
        """
        logging.info("Scaling features and fitting XGBoost model...")
        X_scaled = self.feature_scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
        self.is_trained = True
        logging.info("Model training complete.")

    def predict_probability(self, X):
        """
        Return probability (n_samples,) for positive class.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        X_scaled = self.feature_scaler.transform(X)
        # If binary classification, take column 1
        probs = self.ml_model.predict_proba(X_scaled)
        if probs.shape[1] == 1:
            return probs.ravel()
        return probs[:, 1]

    def _calculate_age_match(self, patient_age, min_age, max_age):
        """
        Computes how well the patient's age fits within the trial's age eligibility range.
        Returns a float between 0 and 1.
        """
        try:
            # Handle None or non-numeric cases
            if patient_age is None:
                return 0.5  # neutral score
            patient_age = float(patient_age)
            min_age = float(min_age) if min_age is not None else 0
            max_age = float(max_age) if max_age is not None else 120

            # Within range
            if min_age <= patient_age <= max_age:
                return 1.0
            # Below minimum or above maximum
            elif patient_age < min_age:
                diff = min_age - patient_age
                return max(0.0, 1 - diff / 50.0)  # penalize if far below
            else:
                diff = patient_age - max_age
                return max(0.0, 1 - diff / 50.0)  # penalize if far above
        except Exception:
            return 0.0

    # ---------------------------
    # Feature extraction - keep consistent with trainer
    # processed_data structure is expected as produced in Trainer.preprocess_and_cache
    # ---------------------------
    def extract_features(self, processed_data, trial_data):
        """
        processed_data: {'profile': {...}, 'entities': [...], 'umls_concepts': [...]}
        trial_data: {'eligibility_criteria': str, 'conditions': [], 'umls_concepts': []}
        """
        profile = processed_data.get("profile", {})
        structured = profile.get("structured_data", {}) if isinstance(
            profile, dict) else {}

        # patient conditions and trial conditions (lowercased)
        patient_conditions = set(
            [c.lower() for c in structured.get("conditions", []) if isinstance(c, str)])
        trial_conditions = set([c.lower() for c in trial_data.get(
            "conditions", []) if isinstance(c, str)])

        # umls concept overlap
        patient_cuis = set(processed_data.get("umls_concepts", []))
        trial_cuis = set(trial_data.get("umls_concepts", []))

        umls_match = len(patient_cuis & trial_cuis)
        condition_match = len(patient_conditions & trial_conditions)

        # entity_count
        entity_count = len(processed_data.get("entities", []) or [])

        # age match: handle None safely
        patient_age = structured.get(
            "age") if structured.get("age") is not None else 0
        age_match = self._calculate_age_match(patient_age, trial_data.get(
            "minimum_age", 18), trial_data.get("maximum_age", 100))

        return {
            "umls_match": int(umls_match),
            "condition_match": int(condition_match),
            "entity_count": int(entity_count),
            "age_match": float(age_match)
        }

    def predict(self, patient_text, trial_data):
        """
        Full pipeline inference. This function:
         - extracts entities / UMLS / profile if possible (uses helpers),
         - extracts numeric features,
         - if model trained obtains ml_score (probability),
         - computes final rule+ML combined score,
         - returns dict with 'score', 'probability', 'feature_contributions', 'processed_data', etc.
        """
        # Prefer to use preprocessed profile if user supplies one in trial_data
        # but here we accept raw patient_text and compute minimal processed_data
        try:
            entities = extract_biobert_entities(patient_text)
        except Exception:
            entities = []

        try:
            umls_concepts = map_to_umls_realtime(
                entities, self.umls_client) if self.umls_client else []
        except Exception:
            umls_concepts = [f"UNKNOWN::{e}" for e in entities]

        try:
            profile = create_structured_profile(
                patient_text, entities, umls_concepts)
        except Exception:
            profile = {"structured_data": {
                "conditions": [], "medications": [], "age": None}}

        processed_data = {
            "entities": entities,
            "umls_concepts": umls_concepts,
            "profile": profile
        }

        # Ensure trial_data contains umls_concepts key for feature extraction
        if "umls_concepts" not in trial_data:
            trial_data["umls_concepts"] = []

        features = self.extract_features(processed_data, trial_data)
        feature_array = np.array([[features[k] for k in self.feature_names]])

        # ML probability if trained
        ml_score = None
        if self.is_trained:
            try:
                ml_score = float(self.predict_probability(feature_array)[0])
            except Exception:
                ml_score = 0.0

        # Combine rule-based and ML score:
        # rule_score is normalized from condition_match count (could be >1) so we clamp/scale sensibly
        # a simple normalized rule_score = min(condition_match, 1.0)
        rule_condition_score = float(min(features["condition_match"], 1.0))
        if ml_score is not None:
            final_score = (0.7 * rule_condition_score + 0.3 * ml_score) * 100
            probability = ml_score
        else:
            final_score = rule_condition_score * 100
            probability = 0.0

        # Feature contributions (simple weighted breakdown)
        # weights used for contributions should match your aggregate logic
        weights = {
            "umls_match": 0.0,
            "condition_match": 0.7,
            "entity_count": 0.0,
            "age_match": 0.0
        }
        # If ml_score present, give it some representation in contributions:
        contribs = {}
        total_weight = sum(weights.values()) + \
            (0.3 if ml_score is not None else 0.0)
        for k, w in weights.items():
            contribs[k] = float(w * features.get(k, 0) /
                                (total_weight if total_weight else 1))
        if ml_score is not None:
            contribs["ml_score"] = float(
                0.3 * ml_score / (total_weight if total_weight else 1))
        # Normalize to simple floats
        for k in contribs:
            contribs[k] = float(contribs[k])

        return {
            "score": int(final_score),
            "probability": float(probability),
            "processed_data": processed_data,
            "features": features,
            "feature_contributions": contribs
        }

    # ---------------------------
    # Utility / persistence
    # ---------------------------
    def save_model(self, path: Path = None):
        """Save model and scaler to path (pickle)"""
        if path is None:
            path = Path("c:/AyuSynapse/models/model.pkl")
        obj = {
            "model": self.ml_model,
            "scaler": self.feature_scaler,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names
        }
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load model + scaler from path"""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.ml_model = obj["model"]
        self.feature_scaler = obj["scaler"]
        self.is_trained = obj.get("is_trained", False)
        self.feature_names = obj.get("feature_names", self.feature_names)
        logging.info(f"Model loaded from {path}")

    def get_feature_importance(self):
        if not self.is_trained:
            return {}
        try:
            importances = self.ml_model.feature_importances_
            return {name: float(val) for name, val in zip(self.feature_names, importances)}
        except Exception:
            return {}
