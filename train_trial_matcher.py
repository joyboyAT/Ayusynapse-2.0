#!/usr/bin/env python3
"""
Refactored training driver for Clinical Trial Matchmaking.
"""

import argparse
import hashlib
import json
import logging
import pickle
import re
import time
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

try:
    from textblob import TextBlob
    sentiment_available = True
except:
    sentiment_available = False

# Import deep learning trainer
try:
    from deep_learning_matcher import DeepLearningTrainer
    dl_available = True
except Exception as e:
    dl_available = False

# Import existing helper functions
from test_trial_matching import (
    extract_biobert_entities,
    map_to_umls_realtime,
    create_structured_profile,
    convert_to_fhir_format
)
from umls_client import UMLSClient
from trial_match_model import TrialMatchModel

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize semantic model
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    semantic_model = None

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


def full_preprocess(patient_text: str, umls_client: UMLSClient, offline: bool = False):
    """Returns dict with entities, UMLS concepts, profile, FHIR"""
    entities = extract_biobert_entities(patient_text)

    umls_concepts = []
    try:
        if offline:
            logging.info("🔴 OFFLINE MODE: Skipping UMLS API calls")
            raise Exception("Offline mode: skipping UMLS API")

        logging.info("🟢 ONLINE MODE: Calling live UMLS API...")
        umls_concepts = map_to_umls_realtime(entities, umls_client)
        logging.info(f"✅ Got {len(umls_concepts)} UMLS concepts from API")
    except Exception as e:
        logging.warning(f"UMLS API failed or offline mode: {str(e)}")
        umls_concepts = []
        for ent in entities:
            if isinstance(ent, str):
                if any(term in ent.lower() for term in ['parkinson', 'tremor']):
                    umls_concepts.append("C0030567::ParkinsonDisease")
                elif any(term in ent.lower() for term in ['cancer', 'tumor']):
                    umls_concepts.append("C0006826::MalignantNeoplasm")
                else:
                    umls_concepts.append(f"FALLBACK::{ent}")
        logging.info(f"🔧 Using {len(umls_concepts)} fallback UMLS concepts")

    profile = create_structured_profile(patient_text, entities, umls_concepts)

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


def analyze_clinical_sentiment(text):
    """Analyze sentiment and tone of clinical text"""
    if not sentiment_available:
        return {
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0,
            'clinical_tone': 0,
            'positive_indicators': 0,
            'negative_indicators': 0
        }

    try:
        blob = TextBlob(text)
        positive_medical = r'\b(improving|stable|controlled|recovery|normal|healthy)\b'
        negative_medical = r'\b(worsening|deteriorating|complications|failure|abnormal|severe)\b'

        positive_count = len(re.findall(positive_medical, text.lower()))
        negative_count = len(re.findall(negative_medical, text.lower()))
        medical_polarity = (positive_count - negative_count) / \
            max(len(text.split()), 1)

        return {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'clinical_tone': medical_polarity,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
    except:
        return {
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0,
            'clinical_tone': 0,
            'positive_indicators': 0,
            'negative_indicators': 0
        }


def detect_negation(text):
    """Detect negation patterns in medical text"""
    negation_patterns = [
        r'\bno\s+(?:history\s+of\s+)?(\w+)',
        r'\bdenies\s+(\w+)',
        r'\bwithout\s+(\w+)',
        r'\bnegative\s+for\s+(\w+)',
        r'\bno\s+evidence\s+of\s+(\w+)',
        r'\bfree\s+from\s+(\w+)',
        r'\bnever\s+had\s+(\w+)',
        r'\bnot\s+(\w+)',
    ]

    negated_terms = []
    for pattern in negation_patterns:
        matches = re.findall(pattern, text.lower())
        negated_terms.extend(matches)

    return {
        'negation_count': len(negated_terms),
        'negated_terms': negated_terms
    }


def extract_temporal_context(text):
    """Extract temporal information from medical text"""
    temporal_patterns = {
        'current': r'\b(current|currently|present|ongoing|active)\b',
        'past': r'\b(history\s+of|previous|former|past|prior)\b',
        'recent': r'\b(recent|recently|new|newly)\b',
        'chronic': r'\b(chronic|long-standing|longstanding)\b',
        'acute': r'\b(acute|sudden|emergency)\b',
        'duration': r'\b(\d+)\s+(years?|months?|weeks?|days?)\b'
    }

    temporal_features = {}
    for category, pattern in temporal_patterns.items():
        matches = re.findall(pattern, text.lower())
        temporal_features[f'temporal_{category}'] = len(matches)

    return temporal_features


def extract_severity_indicators(text):
    """Extract severity and urgency indicators"""
    severity_patterns = {
        'severe': r'\b(severe|serious|critical|life-threatening|grave)\b',
        'moderate': r'\b(moderate|significant|considerable)\b',
        'mild': r'\b(mild|minor|slight|minimal)\b',
        'controlled': r'\b(controlled|stable|managed|well-controlled)\b',
        'uncontrolled': r'\b(uncontrolled|unstable|poorly-controlled|refractory)\b',
        'urgent': r'\b(urgent|emergency|immediate|stat)\b'
    }

    severity_features = {}
    for category, pattern in severity_patterns.items():
        matches = re.findall(pattern, text.lower())
        severity_features[f'severity_{category}'] = len(matches)

    severity_score = (
        severity_features.get('severity_severe', 0) * 3 +
        severity_features.get('severity_moderate', 0) * 2 +
        severity_features.get('severity_mild', 0) * 1 +
        severity_features.get('severity_uncontrolled', 0) * 2 -
        severity_features.get('severity_controlled', 0) * 1
    )
    severity_features['severity_score'] = max(severity_score, 0)

    return severity_features


def extract_medical_keywords(text):
    """Extract medical terms and keywords"""
    medical_patterns = {
        'diseases': r'\b(diabetes|cancer|hypertension|asthma|copd|heart disease|stroke|kidney disease|parkinson)\b',
        'symptoms': r'\b(pain|fever|cough|fatigue|nausea|headache|dizziness|tremor)\b',
        'measurements': r'\b(\d+\.?\d*)\s*(mg|ml|kg|cm|mmHg|bpm)\b',
        'age_mentions': r'\b(\d+)\s*(?:years?|y/o|yo)\b',
        'exclusions': r'\b(pregnant|nursing|allergy|contraindicated|excluded)\b'
    }

    keywords = {}
    for category, pattern in medical_patterns.items():
        matches = re.findall(pattern, text.lower())
        keywords[category] = len(matches)
        keywords[f'{category}_list'] = matches

    return keywords


def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using sentence transformers"""
    if semantic_model is None:
        return 0.0

    try:
        embeddings = semantic_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    except:
        return 0.0


def enhanced_feature_extraction(patient_text, trial_text):
    """Enhanced feature extraction with context and sentiment analysis"""
    features = {}

    # 1. Medical keyword matching
    patient_keywords = extract_medical_keywords(patient_text)
    trial_keywords = extract_medical_keywords(trial_text)

    features['disease_overlap'] = len(set(patient_keywords.get('diseases_list', [])) &
                                      set(trial_keywords.get('diseases_list', [])))
    features['symptom_overlap'] = len(set(patient_keywords.get('symptoms_list', [])) &
                                      set(trial_keywords.get('symptoms_list', [])))

    # 2. Semantic similarity
    features['semantic_similarity'] = calculate_semantic_similarity(
        patient_text, trial_text)

    # 3. Context Analysis - Negation Detection
    patient_negation = detect_negation(patient_text)
    trial_negation = detect_negation(trial_text)
    features['patient_negation_count'] = patient_negation['negation_count']
    features['trial_negation_count'] = trial_negation['negation_count']

    negated_diseases = set(patient_negation['negated_terms']) & set(
        patient_keywords.get('diseases_list', []))
    features['critical_negations'] = len(negated_diseases)

    # 4. Temporal Context
    patient_temporal = extract_temporal_context(patient_text)
    trial_temporal = extract_temporal_context(trial_text)
    features.update(patient_temporal)
    features['trial_requires_current'] = trial_temporal.get(
        'temporal_current', 0)
    features['patient_has_history'] = patient_temporal.get('temporal_past', 0)
    features['temporal_mismatch'] = abs(
        patient_temporal.get('temporal_current', 0) -
        trial_temporal.get('temporal_current', 0)
    )

    # 5. Severity Analysis
    patient_severity = extract_severity_indicators(patient_text)
    trial_severity = extract_severity_indicators(trial_text)
    features.update(patient_severity)
    features['severity_match'] = 1 if abs(
        patient_severity.get('severity_score', 0) -
        trial_severity.get('severity_score', 0)
    ) <= 2 else 0

    # 6. Sentiment/Clinical Tone Analysis
    patient_sentiment = analyze_clinical_sentiment(patient_text)
    trial_sentiment = analyze_clinical_sentiment(trial_text)
    features['patient_clinical_tone'] = patient_sentiment['clinical_tone']
    features['trial_restrictiveness'] = -trial_sentiment['clinical_tone']
    features['sentiment_alignment'] = 1 - abs(
        patient_sentiment['sentiment_polarity'] -
        trial_sentiment['sentiment_polarity']
    )

    # 7. Exclusion criteria detection
    features['has_exclusions'] = patient_keywords.get('exclusions', 0)
    features['trial_exclusions'] = trial_keywords.get('exclusions', 0)

    # 8. Age extraction and matching
    patient_ages = patient_keywords.get('age_mentions_list', [])
    trial_ages = trial_keywords.get('age_mentions_list', [])

    if patient_ages and trial_ages:
        patient_age = int(patient_ages[0]) if patient_ages[0] else 0
        trial_age_nums = [int(a) for a in trial_ages if a]
        if trial_age_nums:
            features['age_in_range'] = 1 if patient_age >= min(
                trial_age_nums) and patient_age <= max(trial_age_nums) else 0
        else:
            features['age_in_range'] = 0
    else:
        features['age_in_range'] = 0

    # 9. Measurement compatibility
    features['measurement_mentions'] = patient_keywords.get('measurements', 0)

    # 10. Text length ratio
    len_ratio = len(patient_text) / max(len(trial_text), 1)
    features['length_ratio'] = min(
        len_ratio, 1/len_ratio) if len_ratio > 0 else 0

    # 11. Keyword density
    patient_words = set(patient_text.lower().split())
    trial_words = set(trial_text.lower().split())
    features['word_overlap_ratio'] = len(
        patient_words & trial_words) / max(len(patient_words), 1)

    return features


def load_dataset():
    """Load or create the training dataset."""
    dataset_path = os.path.join(
        'Dataset', 'training_examples_augmented_CLEAN.csv')

    if not os.path.exists(dataset_path):
        logger.warning("Dataset not found, creating processed dataset...")
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(dataset_path)}")

    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    df['combined_text'] = df['patient_text'] + " " + df['eligibility_text']

    label_mapping = {
        'not_relevant': 0,
        'excluded': 1,
        'eligible': 2
    }

    df['label_numeric'] = df['relevance_text'].map(label_mapping)
    df = df.dropna(subset=['label_numeric'])
    df['label'] = df['label_numeric']

    logger.info(f"Label distribution:")
    logger.info(f"Not relevant: {(df['label'] == 0).sum()}")
    logger.info(f"Excluded: {(df['label'] == 1).sum()}")
    logger.info(f"Eligible: {(df['label'] == 2).sum()}")

    return df


class Trainer:
    def __init__(self, umls_client: UMLSClient, offline: bool = False):
        self.umls_client = umls_client
        self.offline = offline
        # instantiate model (model class handles its own ml model and vectorizers)
        self.model = TrialMatchModel(umls_client=self.umls_client)

    def preprocess_and_cache(self, df: pd.DataFrame, force_recompute: bool = False):
        """
        For each row in df (expects patient_text, eligibility_text, label),
        compute or load cached processed data and features with real-time progress.
        Returns a list of feature dicts and labels.
        """
        features_list = []
        labels = []
        correct_predictions = 0

        rows = list(df.itertuples(index=False))
        total_samples = len(rows)

        print(
            f"\n🔄 Processing {total_samples} samples with caching and real-time evaluation...")
        print("="*90)

        for i, row in enumerate(rows):
            start_time = time.time()
            patient_text = row.patient_text
            eligibility_text = row.eligibility_text
            label = row.label

            key = _hash_text(patient_text)
            cache_file = cache_path_for_text(key)

            # Load cached results (from previous online runs)
            if cache_file.exists() and not force_recompute:
                processed = load_pickle(cache_file)
                cache_status = "📁 CACHED (from previous online run)"
                logging.info("Using cached UMLS results - no API call needed")
            else:
                # Fresh processing - online or offline mode
                processed = full_preprocess(
                    patient_text, self.umls_client, offline=self.offline)
                save_pickle(processed, cache_file)
                cache_status = "🔧 COMPUTED (fresh API call)" if not self.offline else "🔧 COMPUTED (offline fallback)"

            # Build trial_data (include umls_concepts)
            trial_data = {
                "eligibility_criteria": eligibility_text,
                "conditions": processed["profile"].get("structured_data", {}).get("conditions", []),
                "umls_concepts": processed.get("umls_concepts", [])
            }

            # Extract numerical features using model.extract_features
            processed_data_for_model = {
                "profile": processed["profile"],
                "entities": processed["entities"],
                "umls_concepts": processed["umls_concepts"],
                "original_text": patient_text
            }
            feat_dict = self.model.extract_features(
                processed_data_for_model, trial_data)

            # CRITICAL: Add enhanced features to match testing.py
            enhanced_features = enhanced_feature_extraction(
                patient_text, eligibility_text)
            feat_dict.update(enhanced_features)

            features_list.append(feat_dict)
            labels.append(label)

            # Make prediction if model is trained (for later iterations)
            predicted_label = None
            prediction_status = "⚠️ MODEL NOT TRAINED"

            if self.model.is_trained:
                try:
                    # Quick prediction using current features
                    feature_array = np.array(
                        [[feat_dict.get(k, 0) for k in self.model.feature_names]])

                    # For multi-class, use predict instead of predict_probability
                    X_scaled = self.model.feature_scaler.transform(
                        feature_array)
                    probs = self.model.ml_model.predict_proba(X_scaled)[0]
                    predicted_label = self.model.ml_model.predict(X_scaled)[0]
                    pred_prob = max(probs)

                    # Check if correct
                    if predicted_label == label:
                        correct_predictions += 1
                        prediction_status = f"✅ CORRECT (conf: {pred_prob:.3f})"
                    else:
                        prediction_status = f"❌ WRONG - Pred:{predicted_label}, True:{label} (conf: {pred_prob:.3f})"
                except Exception as e:
                    prediction_status = f"⚠️ PRED ERROR: {str(e)[:30]}"

            # Progress calculations
            current_sample = i + 1
            percent_complete = (current_sample / total_samples) * 100
            remaining = total_samples - current_sample
            processing_time = time.time() - start_time
            current_accuracy = (correct_predictions / current_sample *
                                100) if self.model.is_trained and current_sample > 0 else 0

            # Progress bar visualization
            bar_width = 25
            filled = int((current_sample / total_samples) * bar_width)
            progress_bar = "█" * filled + "░" * (bar_width - filled)

            # Main progress line
            print(f"[{current_sample:3d}/{total_samples:3d}] [{progress_bar}] {percent_complete:5.1f}% | "
                  f"Remaining: {remaining:3d} | {processing_time:.2f}s | {cache_status}")

            # Prediction status line
            print(f"    🎯 {prediction_status}")

            # Running accuracy if model is trained
            if self.model.is_trained:
                print(
                    f"    📊 Running Accuracy: {current_accuracy:.1f}% ({correct_predictions}/{current_sample})")

            # Show detailed extraction info for first few samples
            if i < 5 or (i + 1) % 20 == 0:
                extracted_entities = processed.get("entities", {})
                conditions = extracted_entities.get("conditions", [])
                medications = extracted_entities.get("medications", [])
                age = extracted_entities.get("age", "N/A")
                gender = extracted_entities.get("gender", "N/A")

                print(f"    📋 Extracted - Age: {age}, Gender: {gender}")
                print(
                    f"        Conditions ({len(conditions)}): {', '.join(conditions[:3])}{'...' if len(conditions) > 3 else ''}")
                print(
                    f"        Medications ({len(medications)}): {', '.join(medications[:3])}{'...' if len(medications) > 3 else ''}")

                # Show key features
                key_features = [
                    'condition_match', 'medication_overlap', 'age_match', 'text_similarity']
                feature_summary = {k: feat_dict.get(
                    k, 0) for k in key_features}
                print(f"    🔢 Key Features: {feature_summary}")

            print()  # Empty line for readability

        print("="*90)
        print(f"✅ Preprocessing Complete!")
        print(f"📊 Cache Status: {sum(1 for row in rows if cache_path_for_text(_hash_text(row.patient_text)).exists())} cached, {total_samples - sum(1 for row in rows if cache_path_for_text(_hash_text(row.patient_text)).exists())} computed")

        if self.model.is_trained:
            print(
                f"🎯 Final Processing Accuracy: {current_accuracy:.1f}% ({correct_predictions}/{total_samples})")

        return features_list, labels

    def train(self, df: pd.DataFrame, force_recompute: bool = False, class_weights: dict = None):
        """BALANCED FIX: Moderate class rebalancing for better generalization"""
        # 1. Preprocess and cache
        X_dicts, y = self.preprocess_and_cache(
            df, force_recompute=force_recompute)

        if len(X_dicts) == 0:
            raise RuntimeError("No features extracted. Check dataset.")

        # Fit TF-IDF vectorizer on all text data
        all_texts = []
        for _, row in df.iterrows():
            all_texts.extend(
                [row.patient_text.lower(), row.eligibility_text.lower()])
        self.model.fit_tfidf_vectorizer(all_texts)

        # Convert to numpy arrays
        feature_names = sorted(X_dicts[0].keys())
        X = np.array([[d.get(fn, 0) for fn in feature_names] for d in X_dicts])
        y = np.array(y)

        logging.info(f"Feature matrix shape: {X.shape}, labels: {y.shape}")
        logging.info(f"Label distribution: {np.bincount(y)}")

        unique_labels = np.unique(y)
        is_multiclass = len(unique_labels) > 2

        if is_multiclass:
            logging.info("Detected multi-class classification (0, 1, 2)")
            logging.info(
                f"Label counts: {dict(zip(*np.unique(y, return_counts=True)))}")

            # BALANCED FIX: Moderate reweighting (2x instead of 5x)
            from sklearn.utils.class_weight import compute_class_weight

            class_counts = np.bincount(y)
            total_samples = len(y)

            # BALANCED: Moderate boost for minority classes
            class_weights = {}
            for i in range(len(unique_labels)):
                base_weight = total_samples / \
                    (class_counts[i] * len(unique_labels))
                # CRITICAL FIX: 2x multiplier (not 5x) for better balance
                if i in [1, 2]:
                    class_weights[i] = base_weight * 2.0  # REDUCED from 5.0
                else:
                    class_weights[i] = base_weight

            logging.info(f"BALANCED class weights: {class_weights}")

            # BALANCED: XGBoost with moderate class balancing
            from xgboost import XGBClassifier
            self.model.ml_model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,  # INCREASED regularization
                reg_lambda=0.05,
                gamma=0.1,  # ADDED back to prevent overfitting
                min_child_weight=2,  # INCREASED for stability
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=3,
                use_label_encoder=False,
                random_state=42,
                verbosity=1,
                n_jobs=-1,
                tree_method='hist'
            )

            # Set sample weights with moderate boost
            sample_weights = np.array([class_weights[label] for label in y])
            logging.info(
                f"Sample weights range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")

        else:
            raise ValueError(
                f"Dataset must have 3 classes (0,1,2)! Found: {unique_labels}")

        # MODIFIED: Use XGBoost only (skip algorithm comparison)
        logging.info(
            "Training with XGBoost only (skipping algorithm comparison)...")

        # Use the pre-configured XGBoost model directly
        best_model_name = 'XGBoost'

        # Train on full dataset with cross-validation for verification
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            weights_train = sample_weights[train_idx]

            X_train_scaled = self.model.feature_scaler.fit_transform(
                X_train_fold)
            X_val_scaled = self.model.feature_scaler.transform(X_val_fold)

            self.model.ml_model.fit(
                X_train_scaled, y_train_fold, sample_weight=weights_train)

            val_preds = self.model.ml_model.predict(X_val_scaled)
            from sklearn.metrics import f1_score
            val_f1 = f1_score(y_val_fold, val_preds, average="macro")
            cv_scores.append(val_f1)

            logging.info(f"  Fold {fold+1}: F1_macro={val_f1:.3f}")

        best_score = np.mean(cv_scores)
        logging.info(
            f"XGBoost - CV F1_macro: {best_score:.4f} (+/- {np.std(cv_scores):.4f})")

        # Final training on full dataset with sample weights
        X_scaled = self.model.feature_scaler.fit_transform(X)

        # CRITICAL: Train final model with sample weights
        if best_model_name in ['XGBoost', 'RandomForest', 'GradientBoosting']:
            self.model.ml_model.fit(X_scaled, y, sample_weight=sample_weights)
        else:
            self.model.ml_model.fit(X_scaled, y)

        self.model.is_trained = True
        self.model.feature_names = feature_names

        # CRITICAL: After training, verify model is truly multi-class
        if is_multiclass:
            # Test prediction on one sample to verify setup
            test_probs = self.model.ml_model.predict_proba(X_scaled[:1])
            test_pred = self.model.ml_model.predict(X_scaled[:1])

            logging.info(
                f"VERIFICATION - Test prediction shape: {test_probs.shape}")
            logging.info(f"VERIFICATION - Test probabilities: {test_probs}")
            logging.info(f"VERIFICATION - Test prediction: {test_pred}")

            if test_probs.shape[1] != 3:
                raise RuntimeError(
                    f"CRITICAL ERROR: Model is not 3-class! Shape: {test_probs.shape}. Expected: (1, 3)")

            # Verify predictions span all 3 classes
            all_preds = self.model.ml_model.predict(X_scaled)
            pred_classes = set(all_preds)
            if len(pred_classes) == 1:
                logging.warning(
                    f"WARNING: Model only predicts class {pred_classes}. This indicates broken training!")

            logging.info(
                f"SUCCESS: Model properly predicts classes: {sorted(pred_classes)}")

        # Final evaluation with detailed debugging
        if is_multiclass:
            probs = self.model.ml_model.predict_proba(X_scaled)
            preds = self.model.ml_model.predict(X_scaled)

            # Debug multi-class predictions
            logging.info(f"Prediction distribution: {np.bincount(preds)}")
            logging.info(
                f"Probability ranges: {probs.min(axis=0)} to {probs.max(axis=0)}")
        else:
            probs = self.model.predict_probability(X)
            preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")

        try:
            if is_multiclass:
                from sklearn.metrics import roc_auc_score
                roc = roc_auc_score(
                    y, probs, multi_class="ovr", average="weighted")
            else:
                roc = roc_auc_score(y, probs) if len(
                    np.unique(y)) > 1 else float("nan")
        except Exception as e:
            logging.error(f"ROC-AUC calculation failed: {e}")
            roc = float("nan")

        logging.info(
            f"Final Training - Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc}")

        # Show detailed results per class
        from sklearn.metrics import classification_report
        print("\nDetailed Classification Report:")
        if is_multiclass:
            target_names = [
                'No Match (0)', 'Partial Match (1)', 'Good Match (2)']
        else:
            target_names = ['No Match (0)', 'Match (1)']
        print(classification_report(y, preds, target_names=target_names))

        # Create detailed results focusing on dataset accuracy
        detailed_results = []
        for i in range(len(df)):
            patient_text = df.iloc[i].patient_text
            trial_text = df.iloc[i].eligibility_text
            true_label = df.iloc[i].label

            pred_label = preds[i]
            if is_multiclass:
                pred_prob = max(probs[i])
            else:
                pred_prob = probs[i]

            is_correct = pred_label == true_label

            detailed_results.append({
                "index": i,
                "patient_text": patient_text,
                "trial_eligibility_text": trial_text,
                "label": int(true_label),
                "pred_label": int(pred_label),
                "pred_probability": float(pred_prob),
                "correct": bool(is_correct),
                "feature_contributions": {}  # Can be populated if needed
            })

        # Save outputs
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_dir = MODELS_DIR / f"dataset_training_{timestamp}"
        out_dir.mkdir(exist_ok=True)

        metrics = {
            "timestamp": timestamp,
            "n_examples": int(len(df)),
            "n_classes": int(len(unique_labels)),
            "label_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "best_algorithm": best_model_name,
            "validation_accuracy": float(best_score),
            "final_accuracy": float(acc),
            "f1": float(f1),
            "roc_auc": float(roc) if not np.isnan(roc) else None,
            "correct_predictions": int(sum(preds == y)),
            "total_predictions": int(len(preds))
        }

        # Save files
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(out_dir / "detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        pd.DataFrame(detailed_results).to_csv(
            out_dir / "training_examples.csv", index=False)

        # Save trained model
        model_file = out_dir / "model.pkl"
        self.model.save_model(model_file)

        logging.info(f"Saved outputs to {out_dir}")

        return metrics, detailed_results


def train_model(args):
    """Train the trial matching model."""
    try:
        # Load dataset
        df = load_dataset()
        logger.info(f"Loaded dataset with {len(df)} samples")

        # Initialize UMLS client
        umls_client = UMLSClient(api_key="YOUR_API_KEY_HERE")

        if args.use_deep_learning and dl_available:
            logger.info("🚀 Using HYBRID Deep Learning + XGBoost approach")

            # Prepare data for DL model
            trainer = Trainer(umls_client=umls_client, offline=args.offline)
            X_dicts, y = trainer.preprocess_and_cache(
                df, force_recompute=args.force_recompute)

            # Extract structured features
            feature_names = sorted(X_dicts[0].keys())
            X_structured = np.array(
                [[d.get(fn, 0) for fn in feature_names] for d in X_dicts])

            # Split data - FIXED syntax error
            train_idx, val_idx = train_test_split(
                range(len(df)), test_size=0.2, stratify=y, random_state=42
            )

            train_data = {
                'patient_texts': df.iloc[train_idx]['patient_text'].tolist(),
                'trial_texts': df.iloc[train_idx]['eligibility_text'].tolist(),
                'structured_features': X_structured[train_idx],
                'labels': y[train_idx]
            }

            val_data = {
                'patient_texts': df.iloc[val_idx]['patient_text'].tolist(),
                'trial_texts': df.iloc[val_idx]['eligibility_text'].tolist(),
                'structured_features': X_structured[val_idx],
                'labels': y[val_idx]
            }

            # Train DL model
            dl_trainer = DeepLearningTrainer(
                num_structured_features=len(feature_names))
            best_acc = dl_trainer.train(
                train_data, val_data, epochs=args.epochs)

            logger.info(
                f"🎯 Best Deep Learning Validation Accuracy: {best_acc:.4f}")

        else:
            # Standard XGBoost training
            logger.info("Using standard XGBoost approach")
            trainer = Trainer(umls_client=umls_client, offline=args.offline)

            # CRITICAL: Use scale_pos_weight for imbalanced data
            from collections import Counter
            label_counts = Counter(df['label'])

            # Calculate class weights (inverse frequency)
            total = sum(label_counts.values())
            class_weights = {
                cls: total / (len(label_counts) * count)
                for cls, count in label_counts.items()
            }

            logger.info(f"\n🔧 Applying class weights: {class_weights}")

            metrics, detailed_results = trainer.train(
                df,
                force_recompute=args.force_recompute,
                class_weights=class_weights  # Pass weights to trainer
            )

            logger.info("Training complete!")
            logger.info(f"Final metrics: {metrics}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train clinical trial matching model")
    parser.add_argument("--dataset", type=str,
                        default="c:/AyuSynapse/Dataset/training_examples_augmented_CLEAN.csv",
                        help="Path to training dataset CSV")
    parser.add_argument("--offline", action="store_true",
                        help="Use offline mode (no UMLS API calls)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recompute all cached data")
    parser.add_argument("--use-deep-learning", action="store_true",
                        help="Use hybrid DL+XGBoost model (experimental)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs for DL model")

    args = parser.parse_args()

    # Load and preprocess data
    df = load_dataset()
    logger.info(f"Loaded dataset with {len(df)} samples")

    # CRITICAL: Check class distribution
    logger.info("\n📊 Training Data Distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        logger.info(
            f"   Class {label}: {count} samples ({count/len(df)*100:.1f}%)")

    # Initialize UMLS client
    umls_client = UMLSClient(api_key="YOUR_API_KEY_HERE")

    # Standard XGBoost training with FIXED class weights
    logger.info("Using standard XGBoost approach with class balancing")
    trainer = Trainer(umls_client=umls_client, offline=args.offline)

    # CRITICAL FIX: Use scale_pos_weight for imbalanced data
    from collections import Counter
    label_counts = Counter(df['label'])

    # Calculate class weights (inverse frequency)
    total = sum(label_counts.values())
    class_weights = {
        cls: total / (len(label_counts) * count)
        for cls, count in label_counts.items()
    }

    logger.info(f"\n🔧 Applying class weights: {class_weights}")

    # Train with balanced weights
    metrics, detailed_results = trainer.train(
        df,
        force_recompute=args.force_recompute,
        class_weights=class_weights  # Pass weights to trainer
    )

    logger.info("Training complete!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
