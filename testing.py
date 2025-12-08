from trial_match_model import TrialMatchModel
import os
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import enhanced feature extraction from training script
from train_trial_matcher import enhanced_feature_extraction

print("🔄 Loading semantic similarity model...")
try:
    from sentence_transformers import SentenceTransformer
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    semantic_model = None
    print("⚠️ Semantic model not available")


def find_latest_model():
    """Find the most recently trained model"""
    models_dir = Path("c:/AyuSynapse/models")

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return None

    # Find all dataset_training_* directories
    training_dirs = list(models_dir.glob("dataset_training_*"))

    if not training_dirs:
        print("❌ No trained models found")
        return None

    # Sort by timestamp (directory name contains timestamp)
    latest_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
    model_path = latest_dir / "model.pkl"

    if model_path.exists():
        return model_path
    else:
        print(f"❌ Model file not found in {latest_dir}")
        return None


# === Find Latest Model Automatically ===
model_path = find_latest_model()
if model_path is None:
    print("❌ ERROR: No trained model found!")
    print("Please run: python train_trial_matcher.py --offline --epochs 10")
    exit(1)

test_path = r"C:\AyuSynapse\Dataset\test_data.csv"

# QUICK FIX: After loading model
print(f"📦 Loading LATEST trained model from: {model_path}")
model = TrialMatchModel()
model.load_model(model_path)

# Add threshold adjustment for imbalanced model
APPLY_THRESHOLD_BOOST = True  # Set to False to disable

# Check if test data exists
if not Path(test_path).exists():
    print(f"❌ Test data not found: {test_path}")
    print("Creating sample test data...")

    # Create sample test data
    sample_test_data = pd.DataFrame({
        'patient_text': [
            "45-year-old male with anaplastic astrocytoma, history of radiation therapy",
            "32-year-old female with severe headache, intraventricular hemorrhage on CT",
            "48-year-old male with aortic stenosis, bicuspid aortic valve, hypertension"
        ],
        'eligibility_text': [
            "Adult patients with recurrent gliomas, prior radiation therapy required",
            "Patients with cerebrovascular disease, age 18-65",
            "Severe aortic stenosis, candidates for valve replacement"
        ],
        'title': [
            "Glioma Treatment Trial",
            "Stroke Prevention Study",
            "Aortic Valve Replacement Study"
        ],
        'label': [2, 1, 2]  # 0=not_relevant, 1=excluded, 2=eligible
    })

    os.makedirs(Path(test_path).parent, exist_ok=True)
    sample_test_data.to_csv(test_path, index=False)
    print(f"✅ Created sample test data with {len(sample_test_data)} samples")

print("📂 Loading test data...")
test_df = pd.read_csv(test_path)

if test_df.empty or 'patient_text' not in test_df.columns:
    print("❌ Test data is empty or missing required columns!")
    exit(1)

print(f"✅ Loaded {len(test_df)} test samples.")

# CRITICAL FIX: Use the EXACT same feature extraction as training
print("\n🧩 Processing test data with real-time feedback...")
print("="*80)

test_features_list = []
predictions = []
correct_count = 0

for i, row in test_df.iterrows():
    start_time = time.time()

    patient_text = row['patient_text']
    trial_text = row['eligibility_text']
    true_label = row.get('label', -1)

    # CRITICAL: Use enhanced_feature_extraction from training script
    features = enhanced_feature_extraction(patient_text, trial_text)

    # CRITICAL: Only keep features that match the trained model
    aligned_features = {}
    for feature_name in model.feature_names:
        aligned_features[feature_name] = features.get(feature_name, 0)

    test_features_list.append(aligned_features)

    # Make prediction
    feature_array = np.array([[aligned_features.get(k, 0)
                             for k in model.feature_names]])
    X_scaled = model.feature_scaler.transform(feature_array)

    # QUICK FIX: Adjust prediction thresholds for imbalanced model
    if hasattr(model.ml_model, 'predict_proba'):
        probs = model.ml_model.predict_proba(X_scaled)[0]

        # AGGRESSIVE BOOST for minority classes
        adjusted_probs = probs.copy()
        adjusted_probs[1] *= 5.0  # 5x boost for class 1 (excluded)
        adjusted_probs[2] *= 5.0  # 5x boost for class 2 (eligible)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()

        pred_label = np.argmax(adjusted_probs)
        pred_prob = max(adjusted_probs)
    else:
        pred_label = model.ml_model.predict(X_scaled)[0]
        pred_prob = max(probs)

    predictions.append({
        'patient_text': patient_text[:100] + '...',
        'trial_text': trial_text[:100] + '...',
        'true_label': int(true_label) if true_label != -1 else None,
        'predicted_label': int(pred_label),
        'confidence': float(pred_prob),
        'probabilities': {
            'not_relevant': float(probs[0]),
            'excluded': float(probs[1]),
            'eligible': float(probs[2])
        }
    })

    # Check correctness
    is_correct = (pred_label == true_label) if true_label != -1 else None
    if is_correct:
        correct_count += 1

    # Progress reporting
    current = i + 1
    total = len(test_df)
    percent = (current / total) * 100
    processing_time = time.time() - start_time

    bar_width = 20
    filled = int((current / total) * bar_width)
    progress_bar = "█" * filled + "░" * (bar_width - filled)

    status = "✅ CORRECT" if is_correct else (
        "❌ WRONG" if is_correct is False else "⚠️ UNKNOWN")

    print(
        f"Sample {current:3d}/{total:3d} [{progress_bar}] {percent:5.1f}% | {processing_time:.2f}s")
    print(
        f"    🎯 Prediction: {pred_label} (confidence: {pred_prob:.3f}) | {status}")

    if true_label != -1:
        running_acc = (correct_count / current) * 100
        print(
            f"    📊 Running Accuracy: {running_acc:.1f}% ({correct_count}/{current})")

    print()

print("="*80)
print("✅ Testing Complete!")

# Final metrics
if 'label' in test_df.columns:
    y_true = test_df['label'].values
    y_pred = np.array([p['predicted_label'] for p in predictions])

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   F1 Score (macro): {f1:.3f}")
    print(f"   Correct: {correct_count}/{len(test_df)}")

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Not Relevant', 'Excluded', 'Eligible']))

    print("\n🎯 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Save predictions
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_file = f"test_predictions_{timestamp}.csv"
pd.DataFrame(predictions).to_csv(output_file, index=False)
print(f"\n💾 Predictions saved to: {output_file}")
