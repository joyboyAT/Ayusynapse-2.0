import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# === Paths ===
json_path = r"C:\AyuSynapse\models\20251102_225727\detailed_results.json"
patient_embeddings = r"C:\AyuSynapse\models\20251102_225727\patient_embeddings.csv"
trial_embeddings = r"C:\AyuSynapse\models\20251102_225727\trial_embeddings.csv"

# === Load JSON data for labels ===
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON to DataFrame
records = []
for entry in data:
    feats = entry.get("feature_contributions", {})
    records.append({
        "patient_id": hash(entry["patient_text"]) % (10**8),
        "trial_id": hash(entry["trial_eligibility_text"]) % (10**8),
        **feats,
        "label": entry.get("label", 0),
        "pred_score": entry.get("pred_score", 0),
        "pred_probability": entry.get("pred_probability", 0)
    })

df = pd.DataFrame(records)
print(f"✅ Loaded {len(df)} entries from JSON")

# === Load embeddings (optional for consistency check) ===
patients = pd.read_csv(patient_embeddings)
trials = pd.read_csv(trial_embeddings)

print(f"Patients shape: {patients.shape}")
print(f"Trials shape: {trials.shape}")

# === Feature matrix and labels ===
feature_cols = ["umls_match", "condition_match",
                "entity_count", "age_match", "ml_score"]
X = df[feature_cols]
y = df["label"]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train RandomForest ===
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("\n=== Evaluation Report ===")
print(classification_report(y_test, y_pred, digits=3))

# === Save model ===
model_path = os.path.join(os.path.dirname(__file__), "final_classifier.pkl")
joblib.dump(clf, model_path)
print(f"\n✅ Model saved to {model_path}")
