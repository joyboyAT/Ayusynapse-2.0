import json
import pandas as pd
import os

# --- Configuration ---
input_file = r"C:\AyuSynapse\models\20251102_225727\detailed_results.json"
output_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load JSON ---
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📦 Loaded {len(data)} total entries")

# --- Collect all possible feature keys ---
all_features = set()
for entry in data:
    feats = entry.get("feature_contributions", {})
    all_features.update(feats.keys())
all_features = sorted(list(all_features))
print(f"🔍 Found {len(all_features)} unique features: {all_features}")

# --- Build rows ---
patient_rows, trial_rows = [], []

for entry in data:
    features = entry.get("feature_contributions", {})
    # fill missing features with 0
    row_features = {k: features.get(k, 0) for k in all_features}

    patient_id = hash(entry.get("patient_text", "")) % (10**8)
    trial_id = hash(entry.get("trial_eligibility_text", "")) % (10**8)

    patient_rows.append({
        "topic_id": patient_id,
        "label": entry.get("label", None),
        "pred_score": entry.get("pred_score", None),
        "pred_probability": entry.get("pred_probability", None),
        **row_features
    })

    trial_rows.append({
        "trial_id": trial_id,
        "label": entry.get("label", None),
        "pred_score": entry.get("pred_score", None),
        "pred_probability": entry.get("pred_probability", None),
        **row_features
    })

# --- Convert to DataFrames ---
patient_df = pd.DataFrame(patient_rows).drop_duplicates(subset=["topic_id"])
trial_df = pd.DataFrame(trial_rows).drop_duplicates(subset=["trial_id"])

# --- Save ---
patient_path = os.path.join(output_dir, "patient_embeddings.csv")
trial_path = os.path.join(output_dir, "trial_embeddings.csv")

patient_df.to_csv(patient_path, index=False)
trial_df.to_csv(trial_path, index=False)

print(f"✅ Generated patient_embeddings.csv ({len(patient_df)} rows)")
print(f"✅ Generated trial_embeddings.csv ({len(trial_df)} rows)")
print(f"📁 Saved to:\n  {patient_path}\n  {trial_path}")
