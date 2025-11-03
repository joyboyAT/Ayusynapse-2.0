"""
Split training_examples.csv into train and test sets (80/20)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load dataset
base_dir = Path(__file__).parent
data_path = base_dir / "training_examples.csv"
df = pd.read_csv(data_path)

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Save both files
train_df.to_csv(base_dir / "train_data.csv", index=False)
test_df.to_csv(base_dir / "test_data.csv", index=False)

print(f"✅ Dataset split completed:")
print(f"Training set: {len(train_df)} rows")
print(f"Testing set: {len(test_df)} rows")
print(f"Files saved as 'train_data.csv' and 'test_data.csv'")
