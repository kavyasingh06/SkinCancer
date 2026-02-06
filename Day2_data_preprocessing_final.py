# ===========================================
# Day 1: Dataset Setup - Skin Cancer Detection (Corrected)
# ===========================================

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --------------------------
# Load metadata CSV
# --------------------------
metadata_csv = "data/HAM10000_metadata.csv"
metadata = pd.read_csv(metadata_csv)

# --------------------------
# Check unique dx values
# --------------------------
print("Unique lesion types in dataset:")
print(metadata['dx'].unique())

# --------------------------
# Map labels: Malignant vs Benign
# --------------------------
# Change 'mel' to the exact string for malignant in your CSV
malignant_labels = ['mel']  # Example: update based on the print above
metadata['label'] = metadata['dx'].apply(lambda x: 1 if x.lower() in malignant_labels else 0)

# --------------------------
# Create full image path
# --------------------------
metadata['image_path'] = metadata['image_id'].apply(
    lambda x: os.path.join("data/HAM10000_images", x + ".jpg")
)

# --------------------------
# Check missing images
# --------------------------
missing_images = metadata[~metadata['image_path'].apply(os.path.exists)]
print("Number of missing images:", len(missing_images))
if len(missing_images) > 0:
    print(missing_images.head())

# Keep only existing images
metadata = metadata[metadata['image_path'].apply(os.path.exists)]

# --------------------------
# Verify label distribution
# --------------------------
print("Label counts after mapping:")
print(metadata['label'].value_counts())

# --------------------------
# Split into train / validation
# --------------------------
train_df, val_df = train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata['label'],
    random_state=42
)

print("Training label counts:", train_df['label'].value_counts())
print("Validation label counts:", val_df['label'].value_counts())

# --------------------------
# Save CSVs
# --------------------------
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
print("✅ train.csv and val.csv saved in data folder")
