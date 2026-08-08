import pandas as pd
from pathlib import Path

CSV_PATH = Path("dataset/training_features.csv")

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("TRAINING DATASET INSPECTION")
print("=" * 70)

print("\nShape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nColumn names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:03d}. {column}")

print("\nMissing values:")
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("None")

else:
    print(missing.to_string())

print("\nBasic statistics:")
print(df.describe().T.to_string())

print("\nUnique values for important fields:")

for column in [
    "plot_area",
    "net_area",
    "wall_depth",
    "bedroom_exists",
    "bathroom_exists",
    "kitchen_exists",
    "living_exists",
    "parking_exists",
    "garden_exists",
]:
    if column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
