import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("dataset/training_features.csv")
OUTPUT = Path("dataset/training_clean.csv")

print("=" * 70)
print("RESPLAN DATA CLEANING")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"\nOriginal rows    : {len(df)}")
print(f"Original columns : {len(df.columns)}")

# ------------------------------------------------------------
# 1. Remove duplicate plans
# ------------------------------------------------------------

before = len(df)
df = df.drop_duplicates(subset=["plan_id"])
print(f"\nDuplicate plans removed : {before - len(df)}")

# ------------------------------------------------------------
# 2. Detect invalid numeric values
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(include=[np.number]).columns

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

print(
    "\nInfinite values converted to NaN."
)

# ------------------------------------------------------------
# 3. Plot area validation
# ------------------------------------------------------------

invalid_plot = (
    df["plot_area"].isna()
    | (df["plot_area"] <= 0)
)

print(
    f"Invalid plot areas      : {invalid_plot.sum()}"
)

df = df[~invalid_plot].copy()

# ------------------------------------------------------------
# 4. Net area investigation
# ------------------------------------------------------------

print("\nNet area statistics BEFORE cleaning:")
print(df["net_area"].describe().to_string())

# ResPlan geometry is normalized around a 256-style coordinate
# system. Extremely large net_area values are treated as
# corrupted/outlier values rather than blindly used for training.

valid_net = (
    df["net_area"].notna()
    & (df["net_area"] > 0)
    & (df["net_area"] < 100000)
)

invalid_net = (~valid_net).sum()

print(
    f"\nSuspicious net_area values : {invalid_net}"
)

# Replace suspicious values with NaN.
df.loc[~valid_net, "net_area"] = np.nan

# ------------------------------------------------------------
# 5. Create safer normalized room features
# ------------------------------------------------------------

ROOMS = [
    "bedroom",
    "bathroom",
    "kitchen",
    "living",
    "balcony",
    "garden",
    "parking",
    "pool",
    "inner",
]

# Coordinates in the dataset are approximately based around
# a 256 x 256 normalized plan space.

for room in ROOMS:

    exists = f"{room}_exists"

    if exists not in df.columns:
        continue

    # Area relative to inner plan area
    area_col = f"{room}_area"

    if area_col in df.columns:
        df[f"{room}_area_ratio"] = (
            df[area_col] /
            df["inner_area"].replace(0, np.nan)
        )

    # Position normalized to the inner plan
    for axis in ["x", "y"]:

        centroid = f"{room}_centroid_{axis}"

        if centroid in df.columns:
            df[f"{room}_centroid_{axis}_norm"] = (
                df[centroid] / 256.0
            )

    # Dimensions normalized
    for dimension in ["width", "height"]:

        col = f"{room}_{dimension}"

        if col in df.columns:
            df[f"{room}_{dimension}_norm"] = (
                df[col] / 256.0
            )

# ------------------------------------------------------------
# 6. Create requirement-style features
# ------------------------------------------------------------

# Important: these are derived from the available semantic
# geometry labels. They will later become model inputs.

df["bedroom_required"] = df["bedroom_exists"].astype(int)
df["bathroom_required"] = df["bathroom_exists"].astype(int)
df["kitchen_required"] = df["kitchen_exists"].astype(int)
df["living_required"] = df["living_exists"].astype(int)
df["balcony_required"] = df["balcony_exists"].astype(int)
df["parking_required"] = df["parking_exists"].astype(int)
df["garden_required"] = df["garden_exists"].astype(int)
df["pool_required"] = df["pool_exists"].astype(int)

# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(f"\nFinal rows    : {len(df)}")
print(f"Final columns : {len(df.columns)}")
print(f"Output        : {OUTPUT}")

print("\nRequirement feature summary:")

for col in [
    "bedroom_required",
    "bathroom_required",
    "kitchen_required",
    "living_required",
    "balcony_required",
    "parking_required",
    "garden_required",
    "pool_required",
]:
    if col in df.columns:
        print(
            f"{col:25s}: "
            f"{int(df[col].sum())} / {len(df)}"
        )

print("\nDone.")
