import pickle
from pathlib import Path

# Dataset Path
DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "ResPlan.pkl"

print("=" * 60)
print("Loading ResPlan Dataset")
print("=" * 60)

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

print(f"\n✅ Total Plans : {len(data)}")

# First house
plan = data[0]

print("\nPlan Type :", type(plan))

print("\nAvailable Keys")

print("-" * 40)

for key in plan.keys():
    print(key)

print("-" * 40)

print("\nPlan ID :", plan.get("id"))