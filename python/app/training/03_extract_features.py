import pickle
import csv
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "ResPlan.pkl"
OUTPUT_PATH = Path(__file__).resolve().parents[3] / "dataset" / "training_features.csv"

ROOM_TYPES = [
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

def geometry_features(geometry):
    if geometry is None or geometry.is_empty:
        return {
            "exists": 0,
            "area": 0.0,
            "min_x": 0.0,
            "min_y": 0.0,
            "max_x": 0.0,
            "max_y": 0.0,
            "width": 0.0,
            "height": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "parts": 0,
        }

    min_x, min_y, max_x, max_y = geometry.bounds

    return {
        "exists": 1,
        "area": float(geometry.area),
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "width": float(max_x - min_x),
        "height": float(max_y - min_y),
        "centroid_x": float(geometry.centroid.x),
        "centroid_y": float(geometry.centroid.y),
        "parts": len(geometry.geoms) if hasattr(geometry, "geoms") else 1,
    }


print("=" * 70)
print("RESPLAN FEATURE EXTRACTION")
print("=" * 70)

print("\nLoading dataset...")

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} floor plans.")

rows = []

for index, plan in enumerate(data):

    if index % 500 == 0:
        print(f"Processing {index}/{len(data)}...")

    row = {
        "plan_id": plan.get("id"),
        "plot_area": float(plan.get("area", 0) or 0),
        "net_area": float(plan.get("net_area", 0) or 0),
        "wall_depth": float(plan.get("wall_depth", 0) or 0),
    }

    for room in ROOM_TYPES:

        features = geometry_features(plan.get(room))

        prefix = room

        row[f"{prefix}_exists"] = features["exists"]
        row[f"{prefix}_area"] = features["area"]
        row[f"{prefix}_min_x"] = features["min_x"]
        row[f"{prefix}_min_y"] = features["min_y"]
        row[f"{prefix}_max_x"] = features["max_x"]
        row[f"{prefix}_max_y"] = features["max_y"]
        row[f"{prefix}_width"] = features["width"]
        row[f"{prefix}_height"] = features["height"]
        row[f"{prefix}_centroid_x"] = features["centroid_x"]
        row[f"{prefix}_centroid_y"] = features["centroid_y"]
        row[f"{prefix}_parts"] = features["parts"]

    rows.append(row)

fieldnames = list(rows[0].keys())

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nPlans processed : {len(rows)}")
print(f"Features created: {len(fieldnames)}")
print(f"Output file     : {OUTPUT_PATH}")
print("\nNext step: inspect the generated CSV before training any model.")
