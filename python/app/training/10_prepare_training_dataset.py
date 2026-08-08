import json
import random
from pathlib import Path

INPUT = Path("dataset/training_samples.json")
OUTPUT_DIR = Path("dataset/training")

TRAIN_OUT = OUTPUT_DIR / "train.json"
VAL_OUT = OUTPUT_DIR / "val.json"
TEST_OUT = OUTPUT_DIR / "test.json"

SEED = 42

print("=" * 70)
print("FINAL TRAINING DATASET PREPARATION")
print("=" * 70)

print("\nLoading training samples...")

with open(INPUT, "r", encoding="utf-8") as f:
    samples = json.load(f)

print(f"Loaded samples : {len(samples)}")

# ------------------------------------------------------------
# Remove accidental duplicate plan IDs
# ------------------------------------------------------------

unique = {}

for sample in samples:
    plan_id = str(sample.get("plan_id"))

    if plan_id not in unique:
        unique[plan_id] = sample

samples = list(unique.values())

print(f"After deduplication : {len(samples)}")

# ------------------------------------------------------------
# Validate samples
# ------------------------------------------------------------

valid_samples = []
invalid_samples = []

for sample in samples:

    input_data = sample.get("input", {})
    requirements = input_data.get("requirements", {})
    target = sample.get("target", {})

    required_fields = [
        "plot_area",
        "wall_depth",
        "requirements",
    ]

    valid = all(
        field in input_data
        for field in required_fields
    )

    valid = valid and len(target.get("rooms", [])) > 0
    valid = valid and len(target.get("edges", [])) > 0

    if valid:
        valid_samples.append(sample)
    else:
        invalid_samples.append(sample)

print(f"Valid samples   : {len(valid_samples)}")
print(f"Invalid samples : {len(invalid_samples)}")

# ------------------------------------------------------------
# Normalize model-input values
# ------------------------------------------------------------

for sample in valid_samples:

    inp = sample["input"]

    plot_area = float(inp.get("plot_area", 0) or 0)
    wall_depth = float(inp.get("wall_depth", 0) or 0)

    # Keep original values
    inp["plot_area"] = plot_area
    inp["wall_depth"] = wall_depth

    # Normalized scalar features
    inp["plot_area_norm"] = min(plot_area / 1000.0, 10.0)
    inp["wall_depth_norm"] = min(wall_depth / 20.0, 10.0)

    # Do NOT use original net_area blindly.
    # Only use it when it is a reasonable positive value.
    net_area = inp.get("net_area")

    if net_area is None:
        inp["net_area"] = None
        inp["net_area_norm"] = None

    else:
        try:
            net_area = float(net_area)

            if 0 < net_area < 100000:
                inp["net_area"] = net_area
                inp["net_area_norm"] = min(
                    net_area / 1000.0,
                    100.0
                )
            else:
                inp["net_area"] = None
                inp["net_area_norm"] = None

        except (TypeError, ValueError):
            inp["net_area"] = None
            inp["net_area_norm"] = None

    # Ensure requirement counts are integers
    for key in [
        "bedrooms",
        "bathrooms",
        "kitchens",
        "living_rooms",
        "balconies",
        "storages",
        "stairs",
    ]:
        inp["requirements"][key] = int(
            inp["requirements"].get(key, 0) or 0
        )

# ------------------------------------------------------------
# Normalize target geometry
# ------------------------------------------------------------

for sample in valid_samples:

    for room in sample["target"]["rooms"]:

        centroid = room.get("centroid")

        if centroid and len(centroid) == 2:

            room["centroid_norm"] = [
                float(centroid[0]) / 256.0,
                float(centroid[1]) / 256.0,
            ]

        bounds = room.get("bounds")

        if bounds and len(bounds) == 4:

            room["bounds_norm"] = [
                float(bounds[0]) / 256.0,
                float(bounds[1]) / 256.0,
                float(bounds[2]) / 256.0,
                float(bounds[3]) / 256.0,
            ]

# ------------------------------------------------------------
# Shuffle
# ------------------------------------------------------------

random.seed(SEED)

random.shuffle(valid_samples)

# ------------------------------------------------------------
# Split
# ------------------------------------------------------------

total = len(valid_samples)

train_end = int(total * 0.80)
val_end = int(total * 0.90)

train = valid_samples[:train_end]
val = valid_samples[train_end:val_end]
test = valid_samples[val_end:]

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    json.dump(train, f)

with open(VAL_OUT, "w", encoding="utf-8") as f:
    json.dump(val, f)

with open(TEST_OUT, "w", encoding="utf-8") as f:
    json.dump(test, f)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

def average_rooms(dataset):
    if not dataset:
        return 0

    return sum(
        len(sample["target"]["rooms"])
        for sample in dataset
    ) / len(dataset)

def average_edges(dataset):
    if not dataset:
        return 0

    return sum(
        len(sample["target"]["edges"])
        for sample in dataset
    ) / len(dataset)

print("\n" + "=" * 70)
print("FINAL DATASET READY")
print("=" * 70)

print(f"\nTotal valid : {total}")

print("\nSplit:")
print(f"Train : {len(train)} ({len(train)/total:.1%})")
print(f"Val   : {len(val)} ({len(val)/total:.1%})")
print(f"Test  : {len(test)} ({len(test)/total:.1%})")

print("\nAverage target structure:")

print(
    f"Train rooms : {average_rooms(train):.2f}"
)

print(
    f"Train edges : {average_edges(train):.2f}"
)

print(
    f"Val rooms   : {average_rooms(val):.2f}"
)

print(
    f"Test rooms  : {average_rooms(test):.2f}"
)

print("\nFiles created:")

print(f"Train : {TRAIN_OUT}")
print(f"Val   : {VAL_OUT}")
print(f"Test  : {TEST_OUT}")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
