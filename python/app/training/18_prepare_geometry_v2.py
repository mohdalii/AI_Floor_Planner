import json
from pathlib import Path

ROOT = Path(".")

INPUT_DIR = ROOT / "dataset" / "geometry_training"
OUTPUT_DIR = ROOT / "dataset" / "geometry_v2_training"

ROOM_TYPES = [
    "bedroom",
    "bathroom",
    "kitchen",
    "living",
    "balcony",
    "storage",
    "stair",
    "front_door",
]

ROOM_TO_ID = {
    room: i
    for i, room in enumerate(ROOM_TYPES)
}


def prepare_sample(sample):
    rooms = sample.get("rooms", [])

    prepared_rooms = []

    # Count index separately for every room type
    type_indices = {}

    for room in rooms:

        room_type = room.get("type")

        if room_type not in ROOM_TO_ID:
            continue

        bounds = room.get("bounds")

        if not bounds or len(bounds) != 4:
            continue

        type_index = type_indices.get(room_type, 0)
        type_indices[room_type] = type_index + 1

        prepared_rooms.append({
            "id": str(room.get("id", "")),
            "type": room_type,
            "type_id": ROOM_TO_ID[room_type],

            # NEW:
            # bedroom_0, bedroom_1, bedroom_2 ...
            "room_index": type_index,

            # NEW:
            # normalized position inside the plan
            "bounds": [
                float(bounds[0]),
                float(bounds[1]),
                float(bounds[2]),
                float(bounds[3]),
            ],

            "center": [
                (
                    float(bounds[0]) +
                    float(bounds[2])
                ) / 2.0,

                (
                    float(bounds[1]) +
                    float(bounds[3])
                ) / 2.0,
            ],

            "size": [
                max(
                    0.0,
                    float(bounds[2]) -
                    float(bounds[0])
                ),

                max(
                    0.0,
                    float(bounds[3]) -
                    float(bounds[1])
                ),
            ],
        })

    return {
        "plan_id": sample.get("plan_id"),
        "input": sample.get("input", {}),
        "rooms": prepared_rooms,
        "room_count": len(prepared_rooms),
    }


print("=" * 70)
print("ROOM-AWARE GEOMETRY V2 DATASET")
print("=" * 70)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

total = 0

for split in ["train", "val", "test"]:

    input_path = INPUT_DIR / f"{split}.json"
    output_path = OUTPUT_DIR / f"{split}.json"

    print(f"\nLoading {split}.json...")

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:
        samples = json.load(f)

    prepared = []

    for sample in samples:

        result = prepare_sample(sample)

        if result["room_count"] > 0:
            prepared.append(result)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            prepared,
            f,
            indent=2
        )

    print(
        f"Input samples  : {len(samples)}"
    )

    print(
        f"Output samples : {len(prepared)}"
    )

    if prepared:

        average_rooms = (
            sum(
                item["room_count"]
                for item in prepared
            )
            / len(prepared)
        )

        print(
            f"Average rooms  : "
            f"{average_rooms:.2f}"
        )

    total += len(prepared)


print("\n" + "=" * 70)
print("GEOMETRY V2 DATASET COMPLETE")
print("=" * 70)

print(
    f"\nTotal prepared samples : {total}"
)

print(
    f"Output directory       : "
    f"{OUTPUT_DIR}"
)

print("\nRoom type mapping:")

for room, room_id in ROOM_TO_ID.items():

    print(
        f"{room:15s} -> {room_id}"
    )

print("\nNew room-index system:")

print("bedroom_0")
print("bedroom_1")
print("bedroom_2")
print("bathroom_0")
print("bathroom_1")
print("kitchen_0")
print("living_0")
print("balcony_0")
print("...")

print("\nDone.")
