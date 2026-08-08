import json
from pathlib import Path

ROOT = Path(".")

INPUT_DIR = ROOT / "dataset" / "training"
OUTPUT_DIR = ROOT / "dataset" / "geometry_training"

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


def normalize_bounds(bounds):

    if not bounds or len(bounds) != 4:
        return None

    return [
        float(bounds[0]) / 256.0,
        float(bounds[1]) / 256.0,
        float(bounds[2]) / 256.0,
        float(bounds[3]) / 256.0,
    ]


def prepare_sample(sample):

    rooms = sample.get("target", {}).get("rooms", [])

    prepared_rooms = []

    for room in rooms:

        room_type = room.get("type")

        if room_type not in ROOM_TO_ID:
            continue

        bounds = room.get("bounds_norm")

        if bounds is None:
            bounds = normalize_bounds(
                room.get("bounds")
            )

        else:
            bounds = [
                float(v)
                for v in bounds
            ]

        if bounds is None:
            continue

        prepared_rooms.append({
            "id": str(room.get("id", "")),
            "type": room_type,
            "type_id": ROOM_TO_ID[room_type],
            "bounds": bounds,
        })

    # Deterministic ordering:
    # room type first, then original room id.
    prepared_rooms.sort(
        key=lambda room: (
            room["type_id"],
            room["id"],
        )
    )

    return {
        "plan_id": sample.get("plan_id"),

        "input": sample.get("input", {}),

        "rooms": prepared_rooms,

        "room_count": len(prepared_rooms),
    }


print("=" * 70)
print("ROOM-AWARE GEOMETRY DATASET PREPARATION")
print("=" * 70)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

total = 0

for split in ["train", "val", "test"]:

    input_path = INPUT_DIR / f"{split}.json"
    output_path = OUTPUT_DIR / f"{split}.json"

    print(
        f"\nLoading {split}.json..."
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:
        samples = json.load(f)

    prepared = []

    for sample in samples:

        result = prepare_sample(
            sample
        )

        if result["room_count"] > 0:
            prepared.append(result)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            prepared,
            f
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
print("ROOM-AWARE DATASET COMPLETE")
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
        f"  {room:15s} -> {room_id}"
    )

print("\nDone.")
