import pickle
import json
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon

DATASET_PATH = Path("dataset/ResPlan.pkl")
OUTPUT_PATH = Path("dataset/room_adjacency.json")

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

def get_geometry_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if hasattr(geometry, "geoms"):
        return [g for g in geometry.geoms if isinstance(g, Polygon)]

    return []

def room_exists(plan, room):
    geometry = plan.get(room)

    return (
        geometry is not None
        and not geometry.is_empty
        and len(get_geometry_parts(geometry)) > 0
    )

def rooms_are_near(g1, g2, tolerance=0.5):
    if g1 is None or g2 is None:
        return False

    if g1.is_empty or g2.is_empty:
        return False

    distance = g1.distance(g2)

    return distance <= tolerance

print("=" * 70)
print("RESPLAN ROOM ADJACENCY EXTRACTION")
print("=" * 70)

print("\nLoading dataset...")

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} plans.")

results = []

for index, plan in enumerate(data):

    if index % 500 == 0:
        print(f"Processing {index}/{len(data)}...")

    available_rooms = [
        room for room in ROOMS
        if room_exists(plan, room)
    ]

    adjacency = []

    for i in range(len(available_rooms)):

        room_a = available_rooms[i]
        geom_a = plan.get(room_a)

        for j in range(i + 1, len(available_rooms)):

            room_b = available_rooms[j]
            geom_b = plan.get(room_b)

            if rooms_are_near(geom_a, geom_b):

                adjacency.append({
                    "room_a": room_a,
                    "room_b": room_b,
                    "relation": "near"
                })

    results.append({
        "plan_id": plan.get("id"),
        "rooms": available_rooms,
        "adjacency": adjacency
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("ADJACENCY EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nPlans processed : {len(results)}")
print(f"Output          : {OUTPUT_PATH}")

if results:
    sample = results[0]

    print("\nSample Plan:")
    print("Plan ID :", sample["plan_id"])

    print("\nRooms:")
    for room in sample["rooms"]:
        print(" -", room)

    print("\nDetected nearby rooms:")
    for edge in sample["adjacency"][:20]:
        print(
            f" - {edge['room_a']} <-> "
            f"{edge['room_b']} : {edge['relation']}"
        )

print("\nDone.")
