import pickle
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "ResPlan.pkl"

print("=" * 60)
print("Loading ResPlan Dataset")
print("=" * 60)

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

print(f"\n✅ Total Plans : {len(data)}")

plan = data[0]

print("\nPlan Type :", type(plan))

print("\nAvailable Keys")
print("-" * 40)

for key in plan.keys():
    print(key)

print("-" * 40)

print("\nPlan ID :", plan.get("id"))

# Bedroom
print("\n================ BEDROOM DATA ================\n")
print("Type :", type(plan["bedroom"]))
from shapely.geometry import MultiPolygon

print("\n================ BEDROOM DATA ================\n")

bedroom = plan["bedroom"]

print("Type :", type(bedroom))

print("\nGeometry Type :", bedroom.geom_type)

print("\nArea :", bedroom.area)

print("\nBounds :", bedroom.bounds)

print("\nWKT Preview :")

print(str(bedroom)[:500])

# Bathroom
print("\n================ BATHROOM DATA ================\n")

bathroom = plan["bathroom"]

print("Type :", type(bathroom))

print("Geometry :", bathroom.geom_type)

print("Area :", bathroom.area)

print("Bounds :", bathroom.bounds)

# Kitchen
print("\n================ KITCHEN DATA ================\n")

kitchen = plan["kitchen"]

print("Type :", type(kitchen))

print("Geometry :", kitchen.geom_type)

print("Area :", kitchen.area)

print("Bounds :", kitchen.bounds)