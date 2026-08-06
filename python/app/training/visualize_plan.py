import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

# Dataset Path
DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "ResPlan.pkl"

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

plan = data[0]

fig, ax = plt.subplots(figsize=(10,10))

room_colors = {
    "bedroom": "#87CEEB",
    "bathroom": "#FFB6C1",
    "living": "#98FB98",
    "kitchen": "#FFA500",
    "balcony": "#FFFF99",
    "parking": "#D3D3D3",
    "garden": "#90EE90",
    "pool": "#00FFFF"
}

for room, color in room_colors.items():

    geometry = plan.get(room)

    if geometry is None:
        continue

    if geometry.is_empty:
        continue

    # Handle Polygon
    if isinstance(geometry, Polygon):
        x, y = geometry.exterior.xy
        ax.fill(x, y, color=color, alpha=0.7)
        ax.plot(x, y, color="black")

    # Handle MultiPolygon
    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=0.7)
            ax.plot(x, y, color="black")

ax.set_title(f"Floor Plan ID : {plan['id']}")
ax.set_aspect("equal")
plt.grid(True)

plt.show()
