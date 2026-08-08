import json
import pickle
from pathlib import Path

ROOT = Path(".")

GRAPH_PATH = ROOT / "dataset" / "resplan_graphs.json"
DATASET_PATH = ROOT / "dataset" / "ResPlan.pkl"
OUTPUT_PATH = ROOT / "dataset" / "training_samples.json"

print("=" * 70)
print("BUILDING GENERATION TRAINING SAMPLES")
print("=" * 70)

print("\nLoading graph dataset...")

with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    graphs = json.load(f)

print(f"Graphs loaded: {len(graphs)}")

print("\nLoading original geometry dataset...")

with open(DATASET_PATH, "rb") as f:
    plans = pickle.load(f)

print(f"Plans loaded: {len(plans)}")

plan_lookup = {
    str(plan.get("id")): plan
    for plan in plans
}

samples = []

for index, graph in enumerate(graphs):

    if index % 500 == 0:
        print(f"Processing {index}/{len(graphs)}...")

    plan_id = str(graph.get("plan_id"))

    plan = plan_lookup.get(plan_id)

    if plan is None:
        continue

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    room_counts = {}

    for node in nodes:

        room_type = node.get("type")

        if room_type is None:
            continue

        room_counts[room_type] = (
            room_counts.get(room_type, 0) + 1
        )

    requirements = {
        "bedrooms": room_counts.get("bedroom", 0),
        "bathrooms": room_counts.get("bathroom", 0),
        "kitchens": room_counts.get("kitchen", 0),
        "living_rooms": room_counts.get("living", 0),
        "balconies": room_counts.get("balcony", 0),
        "storages": room_counts.get("storage", 0),
        "stairs": room_counts.get("stair", 0),
    }

    target_rooms = []

    for node in nodes:

        target_rooms.append({
            "id": node.get("id"),
            "type": node.get("type"),
            "area": node.get("area"),
            "centroid": node.get("centroid"),
            "bounds": node.get("bounds"),
        })

    target_graph = []

    for edge in edges:

        target_graph.append({
            "source": edge.get("source"),
            "target": edge.get("target"),
            "type": edge.get("type"),
        })

    sample = {
        "plan_id": plan_id,

        "input": {
            "plot_area": float(
                plan.get("area", 0) or 0
            ),

            "net_area": float(
                plan.get("net_area", 0) or 0
            ),

            "wall_depth": float(
                plan.get("wall_depth", 0) or 0
            ),

            "requirements": requirements,
        },

        "target": {
            "rooms": target_rooms,
            "edges": target_graph,
        },
    }

    samples.append(sample)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        samples,
        f,
        indent=2
    )

print("\n" + "=" * 70)
print("TRAINING SAMPLE DATASET COMPLETE")
print("=" * 70)

print(f"\nSamples created : {len(samples)}")
print(f"Output          : {OUTPUT_PATH}")

if samples:

    sample = samples[0]

    print("\nSample training input:")
    print(
        json.dumps(
            sample["input"],
            indent=2
        )
    )

    print("\nTarget rooms:",
          len(sample["target"]["rooms"]))

    print("Target edges:",
          len(sample["target"]["edges"]))

print("\nDone.")
