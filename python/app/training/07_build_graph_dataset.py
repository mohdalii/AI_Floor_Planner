import pickle
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT / "dataset" / "ResPlan.pkl"
UTILS_PATH = ROOT / "dataset"

sys.path.insert(0, str(UTILS_PATH))

from resplan_utils import plan_to_graph, add_adjacency_edges


OUTPUT_PATH = ROOT / "dataset" / "resplan_graphs.json"

print("=" * 70)
print("RESPLAN ACTUAL GRAPH DATASET")
print("=" * 70)

print("\nLoading dataset...")

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} plans.")

graphs = []

for index, plan in enumerate(data):

    if index % 500 == 0:
        print(f"Processing {index}/{len(data)}...")

    try:
        # Build strict ResPlan graph
        graph = plan_to_graph(plan)

        # Convert to paper/Kaggle-compatible taxonomy
        graph = add_adjacency_edges(graph)

        nodes = []

        for node_id, node_data in graph.nodes(data=True):

            geometry = node_data.get("geometry")

            if geometry is None or geometry.is_empty:
                continue

            min_x, min_y, max_x, max_y = geometry.bounds

            nodes.append({
                "id": str(node_id),
                "type": node_data.get("type"),
                "area": float(node_data.get("area", geometry.area)),
                "centroid": [
                    float(geometry.centroid.x),
                    float(geometry.centroid.y)
                ],
                "bounds": [
                    float(min_x),
                    float(min_y),
                    float(max_x),
                    float(max_y)
                ]
            })

        edges = []

        for node_a, node_b, edge_data in graph.edges(data=True):

            edges.append({
                "source": str(node_a),
                "target": str(node_b),
                "type": edge_data.get("type", "unknown")
            })

        graphs.append({
            "plan_id": plan.get("id"),
            "nodes": nodes,
            "edges": edges
        })

    except Exception as e:

        print(
            f"WARNING: Plan {index} "
            f"({plan.get('id')}) failed: {e}"
        )

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(graphs, f, indent=2)

print("\n" + "=" * 70)
print("GRAPH DATASET COMPLETE")
print("=" * 70)

print(f"\nPlans processed : {len(graphs)}")
print(f"Output          : {OUTPUT_PATH}")

if graphs:

    sample = graphs[0]

    print("\nSample Plan")
    print("Plan ID :", sample["plan_id"])

    print("\nNodes:")

    for node in sample["nodes"]:
        print(
            f" - {node['id']} "
            f"| type={node['type']} "
            f"| area={node['area']:.2f}"
        )

    print("\nEdges:")

    for edge in sample["edges"]:
        print(
            f" - {edge['source']} "
            f"<-> {edge['target']} "
            f"| {edge['type']}"
        )

print("\nDone.")
