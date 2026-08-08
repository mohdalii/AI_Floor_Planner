import json
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path("dataset/resplan_graphs.json")

print("=" * 70)
print("RESPLAN GRAPH DATASET STATISTICS")
print("=" * 70)

with open(INPUT, "r", encoding="utf-8") as f:
    graphs = json.load(f)

print(f"\nTotal plans : {len(graphs)}")

room_counter = Counter()
edge_counter = Counter()
room_pair_counter = Counter()

node_counts = []
edge_counts = []

for graph in graphs:

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_counts.append(len(nodes))
    edge_counts.append(len(edges))

    for node in nodes:
        room_type = node.get("type")

        if room_type:
            room_counter[room_type] += 1

    for edge in edges:

        edge_type = edge.get("type", "unknown")
        edge_counter[edge_type] += 1

        source = edge.get("source", "")
        target = edge.get("target", "")

        source_type = source.rsplit("_", 1)[0]
        target_type = target.rsplit("_", 1)[0]

        pair = tuple(sorted([source_type, target_type]))

        room_pair_counter[pair] += 1

print("\n" + "=" * 70)
print("NODE STATISTICS")
print("=" * 70)

print(f"\nAverage nodes/plan : {sum(node_counts) / len(node_counts):.2f}")
print(f"Maximum nodes/plan : {max(node_counts)}")
print(f"Minimum nodes/plan : {min(node_counts)}")

print("\nRoom type frequency:")

for room, count in room_counter.most_common():
    print(f"{room:20s} {count:8d}")

print("\n" + "=" * 70)
print("EDGE STATISTICS")
print("=" * 70)

print(f"\nAverage edges/plan : {sum(edge_counts) / len(edge_counts):.2f}")
print(f"Maximum edges/plan : {max(edge_counts)}")
print(f"Minimum edges/plan : {min(edge_counts)}")

print("\nEdge types:")

for edge_type, count in edge_counter.most_common():
    print(f"{edge_type:20s} {count:8d}")

print("\n" + "=" * 70)
print("MOST COMMON ROOM RELATIONSHIPS")
print("=" * 70)

for pair, count in room_pair_counter.most_common(30):
    print(
        f"{pair[0]:20s} <-> "
        f"{pair[1]:20s} : {count}"
    )

print("\n" + "=" * 70)
print("GRAPH STATISTICS COMPLETE")
print("=" * 70)
