import json
from pathlib import Path

UTILS_PATH = Path("dataset/resplan_utils.py")

print("=" * 70)
print("RESPLAN GRAPH UTILITY INSPECTION")
print("=" * 70)

if not UTILS_PATH.exists():
    print(f"\nERROR: {UTILS_PATH} not found.")
    raise SystemExit(1)

text = UTILS_PATH.read_text(encoding="utf-8")

print("\nSearching for graph-related functions...\n")

keywords = [
    "plan_to_graph",
    "add_adjacency_edges",
    "graph",
    "edge",
    "node",
]

lines = text.splitlines()

for number, line in enumerate(lines, start=1):
    lower = line.lower()

    if any(keyword in lower for keyword in keywords):
        print(f"{number:4d}: {line}")

print("\n" + "=" * 70)
print("FUNCTION DEFINITIONS")
print("=" * 70)

for number, line in enumerate(lines, start=1):
    stripped = line.strip()

    if stripped.startswith("def "):
        print(f"{number:4d}: {stripped}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
