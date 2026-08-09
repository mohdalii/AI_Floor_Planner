"""
RAW V4 vs SOLVED V4 diagnostic.

Runs the geometry model and the priority-tiered layout solver against
real held-out test-set plans (so ground truth is available) and
reports whether the solver actually improves the layout without
destroying the model's prediction - not just whether it drives
collision to 0%.

Runs on a bounded sample of the ~1700-plan test set, not all of it:
the solver's per-candidate cost evaluation is O(rooms^2) in pure
Python, so the full set would take far too long to be useful here.
The sample size is printed below rather than silently truncated.
"""

import json
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("D:/ai-floor-planner")

TEST_PATH = ROOT / "dataset" / "geometry_v2_training" / "test.json"
OUTPUT_DIR = ROOT / "dataset" / "geometry_v4_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 25
VIZ_EXAMPLES = 4
MOVE_SIGNIFICANT_THRESHOLD = 0.05   # center displacement, fraction of plot
RESIZE_SIGNIFICANT_THRESHOLD = 0.15  # relative area change
RULE_TOLERANCE = 0.03               # gap-to-target allowed before "violation"

sys.path.insert(0, str(ROOT / "python" / "app" / "ml"))
from predict import load_model, DEVICE, ROOM_COLORS  # noqa: E402

sys.path.insert(0, str(ROOT / "python" / "app" / "training"))
from geometry._layout._solver import (  # noqa: E402
    refine_layout,
    collision_rate,
    boundary_violation_rate,
    _build_attach_map,
    _gap_to_room,
    _center,
    _distance,
)

MAX_ROOM_INDEX = 10

ROOM_TYPES = [
    "bedroom", "bathroom", "kitchen", "living",
    "balcony", "storage", "stair", "front_door",
]


def build_features(sample):
    inp = sample["input"]
    req = inp["requirements"]
    return torch.tensor(
        [
            float(inp.get("plot_area_norm", 0) or 0),
            float(inp.get("wall_depth_norm", 0) or 0),
            float(req.get("bedrooms", 0)),
            float(req.get("bathrooms", 0)),
            float(req.get("kitchens", 0)),
            float(req.get("living_rooms", 0)),
            float(req.get("balconies", 0)),
            float(req.get("storages", 0)),
            float(req.get("stairs", 0)),
        ],
        dtype=torch.float32,
    )


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return cx - w/2, cy - h/2, cx + w/2, cy + h/2


def box_iou_xyxy(pred_cxcywh, gt_xyxy):
    px1, py1, px2, py2 = cxcywh_to_xyxy(pred_cxcywh)
    gx1, gy1, gx2, gy2 = gt_xyxy

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_g = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_p + area_g - inter
    return inter / max(union, 1e-8)


def mae_xyxy(pred_cxcywh, gt_xyxy):
    px1, py1, px2, py2 = cxcywh_to_xyxy(pred_cxcywh)
    return (
        abs(px1 - gt_xyxy[0]) + abs(py1 - gt_xyxy[1])
        + abs(px2 - gt_xyxy[2]) + abs(py2 - gt_xyxy[3])
    ) / 4.0


def rule_violations(boxes, room_types, ids):
    """How many of the rule engine's attach-map relationships are NOT
    satisfied (gap to target exceeds RULE_TOLERANCE) in this layout."""
    attach_map = _build_attach_map(boxes, room_types, ids)
    violations = 0
    for i, target in attach_map.items():
        if _gap_to_room(boxes, i, target) > RULE_TOLERANCE:
            violations += 1
    return violations, len(attach_map)


def draw_panel(ax, boxes, rooms, title):
    ax.set_title(title, fontsize=9)
    for i, room in enumerate(rooms):
        cx, cy, w, h = [float(v) for v in boxes[i]]
        x1, y1 = cx - w/2, cy - h/2
        color = ROOM_COLORS.get(ROOM_TYPES[int(room["type_id"])], "#999999")
        ax.add_patch(Rectangle(
            (x1, y1), max(w, 0.005), max(h, 0.005),
            facecolor=color, edgecolor="black", alpha=0.65, linewidth=1.2,
        ))
        ax.text(cx, cy, f"{ROOM_TYPES[int(room['type_id'])]}_{room['room_index']}",
                ha="center", va="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


print("=" * 70)
print("AI FLOOR PLANNER - V4 DIAGNOSTIC (RAW vs SOLVED, priority-tiered solver)")
print("=" * 70)

print("\nDevice:", DEVICE)

with open(TEST_PATH, "r", encoding="utf-8") as f:
    all_samples = json.load(f)

samples = [s for s in all_samples if s["rooms"]][:SAMPLE_SIZE]

print(f"Test set size total : {len(all_samples)}")
print(f"Evaluated on sample : {len(samples)} plans "
      f"(bounded for runtime - see module docstring)")

model = load_model()

raw_mae_total = solved_mae_total = 0.0
raw_iou_total = solved_iou_total = 0.0
raw_collision_total = solved_collision_total = 0.0
raw_boundary_total = solved_boundary_total = 0.0
room_count = 0
sample_count = 0

center_shift_total = 0.0
area_change_total = 0.0
resized_rooms = 0
moved_significantly = 0

rule_violations_total = 0
rule_checked_total = 0

viz_saved = 0

for sample_index, sample in enumerate(samples):
    rooms = sample["rooms"]

    x = build_features(sample).unsqueeze(0).to(DEVICE)
    room_types = torch.tensor(
        [int(r["type_id"]) for r in rooms], dtype=torch.long
    ).unsqueeze(0).to(DEVICE)
    room_indices = torch.tensor(
        [min(int(r["room_index"]), MAX_ROOM_INDEX - 1) for r in rooms],
        dtype=torch.long,
    ).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, len(rooms), dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        raw = model(x, room_types, room_indices, mask.logical_not())
        solved = refine_layout(raw, room_types, mask)

    raw_boxes = raw[0].detach().cpu()
    solved_boxes = solved[0].detach().cpu()
    gt_boxes_xyxy = [room["bounds"] for room in rooms]

    raw_collision_total += collision_rate(raw, mask)
    solved_collision_total += collision_rate(solved, mask)
    raw_boundary_total += boundary_violation_rate(raw, mask)
    solved_boundary_total += boundary_violation_rate(solved, mask)

    ids = list(range(len(rooms)))
    v_bad, v_total = rule_violations(solved_boxes, room_types[0].cpu(), ids)
    rule_violations_total += v_bad
    rule_checked_total += v_total

    for i in range(len(rooms)):
        raw_box = raw_boxes[i]
        solved_box = solved_boxes[i]
        gt = gt_boxes_xyxy[i]

        raw_mae_total += mae_xyxy(raw_box, gt)
        solved_mae_total += mae_xyxy(solved_box, gt)
        raw_iou_total += box_iou_xyxy(raw_box, gt)
        solved_iou_total += box_iou_xyxy(solved_box, gt)

        d = _distance(_center(raw_box), _center(solved_box))
        center_shift_total += d
        if d > MOVE_SIGNIFICANT_THRESHOLD:
            moved_significantly += 1

        raw_area = float(raw_box[2] * raw_box[3])
        solved_area = float(solved_box[2] * solved_box[3])
        rel_change = abs(solved_area - raw_area) / max(raw_area, 1e-6)
        area_change_total += abs(solved_area - raw_area)
        if rel_change > RESIZE_SIGNIFICANT_THRESHOLD:
            resized_rooms += 1

        room_count += 1

    sample_count += 1

    if viz_saved < VIZ_EXAMPLES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        gt_as_cxcywh = []
        for x1, y1, x2, y2 in gt_boxes_xyxy:
            gt_as_cxcywh.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])

        draw_panel(axes[0], gt_as_cxcywh, rooms, "GROUND TRUTH")
        draw_panel(axes[1], raw_boxes, rooms,
                   f"RAW V4\nIoU={raw_iou_total/room_count:.3f}")
        draw_panel(axes[2], solved_boxes, rooms,
                   f"SOLVED V4\nIoU={solved_iou_total/room_count:.3f}")

        fig.suptitle(f"Plan {sample['plan_id']}")
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"gt_raw_solved_{viz_saved}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        viz_saved += 1

    if sample_index < 10:
        print(
            f"Plan {sample['plan_id']}: "
            f"RAW IoU={raw_iou_total/room_count:.4f} "
            f"SOLVED IoU={solved_iou_total/room_count:.4f}"
        )


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nSamples: {sample_count}   Rooms: {room_count}")

print("\nRAW V4")
print(f"  Coordinate MAE      : {raw_mae_total/max(room_count,1):.6f}")
print(f"  Mean box IoU        : {raw_iou_total/max(room_count,1):.6f}")
print(f"  Collision rate      : {raw_collision_total/max(sample_count,1):.6f}")
print(f"  Boundary violation  : {raw_boundary_total/max(sample_count,1):.6f}")

print("\nSOLVED V4")
print(f"  Coordinate MAE      : {solved_mae_total/max(room_count,1):.6f}")
print(f"  Mean box IoU        : {solved_iou_total/max(room_count,1):.6f}")
print(f"  Collision rate      : {solved_collision_total/max(sample_count,1):.6f}")
print(f"  Boundary violation  : {solved_boundary_total/max(sample_count,1):.6f}")

print("\nRAW -> SOLVED")
print(f"  Avg center displacement      : {center_shift_total/max(room_count,1):.6f}")
print(f"  Avg area change (abs)        : {area_change_total/max(room_count,1):.6f}")
print(f"  Rooms resized (>{RESIZE_SIGNIFICANT_THRESHOLD:.0%} area change) : "
      f"{resized_rooms} / {room_count} ({resized_rooms/max(room_count,1):.1%})")
print(f"  Rooms moved significantly (>{MOVE_SIGNIFICANT_THRESHOLD}) : "
      f"{moved_significantly} / {room_count} ({moved_significantly/max(room_count,1):.1%})")
print(f"  Architectural rule violations : "
      f"{rule_violations_total} / {rule_checked_total} "
      f"({rule_violations_total/max(rule_checked_total,1):.1%})")

print(f"\nVisual examples saved: {viz_saved} (dataset/geometry_v4_diagnostic/gt_raw_solved_*.png)")
print("\nDone.")
