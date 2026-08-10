"""
End-to-end floor plan generation.

Takes room requirements (counts per room type + plot metadata) and
returns predicted room boxes: geometry model -> layout solver.

This is the first script in the project that generates a plan for
requirements that are NOT pulled from the training/test dataset -
every other script under python/app/training only replays fixed
dataset samples.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "python" / "app" / "models" / "room_geometry_v4.pt"
SOLVER_DIR = ROOT / "python" / "app" / "training"
OUTPUT_DIR = ROOT / "dataset" / "generated_plans"

if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from geometry._layout._solver import (  # noqa: E402
    refine_layout,
    collision_rate,
    boundary_violation_rate,
    validate_layout,
    _build_attach_map,
    SIZE_RANGES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ROOM_TYPES = 8
MAX_ROOM_INDEX = 10

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

ROOM_COLORS = {
    "bedroom": "#8ecae6",
    "bathroom": "#adb5bd",
    "kitchen": "#ffb703",
    "living": "#90be6d",
    "balcony": "#f4a261",
    "storage": "#cdb4db",
    "stair": "#e76f51",
    "front_door": "#264653",
}


class RoomGeometryV4(nn.Module):

    def __init__(self):
        super().__init__()

        self.global_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.room_embedding = nn.Embedding(NUM_ROOM_TYPES, 32)
        self.index_embedding = nn.Embedding(MAX_ROOM_INDEX, 16)
        self.input_projection = nn.Linear(128 + 32 + 16, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, global_x, room_types, room_indices, padding_mask):
        global_features = self.global_encoder(global_x)
        global_features = global_features.unsqueeze(1).expand(
            -1, room_types.shape[1], -1
        )

        room_features = self.room_embedding(room_types)
        index_features = self.index_embedding(room_indices)

        combined = torch.cat(
            [global_features, room_features, index_features], dim=-1
        )
        tokens = self.input_projection(combined)
        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return self.decoder(tokens)


_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = RoomGeometryV4().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _MODEL = model
    return model


def build_room_list(requirements):
    # NOTE: staircases are deliberately not part of this MVP. Even if
    # `requirements` includes a "stairs" count, no stair room is ever
    # generated - the rule engine, solver, and plot sizing below never
    # reserve, move for, or otherwise reason about one.
    counts = [
        ("bedroom", 0, requirements.get("bedrooms", 0)),
        ("bathroom", 1, requirements.get("bathrooms", 0)),
        ("kitchen", 2, requirements.get("kitchens", 0)),
        ("living", 3, requirements.get("living_rooms", 0)),
        ("balcony", 4, requirements.get("balconies", 0)),
        ("storage", 5, requirements.get("storages", 0)),
        ("front_door", 7, requirements.get("front_doors", 1)),
    ]

    rooms = []
    for name, type_id, count in counts:
        for i in range(int(count)):
            rooms.append(
                {
                    "type": name,
                    "type_id": type_id,
                    "room_index": min(i, MAX_ROOM_INDEX - 1),
                }
            )

    return rooms


def build_global_features(requirements, plot_area_norm, wall_depth_norm):
    # The model's 9th input slot is a "stairs count" feature it was
    # trained on. Since the MVP never generates a stair room, this is
    # always fed as 0 - telling the model "no stairs" is what keeps its
    # conditioning consistent with what's actually being generated.
    return torch.tensor(
        [
            float(plot_area_norm),
            float(wall_depth_norm),
            float(requirements.get("bedrooms", 0)),
            float(requirements.get("bathrooms", 0)),
            float(requirements.get("kitchens", 0)),
            float(requirements.get("living_rooms", 0)),
            float(requirements.get("balconies", 0)),
            float(requirements.get("storages", 0)),
            0.0,
        ],
        dtype=torch.float32,
    )


BASE_HOME_AREA_M2 = 90.0
USABLE_PLOT_FRACTION = 0.82


def estimate_plot_dimensions_m(rooms, solved_boxes, base_home_area_m2=BASE_HOME_AREA_M2):
    """The geometry model always predicts room boxes on its own internal
    scale - it doesn't actually shrink rooms when told the plot is
    bigger (verified empirically: predicted sizes barely move across a
    4x range of plot_area_norm, since it wasn't trained to respond to
    that feature that way). The solver no longer forces the layout into
    a fixed [0,1] square either (see _clamp_box in the solver) - it
    lets rooms spread out to however much space they actually need to
    attach properly, rather than compromising to fit an artificial box.
    So the real-world plot size is derived from wherever the solved
    layout actually ended up, not assumed in advance or forced to a
    particular aspect ratio.

    A fixed meters-per-normalized-unit conversion is calibrated from
    the room list's total minimum reasonable area (SIZE_RANGES) - what
    a "just barely fits, nothing to spare" 1.0 x 1.0 layout would need
    in real terms - then applied directly to the actual solved bounding
    box extent, whatever size and shape that turned out to be.
    """
    total_min_fraction = sum(
        SIZE_RANGES.get(r["type_id"], (0.02, 0.30))[0] for r in rooms
    )
    nominal_scale = max(1.0, total_min_fraction / USABLE_PLOT_FRACTION)
    nominal_area_m2 = base_home_area_m2 * nominal_scale
    meters_per_unit = math.sqrt(nominal_area_m2)

    xs = [float(b[0] - b[2]/2) for b in solved_boxes] + [float(b[0] + b[2]/2) for b in solved_boxes]
    ys = [float(b[1] - b[3]/2) for b in solved_boxes] + [float(b[1] + b[3]/2) for b in solved_boxes]
    extent_x = max((max(xs) - min(xs)) if xs else 1.0, 0.2)
    extent_y = max((max(ys) - min(ys)) if ys else 1.0, 0.2)

    return extent_x * meters_per_unit, extent_y * meters_per_unit


def generate_floor_plan(
    requirements,
    plot_area_norm=0.12,
    wall_depth_norm=0.18,
    model=None,
):
    if model is None:
        model = load_model()

    rooms = build_room_list(requirements)
    if not rooms:
        raise ValueError("Requirements must include at least one room.")

    global_x = (
        build_global_features(requirements, plot_area_norm, wall_depth_norm)
        .unsqueeze(0)
        .to(DEVICE)
    )

    room_types = torch.tensor(
        [r["type_id"] for r in rooms], dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    room_indices = torch.tensor(
        [r["room_index"] for r in rooms], dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    mask = torch.ones(1, len(rooms), dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        raw = model(global_x, room_types, room_indices, mask.logical_not())
        solved, move_logs = refine_layout(raw, room_types, mask, return_log=True)

    solved_collision = collision_rate(solved, mask)
    solved_boxes_cpu = solved[0].detach().cpu()
    plot_width_m, plot_depth_m = estimate_plot_dimensions_m(rooms, solved_boxes_cpu)

    ids = list(range(len(rooms)))
    attach_map = _build_attach_map(solved_boxes_cpu, room_types[0].cpu(), ids)
    checklist = validate_layout(solved_boxes_cpu, room_types[0].cpu(), ids, attach_map)

    return {
        "rooms": rooms,
        "raw_boxes": raw[0].detach().cpu(),
        "solved_boxes": solved_boxes_cpu,
        "raw_collision_rate": collision_rate(raw, mask),
        "solved_collision_rate": solved_collision,
        "raw_boundary_violation_rate": boundary_violation_rate(raw, mask),
        "plot_width_m": plot_width_m,
        "plot_depth_m": plot_depth_m,
        "plot_area_m2": plot_width_m * plot_depth_m,
        "plot_expanded": (plot_width_m * plot_depth_m) > BASE_HOME_AREA_M2 + 1e-6,
        "move_log": move_logs[0],
        "validation_checklist": checklist,
    }


def _draw(ax, boxes, rooms, title):
    ax.set_title(title, fontsize=10)

    bedroom_areas = [
        (i, float(boxes[i][2]) * float(boxes[i][3]))
        for i, r in enumerate(rooms) if r["type"] == "bedroom"
    ]
    master_index = max(bedroom_areas, key=lambda p: p[1])[0] if bedroom_areas else None

    for i, room in enumerate(rooms):
        cx, cy, w, h = [float(v) for v in boxes[i]]
        x1, y1 = cx - w / 2, cy - h / 2

        color = ROOM_COLORS.get(room["type"], "#999999")

        rect = Rectangle(
            (x1, y1),
            max(w, 0.005),
            max(h, 0.005),
            facecolor=color,
            edgecolor="black",
            alpha=0.65,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        label = f"{room['type']}_{room['room_index']}"
        if i == master_index:
            label += " (master)"

        ax.text(
            cx, cy, label,
            ha="center", va="center", fontsize=7,
        )

    # The solver no longer clamps rooms to a fixed [0,1] square, so the
    # plotted extent has to be read from the actual boxes instead of
    # assumed - a layout that needed to spread out to attach properly
    # can legitimately extend beyond [0,1] now.
    xs = [float(b[0] - b[2]/2) for b in boxes] + [float(b[0] + b[2]/2) for b in boxes]
    ys = [float(b[1] - b[3]/2) for b in boxes] + [float(b[1] + b[3]/2) for b in boxes]
    pad_x = max((max(xs) - min(xs)) * 0.05, 0.02) if xs else 0.05
    pad_y = max((max(ys) - min(ys)) * 0.05, 0.02) if ys else 0.05
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def render(result, output_path, title="Generated Floor Plan"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    _draw(
        axes[0], result["raw_boxes"], result["rooms"],
        f"RAW MODEL OUTPUT\nCollision: {result['raw_collision_rate']:.1%}",
    )
    _draw(
        axes[1], result["solved_boxes"], result["rooms"],
        f"AFTER LAYOUT SOLVER\nCollision: {result['solved_collision_rate']:.1%}",
    )

    width_m = result.get("plot_width_m")
    depth_m = result.get("plot_depth_m")
    subtitle = ""
    if width_m and depth_m:
        note = " (expanded - too many rooms for the baseline plot)" if result.get("plot_expanded") else ""
        subtitle = (
            f"\nRecommended plot: {width_m:.1f} m x {depth_m:.1f} m "
            f"(~{width_m*depth_m:.0f} m²){note}"
        )

    fig.suptitle(title + subtitle)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    demo_requirements = {
        "bedrooms": 3,
        "bathrooms": 2,
        "kitchens": 1,
        "living_rooms": 1,
        "balconies": 2,
        "storages": 1,
    }

    print("=" * 70)
    print("AI FLOOR PLANNER - FLOOR PLAN GENERATOR")
    print("=" * 70)

    print("\nRequirements:", demo_requirements)

    result = generate_floor_plan(demo_requirements)

    print(f"\nRooms generated              : {len(result['rooms'])}")
    print(f"Raw collision rate            : {result['raw_collision_rate']:.4f}")
    print(f"Solved collision rate         : {result['solved_collision_rate']:.4f}")
    print(f"Raw boundary violation rate   : {result['raw_boundary_violation_rate']:.4f}")
    print(f"Recommended plot               : {result['plot_width_m']:.1f} m x {result['plot_depth_m']:.1f} m"
          f" (~{result['plot_area_m2']:.0f} m²)"
          f"{' (expanded)' if result['plot_expanded'] else ''}")

    print(f"\nRooms moved: {len(result['move_log'])} / {len(result['rooms'])}")
    for entry in result["move_log"]:
        room = result["rooms"][entry["room_index"]]
        print(f"  {room['type']}_{room['room_index']:<2d} moved {entry['displacement']:.3f}"
              f"  reason: {entry['reason']}")

    print("\nFinal validation checklist:")
    for check, passed in result["validation_checklist"].items():
        print(f"  [{'x' if passed else ' '}] {check}")

    output_path = OUTPUT_DIR / "demo_plan.png"
    render(result, output_path)

    print(f"\nSaved visualization to: {output_path}")
    print("\nDone.")
