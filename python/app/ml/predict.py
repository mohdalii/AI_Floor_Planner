"""
End-to-end floor plan generation.

Takes room requirements (counts per room type + plot metadata) and
returns predicted room boxes: geometry model -> layout solver.

This is the first script in the project that generates a plan for
requirements that are NOT pulled from the training/test dataset -
every other script under python/app/training only replays fixed
dataset samples.
"""

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
    counts = [
        ("bedroom", 0, requirements.get("bedrooms", 0)),
        ("bathroom", 1, requirements.get("bathrooms", 0)),
        ("kitchen", 2, requirements.get("kitchens", 0)),
        ("living", 3, requirements.get("living_rooms", 0)),
        ("balcony", 4, requirements.get("balconies", 0)),
        ("storage", 5, requirements.get("storages", 0)),
        ("stair", 6, requirements.get("stairs", 0)),
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
            float(requirements.get("stairs", 0)),
        ],
        dtype=torch.float32,
    )


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
        solved = refine_layout(raw, room_types, mask)

    return {
        "rooms": rooms,
        "raw_boxes": raw[0].detach().cpu(),
        "solved_boxes": solved[0].detach().cpu(),
        "raw_collision_rate": collision_rate(raw, mask),
        "solved_collision_rate": collision_rate(solved, mask),
        "raw_boundary_violation_rate": boundary_violation_rate(raw, mask),
        "solved_boundary_violation_rate": boundary_violation_rate(solved, mask),
    }


def _draw(ax, boxes, rooms, title):
    ax.set_title(title, fontsize=10)

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

        ax.text(
            cx, cy,
            f"{room['type']}_{room['room_index']}",
            ha="center", va="center", fontsize=7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
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

    fig.suptitle(title)
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
        "stairs": 0,
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
    print(f"Solved boundary violation rate: {result['solved_boundary_violation_rate']:.4f}")

    output_path = OUTPUT_DIR / "demo_plan.png"
    render(result, output_path)

    print(f"\nSaved visualization to: {output_path}")
    print("\nDone.")
