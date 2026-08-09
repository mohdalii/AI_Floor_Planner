import json
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(".")

TEST_PATH = ROOT / "dataset" / "geometry_v2_training" / "test.json"
MODEL_PATH = ROOT / "python" / "app" / "models" / "room_geometry_v4.pt"
OUTPUT_DIR = ROOT / "dataset" / "geometry_v3_predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ------------------------------------------------------------
# IMPORT LAYOUT SOLVER
# ------------------------------------------------------------

import sys

SOLVER_DIR = ROOT / "python" / "app" / "training"

if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from geometry_layout_solver import (
    refine_layout,
    collision_rate,
)

# ------------------------------------------------------------
# V3 MODEL
# ------------------------------------------------------------

class RoomGeometryV4(nn.Module):

    def __init__(self):
        super().__init__()

        self.global_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.room_embedding = nn.Embedding(
            NUM_ROOM_TYPES,
            32,
        )

        self.index_embedding = nn.Embedding(
            MAX_ROOM_INDEX,
            16,
        )

        self.input_projection = nn.Linear(
            128 + 32 + 16,
            128,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        global_x,
        room_types,
        room_indices,
        padding_mask,
    ):

        global_features = self.global_encoder(
            global_x
        )

        global_features = (
            global_features
            .unsqueeze(1)
            .expand(
                -1,
                room_types.shape[1],
                -1,
            )
        )

        room_features = self.room_embedding(
            room_types
        )

        index_features = self.index_embedding(
            room_indices
        )

        combined = torch.cat(
            [
                global_features,
                room_features,
                index_features,
            ],
            dim=-1,
        )

        tokens = self.input_projection(
            combined
        )

        tokens = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        return self.decoder(tokens)

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# DRAW FUNCTION
# ------------------------------------------------------------

def draw_layout(
    ax,
    boxes,
    rooms,
    title,
):

    ax.set_title(title)

    for i, room in enumerate(rooms):

        cx, cy, w, h = [
            float(v)
            for v in boxes[i]
        ]

        x1 = cx - w / 2
        y1 = cy - h / 2

        width = max(w, 0.005)
        height = max(h, 0.005)

        rect = Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)

        label = (
            f"{room['type']}_"
            f"{room['room_index']}"
        )

        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

# ------------------------------------------------------------
# START
# ------------------------------------------------------------

print("=" * 70)
print("AI FLOOR PLANNER - V4 + LAYOUT SOLVER")
print("=" * 70)

print("\nDevice:", DEVICE)

with open(
    TEST_PATH,
    "r",
    encoding="utf-8",
) as f:

    samples = json.load(f)

print("Test samples:", len(samples))

model = RoomGeometryV4().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch"),
)

print(
    "Best validation loss:",
    checkpoint.get("best_val_loss"),
)

print("\nGenerating solved layouts...")
print("-" * 70)

raw_collision_total = 0.0
solved_collision_total = 0.0

for sample_index in range(
    min(10, len(samples))
):

    sample = samples[sample_index]
    rooms = sample["rooms"]

    x = (
        build_features(sample)
        .unsqueeze(0)
        .to(DEVICE)
    )

    room_types = torch.tensor(
        [
            int(room["type_id"])
            for room in rooms
        ],
        dtype=torch.long,
    ).unsqueeze(0).to(DEVICE)

    room_indices = torch.tensor(
        [
            min(
                int(room["room_index"]),
                MAX_ROOM_INDEX - 1,
            )
            for room in rooms
        ],
        dtype=torch.long,
    ).unsqueeze(0).to(DEVICE)

    mask = torch.ones(
        1,
        len(rooms),
        dtype=torch.float32,
        device=DEVICE,
    )

    padding_mask = torch.zeros(
        1,
        len(rooms),
        dtype=torch.bool,
        device=DEVICE,
    )

    with torch.no_grad():

        raw_prediction = model(
            x,
            room_types,
            room_indices,
            padding_mask,
        )

    # --------------------------------------------------------
    # RAW MODEL COLLISION
    # --------------------------------------------------------

    raw_collision = collision_rate(
        raw_prediction,
        mask,
    )

    # --------------------------------------------------------
    # SOLVE LAYOUT
    # --------------------------------------------------------

    solved_prediction = refine_layout(
        raw_prediction,
        room_types,
        mask,
        iterations=120,
        margin=0.010,
    )

    solved_collision = collision_rate(
        solved_prediction,
        mask,
    )

    raw_collision_total += raw_collision
    solved_collision_total += solved_collision

    raw_boxes = raw_prediction[0].detach().cpu()
    solved_boxes = solved_prediction[0].detach().cpu()

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 8),
    )

    # Ground truth
    ground_truth_boxes = torch.tensor(
        [
            room["bounds"]
            for room in rooms
        ],
        dtype=torch.float32,
    )

    # Ground truth is XYXY.
    ax = axes[0]

    ax.set_title(
        f"GROUND TRUTH - Plan {sample['plan_id']}"
    )

    for i, room in enumerate(rooms):

        x1, y1, x2, y2 = [
            float(v)
            for v in ground_truth_boxes[i]
        ]

        width = max(
            x2 - x1,
            0.001,
        )

        height = max(
            y2 - y1,
            0.001,
        )

        rect = Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rect)

        label = (
            f"{room['type']}_"
            f"{room['room_index']}"
        )

        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            label,
            ha="center",
            va="center",
            fontsize=7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Raw V3
    draw_layout(
        axes[1],
        raw_boxes,
        rooms,
        f"RAW V3\nCollision: {raw_collision:.1%}",
    )

    # Solved
    draw_layout(
        axes[2],
        solved_boxes,
        rooms,
        f"V4 + LAYOUT SOLVER\nCollision: {solved_collision:.1%}",
    )

    output_path = (
        OUTPUT_DIR
        / f"comparison_{sample_index + 1}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Plan {sample['plan_id']}: "
        f"raw={raw_collision:.2%} "
        f"-> solved={solved_collision:.2%}"
    )

    print(
        f"Created: {output_path}"
    )

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

count = min(10, len(samples))

print("\n" + "=" * 70)
print("V4 + LAYOUT SOLVER COMPLETE")
print("=" * 70)

print(
    f"\nAverage raw collision    : "
    f"{raw_collision_total / max(count, 1):.2%}"
)

print(
    f"Average solved collision : "
    f"{solved_collision_total / max(count, 1):.2%}"
)

print(
    "\nImages saved to:",
    OUTPUT_DIR,
)

print("\nDone.")

