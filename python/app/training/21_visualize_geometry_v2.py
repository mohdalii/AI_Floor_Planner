import json
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(".")

TEST_PATH = ROOT / "dataset" / "geometry_v2_training" / "test.json"
MODEL_PATH = ROOT / "python" / "app" / "models" / "room_geometry_v2.pt"

OUTPUT_DIR = ROOT / "dataset" / "geometry_v2_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

NUM_ROOM_TYPES = 8
MAX_ROOM_INDEX = 10


class RoomGeometryV2(nn.Module):

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
            32
        )

        self.index_embedding = nn.Embedding(
            MAX_ROOM_INDEX,
            16
        )

        self.decoder = nn.Sequential(
            nn.Linear(176, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
        room_types,
        room_indices
    ):

        global_features = self.global_encoder(x)

        global_features = (
            global_features
            .unsqueeze(1)
            .expand(
                -1,
                room_types.shape[1],
                -1
            )
        )

        room_features = self.room_embedding(
            room_types
        )

        index_features = self.index_embedding(
            room_indices
        )

        combined = torch.cat([
            global_features,
            room_features,
            index_features
        ], dim=-1)

        return self.decoder(combined)


def build_features(sample):

    inp = sample["input"]
    req = inp["requirements"]

    return torch.tensor([

        float(inp.get("plot_area_norm", 0) or 0),

        float(inp.get("wall_depth_norm", 0) or 0),

        float(req.get("bedrooms", 0)),

        float(req.get("bathrooms", 0)),

        float(req.get("kitchens", 0)),

        float(req.get("living_rooms", 0)),

        float(req.get("balconies", 0)),

        float(req.get("storages", 0)),

        float(req.get("stairs", 0)),

    ], dtype=torch.float32)


print("=" * 70)
print("AI FLOOR PLANNER - GEOMETRY V2 VISUAL CHECK")
print("=" * 70)

print("\nDevice:", DEVICE)

with open(
    TEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    samples = json.load(f)

print("Test samples:", len(samples))

model = RoomGeometryV2().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# Check first 5 test plans
for sample_index in range(min(5, len(samples))):

    sample = samples[sample_index]

    x = build_features(sample).unsqueeze(0).to(DEVICE)

    rooms = sample["rooms"]

    room_types = torch.tensor(
        [
            int(room["type_id"])
            for room in rooms
        ],
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    room_indices = torch.tensor(
        [
            min(
                int(room["room_index"]),
                MAX_ROOM_INDEX - 1
            )
            for room in rooms
        ],
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        prediction = model(
            x,
            room_types,
            room_indices
        )[0].cpu().numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8)
    )

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    ax = axes[0]

    ax.set_title(
        f"GROUND TRUTH - Plan {sample['plan_id']}"
    )

    for room in rooms:

        bounds = room["bounds"]

        x1, y1, x2, y2 = bounds

        width = x2 - x1
        height = y2 - y1

        rect = Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2
        )

        ax.add_patch(rect)

        label = (
            f"{room['type']}_"
            f"{room['room_index']}"
        )

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=8
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_aspect("equal")

    ax.grid(True)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    ax = axes[1]

    ax.set_title(
        "AI PREDICTION"
    )

    for i, room in enumerate(rooms):

        x1, y1, x2, y2 = prediction[i]

        width = max(
            x2 - x1,
            0.005
        )

        height = max(
            y2 - y1,
            0.005
        )

        rect = Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2
        )

        ax.add_patch(rect)

        label = (
            f"{room['type']}_"
            f"{room['room_index']}"
        )

        cx = x1 + width / 2
        cy = y1 + height / 2

        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=8
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_aspect("equal")

    ax.grid(True)

    output_path = (
        OUTPUT_DIR /
        f"comparison_{sample_index + 1}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Created: {output_path}"
    )


print("\n" + "=" * 70)
print("VISUAL CHECK COMPLETE")
print("=" * 70)

print(
    "\nImages saved to:",
    OUTPUT_DIR
)

print("\nDone.")
