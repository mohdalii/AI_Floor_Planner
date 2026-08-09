import json
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path("D:/ai-floor-planner")

TEST_PATH = ROOT / "dataset/geometry_v2_training/test.json"
MODEL_PATH = ROOT / "python/app/models/room_geometry_v4.pt"
OUTPUT_DIR = ROOT / "dataset/geometry_v4_diagnostic"

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
        global_features = self.global_encoder(global_x)

        global_features = (
            global_features
            .unsqueeze(1)
            .expand(
                -1,
                room_types.shape[1],
                -1,
            )
        )

        room_features = self.room_embedding(room_types)
        index_features = self.index_embedding(room_indices)

        combined = torch.cat(
            [
                global_features,
                room_features,
                index_features,
            ],
            dim=-1,
        )

        tokens = self.input_projection(combined)

        tokens = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        return self.decoder(tokens)


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

    return (
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    )


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = cxcywh_to_xyxy(a)
    bx1, by1, bx2, by2 = cxcywh_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / max(union, 1e-8)


def overlap(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]

    ow = min(
        ax + aw / 2,
        bx + bw / 2,
    ) - max(
        ax - aw / 2,
        bx - bw / 2,
    )

    oh = min(
        ay + ah / 2,
        by + bh / 2,
    ) - max(
        ay - ah / 2,
        by - bh / 2,
    )

    return ow, oh


def collision_rate(boxes):
    total = 0
    bad = 0

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            total += 1

            ow, oh = overlap(
                boxes[i],
                boxes[j],
            )

            if ow > 1e-5 and oh > 1e-5:
                bad += 1

    return bad / max(total, 1)


def boundary_violation(boxes):
    violations = 0

    for box in boxes:
        cx, cy, w, h = [float(v) for v in box]

        if (
            cx - w / 2 < 0
            or cy - h / 2 < 0
            or cx + w / 2 > 1
            or cy + h / 2 > 1
        ):
            violations += 1

    return violations / max(len(boxes), 1)


def center_distance(a, b):
    return (
        (
            float(a[0]) - float(b[0])
        ) ** 2
        +
        (
            float(a[1]) - float(b[1])
        ) ** 2
    ) ** 0.5


print("=" * 70)
print("AI FLOOR PLANNER - V4 RAW vs SOLVER DIAGNOSTIC")
print("=" * 70)

print("Device:", DEVICE)
print("Model:", MODEL_PATH)
print("Test:", TEST_PATH)

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


# Import CURRENT solver.
import sys

sys.path.insert(
    0,
    str(ROOT / "python/app/training")
)

from geometry._layout._solver import (
    refine_layout,
)


raw_mae_total = 0.0
solved_mae_total = 0.0

raw_iou_total = 0.0
solved_iou_total = 0.0

raw_collision_total = 0.0
solved_collision_total = 0.0

raw_boundary_total = 0.0
solved_boundary_total = 0.0

center_shift_total = 0.0
area_change_total = 0.0

room_count = 0
sample_count = 0


for sample_index, sample in enumerate(samples):

    rooms = sample["rooms"]

    if not rooms:
        continue

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
        dtype=torch.bool,
        device=DEVICE,
    )

    with torch.no_grad():

        raw = model(
            x,
            room_types,
            room_indices,
            mask.logical_not(),
        )

        solved = refine_layout(
            raw,
            room_types,
            mask,
        )

    raw_boxes = raw[0].detach().cpu()
    solved_boxes = solved[0].detach().cpu()

    gt_boxes = torch.tensor(
        [
            room["bounds"]
            for room in rooms
        ],
        dtype=torch.float32,
    )

    for i in range(len(rooms)):

        raw_box = raw_boxes[i]
        solved_box = solved_boxes[i]
        gt_box = gt_boxes[i]

        raw_mae_total += float(
            torch.abs(
                raw_box - gt_box
            ).mean()
        )

        solved_mae_total += float(
            torch.abs(
                solved_box - gt_box
            ).mean()
        )

        raw_iou_total += box_iou(
            raw_box.tolist(),
            gt_box.tolist(),
        )

        solved_iou_total += box_iou(
            solved_box.tolist(),
            gt_box.tolist(),
        )

        center_shift_total += center_distance(
            raw_box,
            solved_box,
        )

        raw_area = float(
            raw_box[2] * raw_box[3]
        )

        solved_area = float(
            solved_box[2] * solved_box[3]
        )

        area_change_total += abs(
            solved_area - raw_area
        )

        room_count += 1

    raw_collision_total += collision_rate(
        raw_boxes.tolist()
    )

    solved_collision_total += collision_rate(
        solved_boxes.tolist()
    )

    raw_boundary_total += boundary_violation(
        raw_boxes.tolist()
    )

    solved_boundary_total += boundary_violation(
        solved_boxes.tolist()
    )

    sample_count += 1

    if sample_index < 10:
        print(
            f"Plan {sample['plan_id']}: "
            f"RAW IoU={raw_iou_total / room_count:.4f} "
            f"SOLVED IoU={solved_iou_total / room_count:.4f}"
        )


print()
print("=" * 70)
print("FINAL DIAGNOSTIC RESULTS")
print("=" * 70)

print()
print("Samples:", sample_count)
print("Rooms:", room_count)

print()
print("RAW V4")
print(
    "Coordinate MAE        :",
    f"{raw_mae_total / max(room_count, 1):.6f}",
)
print(
    "Mean box IoU           :",
    f"{raw_iou_total / max(room_count, 1):.6f}",
)
print(
    "Collision rate         :",
    f"{raw_collision_total / max(sample_count, 1):.6f}",
)
print(
    "Boundary violation     :",
    f"{raw_boundary_total / max(sample_count, 1):.6f}",
)

print()
print("V4 + CURRENT SOLVER")
print(
    "Coordinate MAE        :",
    f"{solved_mae_total / max(room_count, 1):.6f}",
)
print(
    "Mean box IoU           :",
    f"{solved_iou_total / max(room_count, 1):.6f}",
)
print(
    "Collision rate         :",
    f"{solved_collision_total / max(sample_count, 1):.6f}",
)
print(
    "Boundary violation     :",
    f"{solved_boundary_total / max(sample_count, 1):.6f}",
)

print()
print("RAW -> SOLVED")
print(
    "Mean center displacement:",
    f"{center_shift_total / max(room_count, 1):.6f}",
)
print(
    "Mean absolute area change:",
    f"{area_change_total / max(room_count, 1):.6f}",
)

print()
print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)

raw_mae = raw_mae_total / max(room_count, 1)
solved_mae = solved_mae_total / max(room_count, 1)

raw_iou = raw_iou_total / max(room_count, 1)
solved_iou = solved_iou_total / max(room_count, 1)

raw_collision = raw_collision_total / max(sample_count, 1)
solved_collision = solved_collision_total / max(sample_count, 1)

print()

if solved_collision < raw_collision:
    print("Collision: IMPROVED")
else:
    print("Collision: NOT IMPROVED")

if solved_iou > raw_iou:
    print("IoU: IMPROVED")
else:
    print("IoU: WORSE")

if solved_mae < raw_mae:
    print("MAE: IMPROVED")
else:
    print("MAE: WORSE")

print()
print(
    "If collision improves while IoU worsens and MAE worsens,"
)
print(
    "the current solver is damaging the neural prediction."
)

print()
print("Done.")
