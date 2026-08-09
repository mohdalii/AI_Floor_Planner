import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TEST_PATH = ROOT / "dataset" / "geometry_v2_training" / "test.json"
MODEL_PATH = ROOT / "python" / "app" / "models" / "room_geometry_v3.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ROOM_TYPES = 8
MAX_ROOM_INDEX = 10

FEATURES = [
    "plot_area_norm",
    "wall_depth_norm",
    "bedrooms",
    "bathrooms",
    "kitchens",
    "living_rooms",
    "balconies",
    "storages",
    "stairs",
]


class GeometryDataset(Dataset):

    def __init__(self, path):

        with open(path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        inp = sample["input"]
        req = inp["requirements"]

        x = torch.tensor(
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

        types = []
        indices = []
        targets = []

        for room in sample["rooms"]:

            types.append(int(room["type_id"]))

            indices.append(
                min(
                    int(room["room_index"]),
                    MAX_ROOM_INDEX - 1,
                )
            )

            targets.append(
                room["bounds"]
            )

        return (
            x,
            torch.tensor(types, dtype=torch.long),
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(targets, dtype=torch.float32),
        )


def collate_fn(batch):

    max_rooms = max(
        len(item[1])
        for item in batch
    )

    xs = []
    types = []
    indices = []
    targets = []
    masks = []

    for x, room_types, room_indices, boxes in batch:

        count = len(room_types)
        pad = max_rooms - count

        xs.append(x)

        types.append(
            torch.cat(
                [
                    room_types,
                    torch.zeros(
                        pad,
                        dtype=torch.long,
                    ),
                ]
            )
        )

        indices.append(
            torch.cat(
                [
                    room_indices,
                    torch.zeros(
                        pad,
                        dtype=torch.long,
                    ),
                ]
            )
        )

        targets.append(
            torch.cat(
                [
                    boxes,
                    torch.zeros(
                        pad,
                        4,
                        dtype=torch.float32,
                    ),
                ],
                dim=0,
            )
        )

        masks.append(
            torch.cat(
                [
                    torch.ones(
                        count,
                        dtype=torch.float32,
                    ),
                    torch.zeros(
                        pad,
                        dtype=torch.float32,
                    ),
                ]
            )
        )

    return (
        torch.stack(xs),
        torch.stack(types),
        torch.stack(indices),
        torch.stack(targets),
        torch.stack(masks),
    )


class RoomGeometryV3(nn.Module):

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
            176,
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


def cxcywh_to_xyxy(boxes):

    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1,
    )


def box_iou(pred, target):

    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)

    iw = torch.relu(ix2 - ix1)
    ih = torch.relu(iy2 - iy1)

    intersection = iw * ih

    pred_area = (
        torch.relu(px2 - px1)
        * torch.relu(py2 - py1)
    )

    target_area = (
        torch.relu(tx2 - tx1)
        * torch.relu(ty2 - ty1)
    )

    union = (
        pred_area
        + target_area
        - intersection
        + 1e-8
    )

    return intersection / union


def collision_rate(boxes, mask):

    count = boxes.shape[0]

    if count < 2:
        return 0.0

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    collisions = 0
    pairs = 0

    for i in range(count):

        if mask[i] == 0:
            continue

        for j in range(i + 1, count):

            if mask[j] == 0:
                continue

            ix1 = max(
                x1[i].item(),
                x1[j].item(),
            )

            iy1 = max(
                y1[i].item(),
                y1[j].item(),
            )

            ix2 = min(
                x2[i].item(),
                x2[j].item(),
            )

            iy2 = min(
                y2[i].item(),
                y2[j].item(),
            )

            overlap = (
                max(0.0, ix2 - ix1)
                * max(0.0, iy2 - iy1)
            )

            pairs += 1

            if overlap > 0.005:
                collisions += 1

    if pairs == 0:
        return 0.0

    return collisions / pairs


print("=" * 70)
print("AI FLOOR PLANNER - GEOMETRY V3 TEST")
print("=" * 70)

print("\nDevice:", DEVICE)

dataset = GeometryDataset(
    TEST_PATH
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
)

model = RoomGeometryV3().to(DEVICE)

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
    "\nCheckpoint epoch:",
    checkpoint.get("epoch"),
)

print(
    "Best validation loss:",
    checkpoint.get("best_val_loss"),
)

total_abs_error = 0.0
total_coordinates = 0

total_iou = 0.0
total_rooms = 0

collision_rates = []

room_type_errors = {
    i: []
    for i in range(NUM_ROOM_TYPES)
}

with torch.no_grad():

    for (
        x,
        room_types,
        room_indices,
        targets,
        masks,
    ) in loader:

        x = x.to(DEVICE)
        room_types = room_types.to(DEVICE)
        room_indices = room_indices.to(DEVICE)
        targets = targets.to(DEVICE)
        masks = masks.to(DEVICE)

        padding_mask = masks == 0

        prediction = model(
            x,
            room_types,
            room_indices,
            padding_mask,
        )

        prediction = cxcywh_to_xyxy(
            prediction
        )

        prediction = prediction.clamp(
            0.0,
            1.0,
        )

        error = torch.abs(
            prediction - targets
        )

        mask = masks.unsqueeze(-1)

        total_abs_error += (
            error * mask
        ).sum().item()

        total_coordinates += (
            masks.sum().item() * 4
        )

        ious = box_iou(
            prediction,
            targets,
        )

        total_iou += (
            ious * masks
        ).sum().item()

        total_rooms += (
            masks.sum().item()
        )

        for type_id in range(
            NUM_ROOM_TYPES
        ):

            type_mask = (
                (room_types == type_id)
                & (masks > 0)
            )

            if type_mask.any():

                type_error = (
                    error.mean(dim=-1)
                    [type_mask]
                    .cpu()
                    .tolist()
                )

                room_type_errors[
                    type_id
                ].extend(type_error)

        for batch_index in range(
            prediction.shape[0]
        ):

            valid = masks[
                batch_index
            ]

            rate = collision_rate(
                prediction[
                    batch_index
                ].cpu(),
                valid.cpu(),
            )

            collision_rates.append(
                rate
            )


mae = (
    total_abs_error
    / max(total_coordinates, 1)
)

mean_iou = (
    total_iou
    / max(total_rooms, 1)
)

mean_collision_rate = (
    sum(collision_rates)
    / max(len(collision_rates), 1)
)

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

print("\n" + "=" * 70)
print("GEOMETRY V3 TEST RESULTS")
print("=" * 70)

print(
    f"\nTest samples          : {len(dataset)}"
)

print(
    f"Rooms evaluated       : {int(total_rooms)}"
)

print(
    f"Coordinate MAE        : {mae:.6f}"
)

print(
    f"Approx pixel MAE      : "
    f"{mae * 256:.2f} pixels"
)

print(
    f"Mean box IoU          : "
    f"{mean_iou:.6f}"
)

print(
    f"Mean collision rate   : "
    f"{mean_collision_rate:.6f}"
)

print("\nPer-room-type error:")
print("-" * 70)

for type_id, name in enumerate(
    ROOM_TYPES
):

    values = room_type_errors[type_id]

    if values:

        avg = (
            sum(values)
            / len(values)
        )

        print(
            f"{name:15s} "
            f"| rooms={len(values):5d} "
            f"| MAE={avg:.6f}"
        )

print("\nModel:", MODEL_PATH)

print("\nTEST COMPLETE")
print("=" * 70)

print("\nDone.")
