import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TEST_PATH = ROOT / "dataset" / "geometry_v2_training" / "test.json"
MODEL_PATH = ROOT / "python" / "app" / "models" / "room_geometry_v2.pt"

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

        x = torch.tensor([
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

        types = []
        indices = []
        boxes = []

        for room in sample["rooms"]:

            types.append(int(room["type_id"]))

            indices.append(
                min(
                    int(room["room_index"]),
                    MAX_ROOM_INDEX - 1
                )
            )

            boxes.append(room["bounds"])

        return (
            x,
            torch.tensor(types, dtype=torch.long),
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(boxes, dtype=torch.float32),
        )


def collate_fn(batch):

    max_rooms = max(
        len(item[1])
        for item in batch
    )

    xs = []
    types = []
    indices = []
    boxes = []
    masks = []

    for x, room_types, room_indices, room_boxes in batch:

        count = len(room_types)
        pad = max_rooms - count

        xs.append(x)

        types.append(
            torch.cat([
                room_types,
                torch.zeros(
                    pad,
                    dtype=torch.long
                )
            ])
        )

        indices.append(
            torch.cat([
                room_indices,
                torch.zeros(
                    pad,
                    dtype=torch.long
                )
            ])
        )

        boxes.append(
            torch.cat([
                room_boxes,
                torch.zeros(
                    pad,
                    4,
                    dtype=torch.float32
                )
            ])
        )

        masks.append(
            torch.cat([
                torch.ones(
                    count,
                    dtype=torch.float32
                ),
                torch.zeros(
                    pad,
                    dtype=torch.float32
                )
            ])
        )

    return (
        torch.stack(xs),
        torch.stack(types),
        torch.stack(indices),
        torch.stack(boxes),
        torch.stack(masks),
    )


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


print("=" * 70)
print("AI FLOOR PLANNER - GEOMETRY V2 TEST")
print("=" * 70)

print("\nDevice:", DEVICE)

dataset = GeometryDataset(TEST_PATH)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)

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

absolute_error = 0.0
total_coordinates = 0

room_errors = []

print("\nEvaluating test dataset...")
print("-" * 70)

with torch.no_grad():

    for (
        x,
        room_types,
        room_indices,
        targets,
        masks
    ) in loader:

        x = x.to(DEVICE)
        room_types = room_types.to(DEVICE)
        room_indices = room_indices.to(DEVICE)
        targets = targets.to(DEVICE)
        masks = masks.to(DEVICE)

        prediction = model(
            x,
            room_types,
            room_indices
        )

        error = torch.abs(
            prediction - targets
        )

        mask = masks.unsqueeze(-1)

        absolute_error += (
            error * mask
        ).sum().item()

        total_coordinates += (
            mask.sum().item() * 4
        )

        room_errors.extend(
            (
                error.mean(dim=-1)
                * masks
            )
            .flatten()
            .cpu()
            .tolist()
        )

mae = absolute_error / total_coordinates

valid_errors = [
    e for e in room_errors
    if e > 0
]

print("\n" + "=" * 70)
print("GEOMETRY V2 TEST RESULTS")
print("=" * 70)

print(
    f"\nTest samples          : {len(dataset)}"
)

print(
    f"Coordinate MAE        : {mae:.6f}"
)

print(
    f"Average room error    : "
    f"{sum(valid_errors) / len(valid_errors):.6f}"
)

print(
    f"Approx pixel MAE      : "
    f"{mae * 256:.2f} pixels"
)

print("\nModel:", MODEL_PATH)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print("\nDone.")
