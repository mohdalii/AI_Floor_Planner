import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TRAIN_PATH = ROOT / "dataset" / "training" / "train.json"
VAL_PATH = ROOT / "dataset" / "training" / "val.json"

MODEL_DIR = ROOT / "python" / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_ROOMS = 23

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

ROOM_TO_ID = {
    room: index
    for index, room in enumerate(ROOM_TYPES)
}

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

        x = [
            float(inp.get("plot_area_norm", 0) or 0),
            float(inp.get("wall_depth_norm", 0) or 0),

            float(req.get("bedrooms", 0)),
            float(req.get("bathrooms", 0)),
            float(req.get("kitchens", 0)),
            float(req.get("living_rooms", 0)),
            float(req.get("balconies", 0)),
            float(req.get("storages", 0)),
            float(req.get("stairs", 0)),
        ]

        room_features = torch.zeros(
            MAX_ROOMS,
            5,
            dtype=torch.float32,
        )

        room_mask = torch.zeros(
            MAX_ROOMS,
            dtype=torch.float32,
        )

        rooms = sample["target"]["rooms"]

        # Sort so room ordering is deterministic
        rooms = sorted(
            rooms,
            key=lambda r: str(r.get("id", ""))
        )

        for i, room in enumerate(
            rooms[:MAX_ROOMS]
        ):

            room_type = room.get("type")

            room_id = ROOM_TO_ID.get(
                room_type,
                0
            )

            bounds = room.get("bounds_norm")

            if not bounds:

                bounds = room.get("bounds")

                if bounds and len(bounds) == 4:

                    bounds = [
                        float(bounds[0]) / 256.0,
                        float(bounds[1]) / 256.0,
                        float(bounds[2]) / 256.0,
                        float(bounds[3]) / 256.0,
                    ]

            if not bounds:
                continue

            room_features[i] = torch.tensor(
                [
                    float(room_id) / len(ROOM_TYPES),
                    float(bounds[0]),
                    float(bounds[1]),
                    float(bounds[2]),
                    float(bounds[3]),
                ],
                dtype=torch.float32,
            )

            room_mask[i] = 1.0

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            room_features,
            room_mask,
        )


class GeometryModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                len(FEATURES),
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),
        )

        self.room_decoder = nn.Sequential(

            nn.Linear(
                128,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                MAX_ROOMS * 5
            ),
        )

    def forward(self, x):

        hidden = self.encoder(x)

        output = self.room_decoder(hidden)

        return output.view(
            -1,
            MAX_ROOMS,
            5
        )


print("=" * 70)
print("AI FLOOR PLANNER - GEOMETRY MODEL")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading datasets...")

train_dataset = GeometryDataset(
    TRAIN_PATH
)

val_dataset = GeometryDataset(
    VAL_PATH
)

print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Val samples  :",
    len(val_dataset)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
)

model = GeometryModel().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

loss_function = nn.SmoothL1Loss()

EPOCHS = 20

print("\nStarting geometry training...")
print("-" * 70)

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for x, target, mask in train_loader:

        x = x.to(DEVICE)
        target = target.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(x)

        # Only calculate geometry loss
        # for rooms that actually exist.
        geometry_prediction = prediction[:, :, 1:]

        geometry_target = target[:, :, 1:]

        room_mask = mask.unsqueeze(-1)

        loss = (
            torch.abs(
                geometry_prediction
                - geometry_target
            )
            * room_mask
        ).sum()

        normalizer = (
            room_mask.sum() * 4
        ).clamp(min=1.0)

        loss = loss / normalizer

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, target, mask in val_loader:

            x = x.to(DEVICE)
            target = target.to(DEVICE)
            mask = mask.to(DEVICE)

            prediction = model(x)

            geometry_prediction = prediction[:, :, 1:]
            geometry_target = target[:, :, 1:]

            room_mask = mask.unsqueeze(-1)

            loss = (
                torch.abs(
                    geometry_prediction
                    - geometry_target
                )
                * room_mask
            ).sum()

            normalizer = (
                room_mask.sum() * 4
            ).clamp(min=1.0)

            loss = loss / normalizer

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f}"
    )

MODEL_PATH = (
    MODEL_DIR /
    "geometry_baseline.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "features":
            FEATURES,

        "max_rooms":
            MAX_ROOMS,

        "room_types":
            ROOM_TYPES,
    },
    MODEL_PATH,
)

print("\n" + "=" * 70)
print("GEOMETRY TRAINING COMPLETE")
print("=" * 70)

print("\nModel saved:")
print(MODEL_PATH)

print("\nDone.")
