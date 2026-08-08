import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TRAIN_PATH = ROOT / "dataset" / "geometry_training" / "train.json"
VAL_PATH = ROOT / "dataset" / "geometry_training" / "val.json"

MODEL_DIR = ROOT / "python" / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "room_aware_geometry.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_ROOMS = 23
NUM_ROOM_TYPES = 8

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


class RoomGeometryDataset(Dataset):

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

        room_types = torch.zeros(
            MAX_ROOMS,
            dtype=torch.long,
        )

        targets = torch.zeros(
            MAX_ROOMS,
            4,
            dtype=torch.float32,
        )

        mask = torch.zeros(
            MAX_ROOMS,
            dtype=torch.float32,
        )

        rooms = sample["rooms"]

        for i, room in enumerate(
            rooms[:MAX_ROOMS]
        ):

            room_types[i] = int(
                room["type_id"]
            )

            bounds = room["bounds"]

            targets[i] = torch.tensor(
                bounds,
                dtype=torch.float32,
            )

            mask[i] = 1.0

        return (
            torch.tensor(
                x,
                dtype=torch.float32,
            ),
            room_types,
            targets,
            mask,
        )


class RoomAwareGeometryModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_encoder = nn.Sequential(

            nn.Linear(
                len(FEATURES),
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),

            nn.ReLU(),
        )

        self.room_embedding = nn.Embedding(
            NUM_ROOM_TYPES,
            32,
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                128 + 32,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                4,
            ),
        )

    def forward(
        self,
        x,
        room_types,
    ):

        global_features = self.input_encoder(x)

        global_features = global_features.unsqueeze(1)

        global_features = global_features.expand(
            -1,
            MAX_ROOMS,
            -1,
        )

        room_features = self.room_embedding(
            room_types
        )

        combined = torch.cat(
            [
                global_features,
                room_features,
            ],
            dim=-1,
        )

        return self.decoder(
            combined
        )


print("=" * 70)
print("AI FLOOR PLANNER - ROOM-AWARE GEOMETRY MODEL")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading datasets...")

train_dataset = RoomGeometryDataset(
    TRAIN_PATH
)

val_dataset = RoomGeometryDataset(
    VAL_PATH
)

print(
    "Train samples:",
    len(train_dataset),
)

print(
    "Val samples  :",
    len(val_dataset),
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

model = RoomAwareGeometryModel().to(
    DEVICE
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

loss_function = nn.SmoothL1Loss(
    reduction="none"
)

EPOCHS = 30

print("\nStarting training...")
print("-" * 70)

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for x, room_types, targets, mask in train_loader:

        x = x.to(DEVICE)
        room_types = room_types.to(DEVICE)
        targets = targets.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(
            x,
            room_types,
        )

        raw_loss = loss_function(
            prediction,
            targets,
        )

        room_mask = mask.unsqueeze(-1)

        loss = (
            raw_loss * room_mask
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

        for x, room_types, targets, mask in val_loader:

            x = x.to(DEVICE)
            room_types = room_types.to(DEVICE)
            targets = targets.to(DEVICE)
            mask = mask.to(DEVICE)

            prediction = model(
                x,
                room_types,
            )

            raw_loss = loss_function(
                prediction,
                targets,
            )

            room_mask = mask.unsqueeze(-1)

            loss = (
                raw_loss * room_mask
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

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "features":
                    FEATURES,

                "max_rooms":
                    MAX_ROOMS,

                "num_room_types":
                    NUM_ROOM_TYPES,
            },
            MODEL_PATH,
        )

        print(
            f"  -> Best model saved "
            f"(val={val_loss:.6f})"
        )


print("\n" + "=" * 70)
print("ROOM-AWARE GEOMETRY TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest validation loss : "
    f"{best_val_loss:.6f}"
)

print(
    f"Model saved          : "
    f"{MODEL_PATH}"
)

print("\nDone.")
