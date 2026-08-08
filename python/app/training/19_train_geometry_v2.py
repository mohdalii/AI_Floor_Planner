import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TRAIN_PATH = ROOT / "dataset" / "geometry_v2_training" / "train.json"
VAL_PATH = ROOT / "dataset" / "geometry_v2_training" / "val.json"

MODEL_DIR = ROOT / "python" / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "room_geometry_v2.pt"

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

        global_features = torch.tensor(
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

        rooms = sample["rooms"]

        room_type_ids = []
        room_indices = []
        targets = []

        for room in rooms:

            room_type_ids.append(
                int(room["type_id"])
            )

            room_indices.append(
                min(
                    int(room["room_index"]),
                    MAX_ROOM_INDEX - 1,
                )
            )

            targets.append(
                room["bounds"]
            )

        return (
            global_features,
            torch.tensor(
                room_type_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                room_indices,
                dtype=torch.long,
            ),
            torch.tensor(
                targets,
                dtype=torch.float32,
            ),
        )


def collate_fn(batch):

    global_features = []
    room_types = []
    room_indices = []
    targets = []

    max_rooms = max(
        len(item[1])
        for item in batch
    )

    for global_x, types, indices, boxes in batch:

        global_features.append(global_x)

        pad_count = max_rooms - len(types)

        room_types.append(
            torch.cat(
                [
                    types,
                    torch.zeros(
                        pad_count,
                        dtype=torch.long,
                    ),
                ]
            )
        )

        room_indices.append(
            torch.cat(
                [
                    indices,
                    torch.zeros(
                        pad_count,
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
                        pad_count,
                        4,
                        dtype=torch.float32,
                    ),
                ],
                dim=0,
            )
        )

    return (
        torch.stack(global_features),
        torch.stack(room_types),
        torch.stack(room_indices),
        torch.stack(targets),
    )


class RoomGeometryV2(nn.Module):

    def __init__(self):

        super().__init__()

        self.global_encoder = nn.Sequential(
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

        self.index_embedding = nn.Embedding(
            MAX_ROOM_INDEX,
            16,
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                128 + 32 + 16,
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
            nn.Sigmoid(),
        )

    def forward(
        self,
        global_x,
        room_types,
        room_indices,
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

        return self.decoder(
            combined
        )


print("=" * 70)
print("AI FLOOR PLANNER - GEOMETRY V2 TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)

train_dataset = GeometryDataset(
    TRAIN_PATH
)

val_dataset = GeometryDataset(
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
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
)

model = RoomGeometryV2().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

loss_function = nn.SmoothL1Loss()

EPOCHS = 40

best_val_loss = float("inf")

print("\nStarting training...")
print("-" * 70)

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for (
        global_x,
        room_types,
        room_indices,
        targets,
    ) in train_loader:

        global_x = global_x.to(DEVICE)
        room_types = room_types.to(DEVICE)
        room_indices = room_indices.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(
            global_x,
            room_types,
            room_indices,
        )

        loss = loss_function(
            prediction,
            targets,
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for (
            global_x,
            room_types,
            room_indices,
            targets,
        ) in val_loader:

            global_x = global_x.to(DEVICE)
            room_types = room_types.to(DEVICE)
            room_indices = room_indices.to(DEVICE)
            targets = targets.to(DEVICE)

            prediction = model(
                global_x,
                room_types,
                room_indices,
            )

            loss = loss_function(
                prediction,
                targets,
            )

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

                "num_room_types":
                    NUM_ROOM_TYPES,

                "max_room_index":
                    MAX_ROOM_INDEX,
            },
            MODEL_PATH,
        )

        print(
            f"  -> Best model saved "
            f"(val={val_loss:.6f})"
        )


print("\n" + "=" * 70)
print("GEOMETRY V2 TRAINING COMPLETE")
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
