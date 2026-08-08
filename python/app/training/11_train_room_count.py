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

print("=" * 70)
print("AI FLOOR PLANNER - BASELINE MODEL")
print("=" * 70)

print("\nDevice:", DEVICE)

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


class FloorPlanDataset(Dataset):

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

        target = [0, 0, 0, 0, 0]

        for room in sample["target"]["rooms"]:

            room_type = room.get("type")

            if room_type == "bedroom":
                target[0] += 1

            elif room_type == "bathroom":
                target[1] += 1

            elif room_type == "kitchen":
                target[2] += 1

            elif room_type == "living":
                target[3] += 1

            elif room_type == "balcony":
                target[4] += 1

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


class RoomCountModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(len(FEATURES), 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 5),
        )

    def forward(self, x):
        return self.network(x)


print("\nLoading datasets...")

train_dataset = FloorPlanDataset(TRAIN_PATH)
val_dataset = FloorPlanDataset(VAL_PATH)

print("Train samples:", len(train_dataset))
print("Val samples  :", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
)

model = RoomCountModel().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

loss_function = nn.MSELoss()

EPOCHS = 20

print("\nStarting training...")
print("-" * 70)

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(x)

        loss = loss_function(
            prediction,
            y,
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(x)

            loss = loss_function(
                prediction,
                y,
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )

MODEL_PATH = MODEL_DIR / "room_count_baseline.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "features": FEATURES,
    },
    MODEL_PATH,
)

print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print("\nModel saved:")
print(MODEL_PATH)

print("\nDone.")
