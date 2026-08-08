import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TEST_PATH = ROOT / "dataset" / "training" / "test.json"
MODEL_PATH = ROOT / "python" / "app" / "models" / "room_count_baseline.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


print("=" * 70)
print("AI FLOOR PLANNER - BASELINE EVALUATION")
print("=" * 70)

print("\nDevice:", DEVICE)

dataset = FloorPlanDataset(TEST_PATH)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
)

model = RoomCountModel().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

total_absolute_error = 0.0
total_values = 0
exact_values = 0
exact_plans = 0

print("\nEvaluating test dataset...")
print("-" * 70)

with torch.no_grad():

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        prediction = model(x)

        predicted_counts = torch.round(
            prediction
        ).clamp(min=0)

        absolute_error = torch.abs(
            predicted_counts - y
        )

        total_absolute_error += (
            absolute_error.sum().item()
        )

        total_values += y.numel()

        exact_values += (
            (predicted_counts == y)
            .all(dim=1)
            .sum()
            .item()
        )

        exact_plans += (
            (predicted_counts == y)
            .all(dim=1)
            .sum()
            .item()
        )

mae = total_absolute_error / total_values
exact_accuracy = exact_plans / len(dataset)

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(f"\nTest samples        : {len(dataset)}")
print(f"Mean absolute error : {mae:.4f}")
print(f"Exact room-count accuracy : {exact_accuracy:.2%}")

# ------------------------------------------------------------
# Show real examples
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

shown = 0

with torch.no_grad():

    for x, y in loader:

        x = x.to(DEVICE)

        prediction = model(x)

        predicted_counts = torch.round(
            prediction
        ).clamp(min=0).cpu()

        y = y.cpu()

        for i in range(len(x)):

            print(
                f"\nSample {shown + 1}"
            )

            print(
                "Actual    :",
                y[i].int().tolist()
            )

            print(
                "Predicted :",
                predicted_counts[i].int().tolist()
            )

            shown += 1

            if shown >= 10:
                break

        if shown >= 10:
            break

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
