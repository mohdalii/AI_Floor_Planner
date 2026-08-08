import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TEST_PATH = ROOT / "dataset" / "training" / "test.json"
MODEL_PATH = ROOT / "python" / "app" / "models" / "geometry_baseline.pt"

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

        target = torch.zeros(
            MAX_ROOMS,
            4,
            dtype=torch.float32,
        )

        mask = torch.zeros(
            MAX_ROOMS,
            dtype=torch.float32,
        )

        rooms = sorted(
            sample["target"]["rooms"],
            key=lambda r: str(r.get("id", ""))
        )

        for i, room in enumerate(rooms[:MAX_ROOMS]):

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

            target[i] = torch.tensor(
                bounds,
                dtype=torch.float32
            )

            mask[i] = 1.0

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            target,
            mask,
        )


class GeometryModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(len(FEATURES), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.room_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, MAX_ROOMS * 5),
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
print("AI FLOOR PLANNER - GEOMETRY EVALUATION")
print("=" * 70)

print("\nDevice:", DEVICE)

dataset = GeometryDataset(TEST_PATH)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
)

model = GeometryModel().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

total_error = 0.0
total_coordinates = 0

valid_rooms = 0
valid_room_coordinate_error = 0.0

with torch.no_grad():

    for x, target, mask in loader:

        x = x.to(DEVICE)
        target = target.to(DEVICE)
        mask = mask.to(DEVICE)

        output = model(x)

        prediction = output[:, :, 1:]

        error = torch.abs(
            prediction - target
        )

        room_mask = mask.unsqueeze(-1)

        total_error += (
            error * room_mask
        ).sum().item()

        total_coordinates += (
            room_mask.sum().item() * 4
        )

        valid_rooms += mask.sum().item()

        valid_room_coordinate_error += (
            error * room_mask
        ).sum().item()


mae_normalized = (
    total_error /
    max(total_coordinates, 1)
)

# Convert normalized 0-1 coordinates
# back to the dataset's approximate 256-space.
mae_pixels = mae_normalized * 256.0

print("\n" + "=" * 70)
print("GEOMETRY EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nTest samples              : {len(dataset)}"
)

print(
    f"Valid room instances      : {int(valid_rooms)}"
)

print(
    f"Bounding-box coordinate MAE: "
    f"{mae_normalized:.6f}"
)

print(
    f"Approx coordinate error   : "
    f"{mae_pixels:.2f} dataset units"
)

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

shown = 0

with torch.no_grad():

    for x, target, mask in loader:

        x = x.to(DEVICE)

        output = model(x)

        prediction = output[:, :, 1:].cpu()

        target = target.cpu()
        mask = mask.cpu()

        for i in range(len(x)):

            print(
                f"\nSample {shown + 1}"
            )

            actual_count = int(
                mask[i].sum().item()
            )

            print(
                "Actual rooms:",
                actual_count
            )

            print(
                "\nFirst 3 actual boxes:"
            )

            for r in range(
                min(3, actual_count)
            ):

                print(
                    " ",
                    [
                        round(float(v), 3)
                        for v in target[i, r]
                    ]
                )

            print(
                "\nFirst 3 predicted boxes:"
            )

            for r in range(
                min(3, actual_count)
            ):

                print(
                    " ",
                    [
                        round(float(v), 3)
                        for v in prediction[i, r]
                    ]
                )

            shown += 1

            if shown >= 5:
                break

        if shown >= 5:
            break

print("\n" + "=" * 70)
print("GEOMETRY EVALUATION COMPLETE")
print("=" * 70)
