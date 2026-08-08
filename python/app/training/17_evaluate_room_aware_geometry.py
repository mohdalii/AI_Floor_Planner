import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")

TEST_PATH = ROOT / "dataset" / "geometry_training" / "test.json"
MODEL_PATH = ROOT / "python" / "app" / "models" / "room_aware_geometry.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_ROOMS = 23
NUM_ROOM_TYPES = 8

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

        for i, room in enumerate(rooms[:MAX_ROOMS]):

            room_types[i] = int(
                room["type_id"]
            )

            targets[i] = torch.tensor(
                room["bounds"],
                dtype=torch.float32,
            )

            mask[i] = 1.0

        return (
            torch.tensor(x, dtype=torch.float32),
            room_types,
            targets,
            mask,
        )


class RoomAwareGeometryModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_encoder = nn.Sequential(
            nn.Linear(len(FEATURES), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.room_embedding = nn.Embedding(
            NUM_ROOM_TYPES,
            32,
        )

        self.decoder = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x, room_types):

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

        return self.decoder(combined)


print("=" * 70)
print("AI FLOOR PLANNER - ROOM-AWARE GEOMETRY TEST")
print("=" * 70)

print("\nDevice:", DEVICE)

dataset = RoomGeometryDataset(
    TEST_PATH
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
)

model = RoomAwareGeometryModel().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

total_abs_error = 0.0
total_values = 0

room_error = {
    room: 0.0
    for room in ROOM_TYPES
}

room_count = {
    room: 0
    for room in ROOM_TYPES
}

plans_with_all_rooms_correct = 0
total_plans = len(dataset)

sample_outputs = []

with torch.no_grad():

    for x, room_types, targets, mask in loader:

        x = x.to(DEVICE)
        room_types = room_types.to(DEVICE)
        targets = targets.to(DEVICE)
        mask = mask.to(DEVICE)

        prediction = model(
            x,
            room_types,
        )

        error = torch.abs(
            prediction - targets
        )

        valid_mask = mask.unsqueeze(-1)

        total_abs_error += (
            error * valid_mask
        ).sum().item()

        total_values += (
            valid_mask.sum().item() * 4
        )

        # Room-wise errors
        for room_id, room_name in enumerate(
            ROOM_TYPES
        ):

            room_mask = (
                (room_types == room_id)
                & (mask > 0)
            )

            if room_mask.any():

                selected_error = error[
                    room_mask
                ]

                room_error[room_name] += (
                    selected_error.sum().item()
                )

                room_count[room_name] += (
                    room_mask.sum().item()
                )

        # Exact plan check using
        # rounded 2-decimal coordinates.
        rounded_prediction = torch.round(
            prediction * 100
        ) / 100

        rounded_target = torch.round(
            targets * 100
        ) / 100

        coordinate_match = (
            torch.abs(
                rounded_prediction
                - rounded_target
            ) < 0.05
        ).all(dim=2)

        plan_match = (
            coordinate_match
            | (mask == 0)
        ).all(dim=1)

        plans_with_all_rooms_correct += (
            plan_match.sum().item()
        )

        # Save first few examples
        if len(sample_outputs) < 5:

            for i in range(len(x)):

                if len(sample_outputs) >= 5:
                    break

                valid_count = int(
                    mask[i].sum().item()
                )

                rooms = []

                for r in range(valid_count):

                    room_id = int(
                        room_types[i, r].item()
                    )

                    room_name = ROOM_TYPES[
                        room_id
                    ]

                    rooms.append({
                        "type": room_name,

                        "actual": [
                            round(
                                float(v),
                                3
                            )
                            for v in targets[
                                i, r
                            ].cpu()
                        ],

                        "predicted": [
                            round(
                                float(v),
                                3
                            )
                            for v in prediction[
                                i, r
                            ].cpu()
                        ],
                    })

                sample_outputs.append({
                    "rooms": rooms
                })


mae = (
    total_abs_error
    / max(total_values, 1)
)

pixel_mae = mae * 256.0

plan_accuracy = (
    plans_with_all_rooms_correct
    / max(total_plans, 1)
)


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"\nTest plans              : "
    f"{total_plans}"
)

print(
    f"Coordinate MAE          : "
    f"{mae:.6f}"
)

print(
    f"Approx 256-space error  : "
    f"{pixel_mae:.2f}"
)

print(
    f"Near-exact plan accuracy: "
    f"{plan_accuracy:.2%}"
)


print("\n" + "=" * 70)
print("ROOM-WISE ERROR")
print("=" * 70)

for room in ROOM_TYPES:

    count = room_count[room]

    if count == 0:
        print(
            f"{room:15s}: no samples"
        )
        continue

    room_mae = (
        room_error[room]
        / (count * 4)
    )

    print(
        f"{room:15s}: "
        f"MAE={room_mae:.6f} "
        f"| samples={count}"
    )


print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for index, sample in enumerate(
    sample_outputs,
    start=1
):

    print(
        f"\nSample {index}"
    )

    for room in sample["rooms"]:

        print(
            f"\n{room['type']}"
        )

        print(
            "  Actual   :",
            room["actual"]
        )

        print(
            "  Predicted:",
            room["predicted"]
        )


print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)
