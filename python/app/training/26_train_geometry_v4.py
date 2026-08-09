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

MODEL_PATH = MODEL_DIR / "room_geometry_v4.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ROOM_TYPES = 8
MAX_ROOM_INDEX = 10
MAX_ROOMS = 23

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


# ============================================================
# DATASET
# ============================================================

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

        types = []
        indices = []
        targets = []

        for room in rooms[:MAX_ROOMS]:

            types.append(
                int(room["type_id"])
            )

            indices.append(
                min(
                    int(room["room_index"]),
                    MAX_ROOM_INDEX - 1,
                )
            )

            x1, y1, x2, y2 = [
                float(v)
                for v in room["bounds"]
            ]

            # Convert:
            # x1,y1,x2,y2
            #
            # to:
            # cx,cy,width,height

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            width = max(
                0.001,
                x2 - x1,
            )

            height = max(
                0.001,
                y2 - y1,
            )

            targets.append(
                [
                    cx,
                    cy,
                    width,
                    height,
                ]
            )

        return (
            global_features,
            torch.tensor(
                types,
                dtype=torch.long,
            ),
            torch.tensor(
                indices,
                dtype=torch.long,
            ),
            torch.tensor(
                targets,
                dtype=torch.float32,
            ),
        )


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    global_features = []
    room_types = []
    room_indices = []
    targets = []
    masks = []

    max_rooms = max(
        len(item[1])
        for item in batch
    )

    for global_x, types, indices, boxes in batch:

        count = len(types)

        pad_count = max_rooms - count

        global_features.append(global_x)

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

        masks.append(
            torch.cat(
                [
                    torch.ones(
                        count,
                        dtype=torch.float32,
                    ),
                    torch.zeros(
                        pad_count,
                        dtype=torch.float32,
                    ),
                ]
            )
        )

    return (
        torch.stack(global_features),
        torch.stack(room_types),
        torch.stack(room_indices),
        torch.stack(targets),
        torch.stack(masks),
    )


# ============================================================
# MODEL
# ============================================================

class RoomGeometryV4(nn.Module):

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
            nn.Linear(
                128,
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


# ============================================================
# LOSSES
# ============================================================

def masked_box_loss(
    prediction,
    target,
    mask,
):

    error = torch.abs(
        prediction - target
    )

    mask = mask.unsqueeze(-1)

    return (
        error * mask
    ).sum() / (
        mask.sum() * 4 + 1e-8
    )


def boundary_loss(
    prediction,
    mask,
):

    cx = prediction[..., 0]
    cy = prediction[..., 1]
    width = prediction[..., 2]
    height = prediction[..., 3]

    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0

    penalty = (
        torch.relu(-x1)
        + torch.relu(-y1)
        + torch.relu(x2 - 1.0)
        + torch.relu(y2 - 1.0)
    )

    return (
        penalty * mask
    ).sum() / (
        mask.sum() + 1e-8
    )


def collision_loss(
    prediction,
    mask,
):

    cx = prediction[..., 0]
    cy = prediction[..., 1]
    width = prediction[..., 2]
    height = prediction[..., 3]

    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0

    left = torch.maximum(
        x1.unsqueeze(2),
        x1.unsqueeze(1),
    )

    right = torch.minimum(
        x2.unsqueeze(2),
        x2.unsqueeze(1),
    )

    top = torch.maximum(
        y1.unsqueeze(2),
        y1.unsqueeze(1),
    )

    bottom = torch.minimum(
        y2.unsqueeze(2),
        y2.unsqueeze(1),
    )

    overlap_width = torch.relu(
        right - left
    )

    overlap_height = torch.relu(
        bottom - top
    )

    overlap = (
        overlap_width
        * overlap_height
    )

    pair_mask = (
        mask.unsqueeze(2)
        * mask.unsqueeze(1)
    )

    eye = torch.eye(
        prediction.shape[1],
        device=prediction.device,
    )

    pair_mask = (
        pair_mask
        * (1.0 - eye.unsqueeze(0))
    )

    overlap = (
        overlap * pair_mask
    )

    return (
        overlap.sum()
        /
        (pair_mask.sum() + 1e-8)
    )


def total_loss(
    prediction,
    target,
    mask,
):

    box = masked_box_loss(
        prediction,
        target,
        mask,
    )

    boundary = boundary_loss(
        prediction,
        mask,
    )

    collision = collision_loss(
        prediction,
        mask,
    )

    loss = (
        box
        + 0.30 * boundary
        + 1.00 * collision
    )

    return (
        loss,
        box,
        boundary,
        collision,
    )


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("AI FLOOR PLANNER - GEOMETRY V4 TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)

train_dataset = GeometryDataset(
    TRAIN_PATH
)

val_dataset = GeometryDataset(
    VAL_PATH
)

print(
    "\nTrain samples:",
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

model = RoomGeometryV4().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4,
)

EPOCHS = 50

best_val_loss = float("inf")

print("\nStarting training...")
print("-" * 70)

for epoch in range(EPOCHS):

    model.train()

    train_total = 0.0
    train_box = 0.0
    train_boundary = 0.0
    train_collision = 0.0

    for (
        global_x,
        room_types,
        room_indices,
        targets,
        masks,
    ) in train_loader:

        global_x = global_x.to(DEVICE)
        room_types = room_types.to(DEVICE)
        room_indices = room_indices.to(DEVICE)
        targets = targets.to(DEVICE)
        masks = masks.to(DEVICE)

        padding_mask = masks == 0

        optimizer.zero_grad()

        prediction = model(
            global_x,
            room_types,
            room_indices,
            padding_mask,
        )

        (
            loss,
            box_loss,
            boundary,
            collision,
        ) = total_loss(
            prediction,
            targets,
            masks,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        train_total += loss.item()
        train_box += box_loss.item()
        train_boundary += boundary.item()
        train_collision += collision.item()

    train_total /= len(train_loader)
    train_box /= len(train_loader)
    train_boundary /= len(train_loader)
    train_collision /= len(train_loader)

    model.eval()

    val_total = 0.0
    val_box = 0.0
    val_boundary = 0.0
    val_collision = 0.0

    with torch.no_grad():

        for (
            global_x,
            room_types,
            room_indices,
            targets,
            masks,
        ) in val_loader:

            global_x = global_x.to(DEVICE)
            room_types = room_types.to(DEVICE)
            room_indices = room_indices.to(DEVICE)
            targets = targets.to(DEVICE)
            masks = masks.to(DEVICE)

            padding_mask = masks == 0

            prediction = model(
                global_x,
                room_types,
                room_indices,
                padding_mask,
            )

            (
                loss,
                box_loss,
                boundary,
                collision,
            ) = total_loss(
                prediction,
                targets,
                masks,
            )

            val_total += loss.item()
            val_box += box_loss.item()
            val_boundary += boundary.item()
            val_collision += collision.item()

    val_total /= len(val_loader)
    val_box /= len(val_loader)
    val_boundary /= len(val_loader)
    val_collision /= len(val_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS}"
        f" | Train {train_total:.5f}"
        f" | Val {val_total:.5f}"
        f" | Box {val_box:.5f}"
        f" | Boundary {val_boundary:.5f}"
        f" | Collision {val_collision:.5f}"
    )

    if val_total < best_val_loss:

        best_val_loss = val_total

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

                "max_rooms":
                    MAX_ROOMS,

                "best_val_loss":
                    best_val_loss,

                "epoch":
                    epoch + 1,
            },
            MODEL_PATH,
        )

        print(
            f"  -> Best V4 model saved "
            f"(val={val_total:.6f})"
        )


print("\n" + "=" * 70)
print("GEOMETRY V4 TRAINING COMPLETE")
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

