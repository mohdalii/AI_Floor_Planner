import torch
from pathlib import Path

MODEL_PATH = Path("python/app/models/room_geometry_v2.pt")

print("=" * 70)
print("GEOMETRY V2 CHECKPOINT DIAGNOSTICS")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(" -", key)

print("\nTraining information:")

for key in [
    "epoch",
    "epochs",
    "train_loss",
    "val_loss",
    "best_val_loss",
    "loss",
]:
    if key in checkpoint:
        print(f"{key:20s}: {checkpoint[key]}")

state = checkpoint.get("model_state_dict")

if state is not None:
    print("\nModel parameters:")
    total = 0

    for name, tensor in state.items():
        count = tensor.numel()
        total += count

    print("Total parameters :", total)

print("\nModel path:")
print(MODEL_PATH)

print("\nDone.")
