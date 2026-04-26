"""Quick inspection of available checkpoints to compare training quality."""

from pathlib import Path

import torch

CHECKPOINTS = [
    "checkpoints/best.pt",
    "checkpoints/triplet/best.pt",
    "checkpoints/triplet/epoch_05.pt",
    "checkpoints/triplet/epoch_10.pt",
    "checkpoints/triplet/epoch_15.pt",
]


def main() -> None:
    for path_str in CHECKPOINTS:
        path = Path(path_str)
        if not path.exists():
            print(f"{path}: NOT FOUND")
            continue
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            print(
                f"{path} -> "
                f"epoch={ckpt.get('epoch', '?')}, "
                f"loss={ckpt.get('loss', '?')}, "
                f"val_auc={ckpt.get('val_auc', ckpt.get('best_val_auc', '?'))}, "
                f"keys={list(ckpt.keys())}"
            )
        else:
            print(f"{path}: tensor-only checkpoint")


if __name__ == "__main__":
    main()
