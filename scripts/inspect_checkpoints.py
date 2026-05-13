"""Quick inspection of available checkpoints to compare training quality."""

from pathlib import Path

import torch

CHECKPOINTS = [
    "checkpoints/triplet/best_v1_margin0.5_local.pt",
    "checkpoints/triplet/best_v2_margin1.0_colab.pt",
    "checkpoints/triplet/best_v3_margin1.0_phase2.pt",
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
