"""Training script for DSCNN encoder with Triplet Loss.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pt
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.models.dscnn import DSCNN
from src.models.prototypical import EpisodicBatchSampler, TripletLoss, train_one_epoch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    epoch: int,
    val_auc: float,
    loss: float,
    path: Path,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_auc": val_auc,
            "loss": loss,
        },
        path,
    )
    logger.info("Checkpoint saved: %s (epoch=%d, val_auc=%.4f)", path, epoch, val_auc)


def load_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
) -> int:
    """Load checkpoint. Returns the epoch to resume from."""
    checkpoint = torch.load(path, weights_only=False)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.info(
        "Resumed from %s (epoch=%d, loss=%.4f)",
        path,
        checkpoint["epoch"],
        checkpoint["loss"],
    )
    return checkpoint["epoch"] + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DSCNN encoder")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file path"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Model
    encoder = DSCNN(model_size=cfg["model"]["architecture"][-1]).to(device)
    param_count = sum(p.numel() for p in encoder.parameters())
    logger.info("DSCNN-%s: %d parameters", cfg["model"]["architecture"][-1], param_count)

    # Training setup
    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(encoder.parameters(), lr=train_cfg["optimizer"]["lr"])
    scheduler = StepLR(
        optimizer,
        step_size=train_cfg["scheduler"]["step_size"],
        gamma=train_cfg["scheduler"]["gamma"],
    )
    loss_fn = TripletLoss(margin=train_cfg["triplet_margin"])

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), encoder, optimizer, scheduler)

    # TODO: Replace with real MSWC dataset loading
    # For now, create a placeholder that shows the training loop structure
    logger.info("=" * 60)
    logger.info("Training loop ready. Requires MSWC dataset to run.")
    logger.info("Download MSWC first: python data/download_mswc.py")
    logger.info("=" * 60)
    logger.info(
        "Config: %d epochs x %d episodes, %d classes x %d samples",
        train_cfg["epochs"],
        train_cfg["episodes_per_epoch"],
        train_cfg["n_classes"],
        train_cfg["n_samples"],
    )
    logger.info(
        "Optimizer: Adam lr=%.4f, StepLR(step=%d, gamma=%.1f)",
        train_cfg["optimizer"]["lr"],
        train_cfg["scheduler"]["step_size"],
        train_cfg["scheduler"]["gamma"],
    )

    # Training loop skeleton (uncomment when data is ready)
    # best_val_auc = 0.0
    # for epoch in range(start_epoch, train_cfg["epochs"]):
    #     metrics = train_one_epoch(encoder, train_loader, optimizer, loss_fn, device)
    #     scheduler.step()
    #     logger.info("Epoch %d/%d - loss: %.4f", epoch+1, train_cfg["epochs"], metrics["loss"])
    #
    #     if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:
    #         # Run validation
    #         # val_auc = validate(encoder, val_loader, device)
    #         val_auc = 0.0  # placeholder
    #         save_checkpoint(encoder, optimizer, scheduler, epoch, val_auc, metrics["loss"],
    #                        Path(cfg["checkpoint"]["dir"]) / "latest.pt")
    #         if val_auc > best_val_auc:
    #             best_val_auc = val_auc
    #             save_checkpoint(encoder, optimizer, scheduler, epoch, val_auc, metrics["loss"],
    #                            Path(cfg["checkpoint"]["dir"]) / "best.pt")


if __name__ == "__main__":
    main()
