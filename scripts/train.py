"""Training script for DSCNN encoder with Triplet Loss or ArcFace.

Trains on MSWC English (450 train words, 50 val words) by default.
GSC v2 is used ONLY for evaluation, not for training.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --loss arcface
    python scripts/train.py --config configs/default.yaml --loss scaf
    python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pt
    python scripts/train.py --config configs/default.yaml --data-dir data/gsc_v2  # fallback for testing
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dscnn import DSCNN
from src.models.prototypical import EpisodicBatchSampler, TripletLoss, train_one_epoch
from src.models.arcface import ArcFaceLoss, SubCenterArcFaceLoss
from src.data.mswc_dataset import MSWCDataset, build_episodic_loader
from src.features.augmentation import NoiseAugmenter, WaveformAugmenter
from src.features.spec_augment import SpecAugment

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_auc: float,
    loss: float,
    path: Path,
    loss_head: torch.nn.Module | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_auc": val_auc,
        "loss": loss,
    }
    if loss_head is not None:
        ckpt["loss_head_state_dict"] = loss_head.state_dict()
    torch.save(ckpt, path)
    logger.info("Checkpoint saved: %s (epoch=%d, loss=%.6f)", path, epoch, loss)


def load_checkpoint(path: Path, encoder, optimizer, scheduler, loss_head=None) -> int:
    checkpoint = torch.load(path, weights_only=False)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if loss_head is not None and "loss_head_state_dict" in checkpoint:
        loss_head.load_state_dict(checkpoint["loss_head_state_dict"])
    logger.info("Resumed from %s (epoch=%d)", path, checkpoint["epoch"])
    return checkpoint["epoch"] + 1


def train_one_epoch_arcface(
    encoder: torch.nn.Module,
    loss_head: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
) -> dict:
    """Train one epoch with ArcFace/SCAF loss."""
    encoder.train()
    loss_head.train()
    total_loss = 0.0
    n_batches = 0

    for batch_mfcc, batch_labels in dataloader:
        batch_mfcc = batch_mfcc.to(device)
        batch_labels = batch_labels.to(device)

        embeddings = encoder(batch_mfcc)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        loss = loss_head(embeddings, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(loss_head.parameters()),
                max_norm=grad_clip,
            )
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "num_episodes": n_batches}


def validate_few_shot(
    encoder: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
) -> dict:
    """Run few-shot validation episodes and compute AUC.

    Args:
        encoder: DSCNN encoder in eval mode.
        val_loader: DataLoader with episodic batches from validation words.
        device: Computation device.
        n_way: Number of classes per episode.
        k_shot: Support samples per class.
        n_query: Query samples per class.

    Returns:
        Dict with 'val_auc' and 'val_acc'.
    """
    encoder.eval()
    all_correct = 0
    all_total = 0
    all_y_true = []
    all_scores = []

    with torch.no_grad():
        for batch_mfcc, batch_labels in val_loader:
            batch_mfcc = batch_mfcc.to(device)
            batch_labels = batch_labels.to(device)

            embeddings = encoder(batch_mfcc)
            embeddings = F.normalize(embeddings, p=2, dim=-1)

            unique_classes = batch_labels.unique()
            actual_n_way = len(unique_classes)
            samples_per_class = (batch_labels == unique_classes[0]).sum().item()
            actual_k = min(k_shot, samples_per_class - 1)
            actual_q = samples_per_class - actual_k

            if actual_k < 1 or actual_q < 1:
                continue

            prototypes = []
            query_embs = []
            query_labels = []

            for i, cls in enumerate(unique_classes):
                mask = batch_labels == cls
                cls_embs = embeddings[mask]
                prototypes.append(cls_embs[:actual_k].mean(dim=0))
                query_embs.append(cls_embs[actual_k:])
                query_labels.extend([i] * (cls_embs.shape[0] - actual_k))

            proto_tensor = torch.stack(prototypes)  # (n_way, D)
            query_tensor = torch.cat(query_embs, dim=0)  # (n_query*n_way, D)
            query_labels_t = torch.tensor(query_labels, device=device)

            dists = torch.cdist(query_tensor, proto_tensor, p=2)  # (Q, n_way)
            preds = dists.argmin(dim=1)
            all_correct += (preds == query_labels_t).sum().item()
            all_total += len(query_labels_t)

            neg_min_dists = -dists.min(dim=1).values
            for i in range(len(query_labels)):
                for c in range(actual_n_way):
                    is_pos = 1 if c == query_labels[i] else 0
                    all_y_true.append(is_pos)
                    all_scores.append(-dists[i, c].item())

    val_acc = all_correct / max(all_total, 1)

    val_auc = 0.0
    if len(set(all_y_true)) >= 2:
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_y_true, all_scores)

    encoder.train()
    return {"val_auc": float(val_auc), "val_acc": float(val_acc)}


def _load_word_splits(data_dir: Path, cfg: dict) -> tuple[list[str] | None, list[str]]:
    """Load pre-computed train/val word splits from MSWC download.

    Args:
        data_dir: Data directory (may contain splits/ subdirectory).
        cfg: Config dict.

    Returns:
        (train_words, val_words) or (None, []) if splits not found.
    """
    import json

    splits_dir = data_dir / "splits"
    train_path = splits_dir / "train_words.json"
    val_path = splits_dir / "val_words.json"

    if train_path.exists() and val_path.exists():
        with open(train_path) as f:
            train_words = json.load(f)
        with open(val_path) as f:
            val_words = json.load(f)
        logger.info(
            "Loaded word splits: %d train, %d val from %s",
            len(train_words), len(val_words), splits_dir,
        )
        return train_words, val_words

    return None, []


def discover_words(data_dir: Path) -> list[str]:
    """Find all word directories with WAV files."""
    words = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and not d.name.startswith(("_", ".")):
            if any(d.glob("*.wav")):
                words.append(d.name)
    clips_dir = data_dir / "clips"
    if clips_dir.exists():
        for d in sorted(clips_dir.iterdir()):
            if d.is_dir() and any(d.glob("*.wav")):
                words.append(d.name)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DSCNN encoder")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: from config gsc_dir)")
    parser.add_argument("--loss", type=str, default="triplet",
                        choices=["triplet", "arcface", "scaf"],
                        help="Loss function")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--mining", type=str, default=None,
                        choices=["random", "hard", "semi_hard"],
                        help="Triplet mining strategy (override config)")
    parser.add_argument("--margin", type=float, default=None,
                        help="Triplet margin (override config)")
    parser.add_argument("--no-spec-augment", action="store_true",
                        help="Disable SpecAugment")
    parser.add_argument("--max-per-word", type=int, default=None,
                        help="Max samples per word (override config)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (lower for low-RAM systems)")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Run validation every N epochs (default 1)")
    parser.add_argument("--ckpt-subdir", type=str, default=None,
                        help="Subdirectory under checkpoint.dir for this run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Model
    encoder = DSCNN(model_size=cfg["model"]["architecture"][-1]).to(device)
    param_count = sum(p.numel() for p in encoder.parameters())
    logger.info("DSCNN-%s: %d parameters", cfg["model"]["architecture"][-1], param_count)

    # Data: default to MSWC for training, GSC only for evaluation
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        mswc_dir = Path(cfg["data"]["mswc_dir"])
        gsc_dir = Path(cfg["data"]["gsc_dir"])
        if mswc_dir.exists() and (
            any((mswc_dir / "clips").glob("*/*.wav")) if (mswc_dir / "clips").exists()
            else any(mswc_dir.glob("*/*.wav"))
        ):
            data_dir = mswc_dir
        elif gsc_dir.exists():
            logger.warning(
                "MSWC data not found at %s, falling back to GSC at %s. "
                "For proper training, download MSWC first: python data/download_mswc.py",
                mswc_dir, gsc_dir,
            )
            data_dir = gsc_dir
        else:
            raise FileNotFoundError(
                f"No training data found. Expected MSWC at {mswc_dir} "
                f"or GSC at {gsc_dir}. Run: python data/download_mswc.py"
            )

    # Load train/val word splits if available (from MSWC download)
    train_words, val_word_list = _load_word_splits(data_dir, cfg)
    if train_words is None:
        all_words = discover_words(data_dir)
        logger.info("Discovered %d words in %s", len(all_words), data_dir)
        val_count = cfg["data"].get("mswc_val_words", 50)
        if len(all_words) > val_count + 30:
            train_words = all_words[:-val_count]
            val_word_list = all_words[-val_count:]
        else:
            train_words = all_words
            val_word_list = []
    logger.info("Train words: %d, Val words: %d", len(train_words), len(val_word_list))

    noise_aug = None
    noise_dir = Path(cfg.get("noise", {}).get("demand_dir", "data/demand"))
    if noise_dir.exists():
        noise_aug = NoiseAugmenter(
            noise_dir, prob=cfg["noise"]["prob"],
            snr_db=(0.0, 10.0),
        )
        logger.info("Noise augmentation: %d files, SNR=(0,10)dB", len(noise_aug.noise_files))

    wave_aug = WaveformAugmenter()

    spec_cfg = cfg.get("augmentation", {}).get("spec_augment", {}) or {}
    spec_aug = None
    if spec_cfg.get("enabled", True) and not args.no_spec_augment:
        spec_aug = SpecAugment(
            freq_mask_width=int(spec_cfg.get("freq_mask", 5)),
            time_mask_width=int(spec_cfg.get("time_mask", 12)),
            n_freq_masks=int(spec_cfg.get("n_freq_masks", 2)),
            n_time_masks=int(spec_cfg.get("n_time_masks", 2)),
        )
        logger.info(
            "SpecAugment: freq=%d (x%d), time=%d (x%d)",
            spec_aug.freq_mask_width, spec_aug.n_freq_masks,
            spec_aug.time_mask_width, spec_aug.n_time_masks,
        )

    max_per_word = args.max_per_word or cfg["data"].get("max_per_word", 200)
    dataset = MSWCDataset(
        root_dir=data_dir, words=train_words, max_per_word=max_per_word,
        noise_augmenter=noise_aug, wave_augmenter=wave_aug,
        spec_augmenter=spec_aug,
    )

    train_cfg = cfg["training"]
    n_classes = min(train_cfg["n_classes"], len(dataset.word_to_idx))
    n_samples = train_cfg["n_samples"]
    n_episodes = args.episodes or train_cfg["episodes_per_epoch"]
    n_epochs = args.epochs or train_cfg["epochs"]

    train_loader = build_episodic_loader(
        dataset, n_classes=n_classes, n_samples=n_samples,
        n_episodes=n_episodes, num_workers=args.num_workers,
    )
    logger.info("DataLoader: %d classes x %d samples x %d episodes", n_classes, n_samples, n_episodes)

    # Loss + Optimizer
    opt_cfg = train_cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 0.001))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    mining = args.mining or train_cfg.get("mining", "semi_hard")
    margin = args.margin if args.margin is not None else train_cfg.get("triplet_margin", 1.0)

    loss_fn = None
    loss_head = None
    if args.loss == "triplet":
        loss_fn = TripletLoss(margin=margin, mining=mining)
        optimizer = torch.optim.Adam(
            encoder.parameters(), lr=lr, weight_decay=weight_decay,
        )
        logger.info("TripletLoss: margin=%.2f, mining=%s", margin, mining)
    elif args.loss == "arcface":
        loss_head = ArcFaceLoss(
            embedding_dim=encoder.embedding_dim,
            num_classes=len(dataset.word_to_idx),
            scale=30.0, margin=0.5,
        ).to(device)
        all_params = list(encoder.parameters()) + list(loss_head.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)
    elif args.loss == "scaf":
        loss_head = SubCenterArcFaceLoss(
            embedding_dim=encoder.embedding_dim,
            num_classes=len(dataset.word_to_idx),
            K=3, scale=30.0, margin=0.5,
        ).to(device)
        all_params = list(encoder.parameters()) + list(loss_head.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    sched_cfg = train_cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "StepLR")
    if sched_type == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(sched_cfg.get("T_0", 10)),
            T_mult=int(sched_cfg.get("T_mult", 2)),
            eta_min=float(sched_cfg.get("eta_min", 1e-5)),
        )
        logger.info(
            "Scheduler: CosineAnnealingWarmRestarts T_0=%d, T_mult=%d, eta_min=%g",
            int(sched_cfg.get("T_0", 10)), int(sched_cfg.get("T_mult", 2)),
            float(sched_cfg.get("eta_min", 1e-5)),
        )
    elif sched_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("T_max", n_epochs)),
            eta_min=float(sched_cfg.get("eta_min", 1e-5)),
        )
    else:
        scheduler = StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 20)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )
        logger.info(
            "Scheduler: StepLR step_size=%d, gamma=%g",
            int(sched_cfg.get("step_size", 20)),
            float(sched_cfg.get("gamma", 0.5)),
        )

    if weight_decay > 0:
        logger.info("Adam weight_decay=%g", weight_decay)
    if grad_clip > 0:
        logger.info("Gradient clipping: max_norm=%.2f", grad_clip)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), encoder, optimizer, scheduler, loss_head)

    # Validation set (separate val words, never seen during training)
    val_loader = None
    if val_word_list:
        logger.info("Building validation set from %d held-out words", len(val_word_list))
        val_dataset = MSWCDataset(
            root_dir=data_dir, words=val_word_list, max_per_word=50,
        )
        if len(val_dataset) > 0:
            val_n_classes = min(5, len(val_dataset.word_to_idx))
            val_n_samples = min(20, min(
                len([s for s in val_dataset.samples if s[1] == i])
                for i in range(val_n_classes)
            ) if val_n_classes > 0 else 20)
            try:
                val_loader = build_episodic_loader(
                    val_dataset, n_classes=val_n_classes, n_samples=val_n_samples,
                    n_episodes=20, num_workers=0,
                )
                logger.info("Validation loader: %d classes x %d samples x 20 episodes", val_n_classes, val_n_samples)
            except ValueError:
                logger.warning("Could not build validation loader, skipping validation")
                val_loader = None
    else:
        logger.info("No validation words available, using training loss only")

    # TensorBoard / checkpoint dir
    ckpt_subdir = args.ckpt_subdir or args.loss
    ckpt_dir = Path(cfg["checkpoint"]["dir"]) / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "runs"))
    logger.info("TensorBoard: %s", writer.log_dir)
    logger.info("Checkpoints: %s", ckpt_dir)

    # Training loop
    save_every = cfg["checkpoint"]["save_every"]
    best_metric = -float("inf")
    val_every = max(1, args.val_every)

    logger.info("=" * 60)
    logger.info("Training: %s loss, %d epochs, %d episodes/epoch", args.loss, n_epochs, n_episodes)
    logger.info("=" * 60)

    val_auc = 0.0
    metrics: dict = {"loss": float("nan")}

    for epoch in range(start_epoch, n_epochs):
        if args.loss == "triplet":
            metrics = train_one_epoch(
                encoder, train_loader, optimizer, loss_fn, device,
                grad_clip=grad_clip,
            )
        else:
            metrics = train_one_epoch_arcface(
                encoder, loss_head, train_loader, optimizer, device,
                grad_clip=grad_clip,
            )

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        log_line = (
            f"Epoch {epoch+1}/{n_epochs} | loss={metrics['loss']:.6f} | "
            f"lr={cur_lr:.6f} | episodes={metrics['num_episodes']}"
        )
        if args.loss == "triplet":
            log_line += (
                f" | active={metrics.get('active_per_ep', 0):.1f}"
                f" | semi_hard={metrics.get('semi_hard_per_ep', 0):.1f}"
                f" | hard_fb={metrics.get('hard_fallback_per_ep', 0):.1f}"
                f" | d_pos={metrics.get('mean_d_pos', 0):.3f}"
                f" | d_neg={metrics.get('mean_d_neg', 0):.3f}"
            )
            if grad_clip > 0:
                log_line += f" | grad_norm={metrics.get('mean_grad_norm', 0):.2f}"
        logger.info(log_line)

        writer.add_scalar("train/loss", metrics["loss"], epoch + 1)
        writer.add_scalar("train/lr", cur_lr, epoch + 1)
        if args.loss == "triplet":
            writer.add_scalar("train/active_triplets", metrics.get("active_per_ep", 0), epoch + 1)
            writer.add_scalar("train/semi_hard_count", metrics.get("semi_hard_per_ep", 0), epoch + 1)
            writer.add_scalar("train/hard_fallback_count", metrics.get("hard_fallback_per_ep", 0), epoch + 1)
            writer.add_scalar("train/mean_d_pos", metrics.get("mean_d_pos", 0), epoch + 1)
            writer.add_scalar("train/mean_d_neg", metrics.get("mean_d_neg", 0), epoch + 1)
            if grad_clip > 0:
                writer.add_scalar("train/grad_norm", metrics.get("mean_grad_norm", 0), epoch + 1)

        # Validation -- now every epoch by default for finer-grained best tracking
        val_auc = 0.0
        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_metrics = validate_few_shot(encoder, val_loader, device)
            val_auc = val_metrics["val_auc"]
            writer.add_scalar("val/auc", val_auc, epoch + 1)
            writer.add_scalar("val/acc", val_metrics["val_acc"], epoch + 1)
            logger.info(
                "  Val: AUC=%.4f, ACC=%.4f", val_auc, val_metrics["val_acc"],
            )

        current_metric = val_auc if val_auc > 0 else -metrics["loss"]

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                encoder, optimizer, scheduler, epoch, val_auc, metrics["loss"],
                ckpt_dir / f"epoch_{epoch+1:02d}.pt", loss_head,
            )

        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                encoder, optimizer, scheduler, epoch, val_auc, metrics["loss"],
                ckpt_dir / "best.pt", loss_head,
            )
            if val_auc > 0:
                logger.info("  * New best val AUC: %.4f", val_auc)
            else:
                logger.info("  * New best loss: %.6f", metrics["loss"])

    save_checkpoint(
        encoder, optimizer, scheduler, n_epochs - 1, val_auc, metrics["loss"],
        ckpt_dir / "latest.pt", loss_head,
    )
    writer.close()
    logger.info("Done! Best metric: %.4f", best_metric)


if __name__ == "__main__":
    main()
