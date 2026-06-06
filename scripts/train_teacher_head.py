"""Train the Wav2Vec2 teacher projection head with Sub-center ArcFace.

The EdgeSpot KD recipe distills a compact student from a frozen Wav2Vec2
teacher whose 64-D projection head is trained with Sub-center ArcFace on word
labels. Without a trained head, ``Wav2Vec2Teacher`` uses a random projection,
which is only valid for smoke tests. This script produces a real head
checkpoint compatible with ``Wav2Vec2Teacher(head_checkpoint=...)`` and with
``scripts/precompute_teacher_embeddings.py --head-checkpoint ...``.

Efficiency: the Wav2Vec2 encoder is frozen, so pooled features are extracted in
a single pass and cached in CPU memory; the projection head + Sub-center
ArcFace are then trained for many epochs cheaply over the cache.

Example:
    python scripts/train_teacher_head.py \
      --data-dir data/mswc_en \
      --train-files train_files_cap220_flac.json \
      --max-per-word 50 \
      --epochs 30 \
      --output outputs/teacher_head/teacher_head.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_waveform
from src.models.arcface import SubCenterArcFaceLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
AUDIO_EXTENSIONS = (".opus", ".wav", ".ogg", ".flac")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_words(data_dir: Path, split: str) -> list[str] | None:
    path = data_dir / "splits" / f"{split}_words.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_word_files(word_dir: Path, limit: int) -> list[Path]:
    files: list[Path] = []
    for ext in AUDIO_EXTENSIONS:
        for path in word_dir.glob(f"*{ext}"):
            files.append(path)
            if limit > 0 and len(files) >= limit:
                return sorted(files)
    return sorted(set(files))


def build_samples(
    data_dir: Path,
    words: list[str],
    train_files: str | None,
    max_per_word: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """Return (path, label) pairs and the word->index map."""
    selected_words = sorted(words)
    word_to_idx = {w: i for i, w in enumerate(selected_words)}
    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []

    if train_files:
        manifest_path = Path(train_files)
        if not manifest_path.is_absolute():
            manifest_path = data_dir / "splits" / train_files
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[Path]] = {w: [] for w in selected_words}
        for item in items:
            p = Path(item)
            if not p.is_absolute():
                p = data_dir / p
            word = p.parent.name
            if word in grouped:
                grouped[word].append(p)
        for word in selected_words:
            files = sorted(set(grouped.get(word, [])))
            if max_per_word > 0 and len(files) > max_per_word:
                files = rng.sample(files, max_per_word)
            for f in files:
                samples.append((f, word_to_idx[word]))
    else:
        for word in selected_words:
            word_dir = data_dir / word
            if not word_dir.exists():
                word_dir = data_dir / "clips" / word
            if not word_dir.exists():
                continue
            files = _scan_word_files(word_dir, limit=max_per_word)
            if max_per_word > 0 and len(files) > max_per_word:
                files = rng.sample(files, max_per_word)
            for f in files:
                samples.append((f, word_to_idx[word]))

    return samples, word_to_idx


class WaveformDataset(Dataset):
    """Yields ``(waveform[T], label)`` for frozen-encoder feature extraction."""

    def __init__(self, samples: list[tuple[Path, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            wav = load_waveform(path, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH)
        except Exception as exc:  # noqa: BLE001 - degrade to silence on a bad file
            logger.warning("Failed to load %s: %s", path, exc)
            wav = torch.zeros(1, TARGET_LENGTH)
        return wav.squeeze(0), label


@torch.no_grad()
def extract_pooled_features(
    samples: list[tuple[Path, int]],
    model_name: str,
    layer: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Single frozen pass over Wav2Vec2; returns (features[N,H] fp16, labels[N], H)."""
    try:
        from transformers import Wav2Vec2Model
    except ImportError as exc:
        raise ImportError("Install transformers: pip install transformers") from exc

    model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    loader = DataLoader(
        WaveformDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    hidden_size = int(model.config.hidden_size)
    for waves, label_batch in tqdm(loader, desc="extract w2v2"):
        waves = waves.to(device, non_blocking=True)
        out = model(waves, output_hidden_states=True)
        hidden_states = out.hidden_states
        idx = min(max(int(layer), 0), len(hidden_states) - 1)
        pooled = hidden_states[idx].mean(dim=1)
        feats.append(pooled.detach().to(torch.float16).cpu())
        labels.append(label_batch)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0), hidden_size


def train_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int,
    num_classes: int,
    embedding_dim: int,
    sub_centers: int,
    scale: float,
    margin: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Linear, dict]:
    """Train projection head + Sub-center ArcFace over cached pooled features."""
    proj = nn.Linear(hidden_size, embedding_dim).to(device)
    scaf = SubCenterArcFaceLoss(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        K=sub_centers,
        scale=scale,
        margin=margin,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(proj.parameters()) + list(scaf.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    n = features.shape[0]
    best_state = {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()}
    best_acc = -1.0

    for epoch in range(epochs):
        perm = torch.randperm(n)
        proj.train()
        scaf.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, n, batch_size):
            sel = perm[start:start + batch_size]
            feat = features[sel].to(device, non_blocking=True).float()
            lab = labels[sel].to(device, non_blocking=True).long()

            emb = F.normalize(proj(feat), p=2, dim=-1)
            loss = scaf(emb, lab)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * sel.numel()
            with torch.no_grad():
                normed_w = F.normalize(scaf.weight, p=2, dim=1)
                cos = F.linear(emb, normed_w).view(-1, num_classes, sub_centers).max(dim=2)[0]
                correct += int((cos.argmax(dim=1) == lab).sum().item())
            seen += sel.numel()

        acc = correct / max(seen, 1)
        logger.info(
            "Epoch %d/%d | loss=%.4f | train_top1=%.4f",
            epoch + 1, epochs, total_loss / max(seen, 1), acc,
        )
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()}

    meta = {"best_train_top1": best_acc, "num_classes": num_classes}
    proj.load_state_dict(best_state)
    return proj, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 teacher head with Sub-center ArcFace")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mswc_en"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "eval"])
    parser.add_argument("--train-files", type=str, default=None,
                        help="Manifest json (e.g. train_files_cap220_flac.json) under <data-dir>/splits.")
    parser.add_argument("--max-per-word", type=int, default=50,
                        help="Cap samples per word for head training (speed). 0 = all.")
    parser.add_argument("--max-words", type=int, default=0,
                        help="Optional cap on number of words (0 = all train words).")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--sub-centers", type=int, default=3)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for head training over cached features.")
    parser.add_argument("--extract-batch-size", type=int, default=32,
                        help="Batch size for the Wav2Vec2 extraction pass.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/teacher_head/teacher_head.pt"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    words = load_words(args.data_dir, args.split)
    if not words:
        raise FileNotFoundError(
            f"No {args.split}_words.json under {args.data_dir}/splits. Prepare MSWC splits first."
        )
    if args.max_words > 0:
        words = sorted(words)[:args.max_words]

    samples, word_to_idx = build_samples(
        args.data_dir, words, args.train_files, args.max_per_word, args.seed
    )
    if not samples:
        raise FileNotFoundError("No audio files resolved for teacher head training.")
    num_classes = len(word_to_idx)
    logger.info("Teacher head training: %d files, %d word classes", len(samples), num_classes)

    features, labels, hidden_size = extract_pooled_features(
        samples, args.model_name, args.layer,
        args.extract_batch_size, args.num_workers, device,
    )
    logger.info("Cached pooled features: %s (hidden=%d)", tuple(features.shape), hidden_size)

    proj, meta = train_head(
        features, labels, hidden_size, num_classes,
        args.embedding_dim, args.sub_centers, args.scale, args.margin,
        args.epochs, args.batch_size, args.lr, args.weight_decay, device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "projection_state_dict": proj.state_dict(),
        "model_name": args.model_name,
        "layer": args.layer,
        "embedding_dim": args.embedding_dim,
        "hidden_size": hidden_size,
        "sub_centers": args.sub_centers,
        "num_classes": num_classes,
        "word_to_idx": word_to_idx,
        "best_train_top1": meta["best_train_top1"],
    }
    torch.save(payload, args.output)
    logger.info(
        "Saved teacher head: %s (best_train_top1=%.4f). Use with --head-checkpoint for precompute.",
        args.output, meta["best_train_top1"],
    )


if __name__ == "__main__":
    main()
