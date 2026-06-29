"""Precompute Wav2Vec2 teacher embeddings for KD training.

Supports WAV/FLAC/OPUS/OGG via ``src.audio_io`` (no torchaudio dependency, so it
also works on the FLAC capped MSWC subsets). Prefer ``--train-files`` so the
saved path keys match exactly the manifest the student uses; otherwise the
script scans the split word folders.

Example:
    python scripts/precompute_teacher_embeddings.py \
      --data-dir data/mswc_en --train-files train_files_cap220_flac.json \
      --head-checkpoint outputs/teacher_head/teacher_head.pt \
      --output-dir outputs/teacher_w2v2_train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io import load_waveform
from src.training.teacher import Wav2Vec2Teacher

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
AUDIO_EXTENSIONS = (".opus", ".wav", ".ogg", ".flac")


def load_words(data_dir: Path, split: str) -> list[str] | None:
    path = data_dir / "splits" / f"{split}_words.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_from_manifest(
    data_dir: Path,
    train_files: str,
    max_per_word: int,
    seed: int,
    allowed_words: set[str] | None = None,
) -> list[Path]:
    """Resolve manifest items the same way the student MSWCDataset does."""
    import random

    manifest_path = Path(train_files)
    if not manifest_path.is_absolute():
        manifest_path = data_dir / "splits" / train_files
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[Path]] = {}
    for item in items:
        p = Path(item)
        if not p.is_absolute():
            p = data_dir / p
        if allowed_words is not None and p.parent.name not in allowed_words:
            continue
        grouped.setdefault(p.parent.name, []).append(p)
    rng = random.Random(seed)
    paths: list[Path] = []
    for word in sorted(grouped):
        word_paths = sorted(set(grouped[word]))
        if max_per_word > 0 and len(word_paths) > max_per_word:
            word_paths = rng.sample(word_paths, max_per_word)
        paths.extend(word_paths)
    return paths


def discover_by_scan(data_dir: Path, words: list[str] | None, max_per_word: int) -> list[Path]:
    roots = [data_dir / "clips", data_dir]
    base = next((p for p in roots if p.exists()), data_dir)

    def has_audio(d: Path) -> bool:
        return any(any(d.glob(f"*{ext}")) for ext in AUDIO_EXTENSIONS)

    selected_words = words or sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and has_audio(d)
    )
    paths: list[Path] = []
    for word in selected_words:
        word_dir = base / word
        files: list[Path] = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(word_dir.glob(f"*{ext}"))
        files = sorted(set(files))
        if max_per_word > 0:
            files = files[:max_per_word]
        paths.extend(files)
    return paths


def load_word_list_override(data_dir: Path, path_or_name: str | None) -> list[str] | None:
    if not path_or_name:
        return None
    path = Path(path_or_name)
    if not path.is_absolute() and path.parent == Path("."):
        path = data_dir / "splits" / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return [str(item) for item in payload]


def load_wave(path: Path) -> torch.Tensor:
    """Return mono waveform shaped ``(1, TARGET_LENGTH)``."""
    return load_waveform(path, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Wav2Vec2 teacher embeddings")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mswc_en"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "eval", "all"])
    parser.add_argument("--train-files", type=str, default=None,
                        help="Manifest json (e.g. train_files_cap220_flac.json). "
                             "Preferred so path keys match the student manifest.")
    parser.add_argument("--train-words-file", type=str, default=None,
                        help="Optional word-list JSON. Basenames resolve under <data-dir>/splits.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--head-checkpoint", type=str, default=None,
                        help="Trained teacher head from scripts/train_teacher_head.py. "
                             "Without it the projection is random (smoke test only).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-per-word", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    word_override = load_word_list_override(args.data_dir, args.train_words_file)
    allowed_words = set(word_override) if word_override is not None else None
    if args.train_files:
        audio_paths = discover_from_manifest(
            args.data_dir,
            args.train_files,
            args.max_per_word,
            args.seed,
            allowed_words=allowed_words,
        )
    else:
        words = word_override if word_override is not None else (None if args.split == "all" else load_words(args.data_dir, args.split))
        audio_paths = discover_by_scan(args.data_dir, words, args.max_per_word)
    if not audio_paths:
        raise FileNotFoundError(f"No audio files found under {args.data_dir}")

    if not args.head_checkpoint:
        print("WARNING: no --head-checkpoint; projection head is random (smoke test only).")

    done_paths = set()
    for shard in args.output_dir.glob("teacher_*.pt"):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        done_paths.update(payload.get("paths", []))

    pending = [p for p in audio_paths if p.as_posix() not in done_paths]
    teacher = Wav2Vec2Teacher(
        model_name=args.model_name,
        layer=args.layer,
        embedding_dim=args.embedding_dim,
        head_checkpoint=args.head_checkpoint,
    ).to(device).eval()

    shard_paths: list[str] = []
    shard_embeddings: list[torch.Tensor] = []
    shard_idx = len(list(args.output_dir.glob("teacher_*.pt")))

    for i in tqdm(range(0, len(pending), args.batch_size), desc="teacher"):
        batch_paths = pending[i:i + args.batch_size]
        waves = torch.cat([load_wave(p) for p in batch_paths], dim=0).to(device)
        with torch.no_grad():
            embs = teacher(waves).cpu()
        shard_paths.extend([p.as_posix() for p in batch_paths])
        shard_embeddings.append(embs)

        if len(shard_paths) >= args.shard_size:
            out = args.output_dir / f"teacher_{shard_idx:05d}.pt"
            torch.save({"paths": shard_paths, "embeddings": torch.cat(shard_embeddings, dim=0)}, out)
            shard_idx += 1
            shard_paths = []
            shard_embeddings = []

    if shard_paths:
        out = args.output_dir / f"teacher_{shard_idx:05d}.pt"
        torch.save({"paths": shard_paths, "embeddings": torch.cat(shard_embeddings, dim=0)}, out)


if __name__ == "__main__":
    main()
