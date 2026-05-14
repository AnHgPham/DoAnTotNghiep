"""Precompute Wav2Vec2 teacher embeddings for KD training.

Example:
    python scripts/precompute_teacher_embeddings.py \
      --data-dir data/mswc_en --split train --output-dir outputs/teacher_w2v2_train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.teacher import Wav2Vec2Teacher

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000


def load_words(data_dir: Path, split: str) -> list[str] | None:
    path = data_dir / "splits" / f"{split}_words.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_wavs(data_dir: Path, words: list[str] | None, max_per_word: int) -> list[Path]:
    roots = [data_dir / "clips", data_dir]
    base = next((p for p in roots if p.exists()), data_dir)
    selected_words = words or sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and any(d.glob("*.wav"))
    )
    paths: list[Path] = []
    for word in selected_words:
        word_dir = base / word
        wavs = sorted(word_dir.glob("*.wav"))
        if max_per_word > 0:
            wavs = wavs[:max_per_word]
        paths.extend(wavs)
    return paths


def load_wave(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[-1] < TARGET_LENGTH:
        wav = F.pad(wav, (0, TARGET_LENGTH - wav.shape[-1]))
    return wav[..., :TARGET_LENGTH]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Wav2Vec2 teacher embeddings")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mswc_en"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "eval", "all"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--head-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-per-word", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    words = None if args.split == "all" else load_words(args.data_dir, args.split)
    wav_paths = discover_wavs(args.data_dir, words, args.max_per_word)
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under {args.data_dir}")

    done_paths = set()
    for shard in args.output_dir.glob("teacher_*.pt"):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        done_paths.update(payload.get("paths", []))

    pending = [p for p in wav_paths if p.as_posix() not in done_paths]
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
