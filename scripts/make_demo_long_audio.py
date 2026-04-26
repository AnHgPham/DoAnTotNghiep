"""Create a long GSC demo audio file with known word labels and timings.

Usage:
    python scripts/make_demo_long_audio.py
    python scripts/make_demo_long_audio.py --words yes,no,stop,go,on,off
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

SR = 16000


def load_gsc_word(word: str, index: int, gsc_dir: Path, raw: bool = False) -> torch.Tensor:
    """Load one GSC sample for a word.

    If raw is False, crop around active speech and normalize amplitude (legacy).
    If raw is True, return the original 1-second clip with no extra processing.
    """
    files = sorted((gsc_dir / word).glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found for word: {word}")

    path = files[index % len(files)]
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[-1] < SR:
        wav = F.pad(wav, (0, SR - wav.shape[-1]))
    wav = wav[..., :SR]
    if raw:
        return wav
    return crop_and_normalize(wav)


def crop_and_normalize(wav: torch.Tensor, pad_ms: int = 120) -> torch.Tensor:
    """Crop around active speech and normalize RMS for clearer demo audio."""
    mono = wav.mean(dim=0) if wav.dim() == 2 else wav.squeeze(0)
    if mono.numel() == 0:
        return torch.zeros(1, SR)

    frame = int(SR * 0.02)
    hop = int(SR * 0.01)
    if mono.numel() < frame:
        cropped = mono
    else:
        starts = list(range(0, mono.numel() - frame + 1, hop))
        energies = torch.tensor([
            torch.sqrt(torch.mean(mono[start:start + frame] ** 2)).item()
            for start in starts
        ])
        max_energy = float(energies.max().item()) if energies.numel() else 0.0
        if max_energy <= 1e-6:
            cropped = mono
        else:
            active = torch.where(energies >= max(0.0005, max_energy * 0.12))[0]
            if active.numel() == 0:
                cropped = mono
            else:
                pad = int(SR * pad_ms / 1000)
                start = max(0, starts[int(active[0])] - pad)
                end = min(mono.numel(), starts[int(active[-1])] + frame + pad)
                cropped = mono[start:end]

    rms = torch.sqrt(torch.mean(cropped ** 2)).clamp_min(1e-6)
    cropped = (cropped / rms * 0.08).clamp(-0.9, 0.9)
    return cropped.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create long GSC demo audio")
    parser.add_argument(
        "--words",
        default="yes,no,stop,go,on,off,up,down,left,right,stop,yes,no,go,down,yes",
        help="Comma-separated word sequence",
    )
    parser.add_argument("--gsc-dir", type=Path, default=Path("data/gsc_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/test/gsc_long_many_words.wav"))
    parser.add_argument("--silence-ms", type=int, default=800)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Concatenate raw GSC clips without cropping or RMS normalization.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=120,
        help="Starting sample index; keep away from first enrollment samples.",
    )
    args = parser.parse_args()

    words = [w.strip().lower() for w in args.words.split(",") if w.strip()]
    silence = torch.zeros(1, int(SR * args.silence_ms / 1000))

    chunks = []
    timings = []
    cursor = 0
    for idx, word in enumerate(words):
        clip = load_gsc_word(word, args.start_index + idx * 17, args.gsc_dir, raw=args.raw)
        timings.append({
            "label": word,
            "start_sample": cursor,
            "end_sample": cursor + clip.shape[-1],
            "start_sec": cursor / SR,
            "end_sec": (cursor + clip.shape[-1]) / SR,
        })
        chunks.append(clip)
        cursor += clip.shape[-1]
        if idx != len(words) - 1:
            chunks.append(silence)
            cursor += silence.shape[-1]

    audio = torch.cat(chunks, dim=-1).clamp(-1.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), audio, SR)

    labels_path = args.output.with_suffix(".labels.txt")
    labels_path.write_text(",".join(words), encoding="utf-8")

    timings_path = args.output.with_suffix(".timings.json")
    timings_path.write_text(
        json.dumps({"sample_rate": SR, "words": timings}, indent=2), encoding="utf-8"
    )

    print(f"Saved audio   : {args.output}")
    print(f"Duration      : {audio.shape[-1] / SR:.2f}s")
    print(f"Labels        : {labels_path}")
    print(f"Timings       : {timings_path}")
    print(f"Expected      : {','.join(words)}")


if __name__ == "__main__":
    main()
