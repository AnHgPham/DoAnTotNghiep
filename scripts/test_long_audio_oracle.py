"""Test long audio prediction using ground-truth segment boundaries.

Bypasses energy-based segmentation entirely and runs the encoder/classifier
on each known word region, so we measure model accuracy in isolation.

Usage:
    python scripts/test_long_audio_oracle.py \
        --audio data/test/gsc_demo_20words.wav \
        --timings data/test/gsc_demo_20words.timings.json \
        --enroll yes,no,stop,go,up,down,left,right,on,off \
        --k 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN

SR = 16000


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def pad_or_trim_1s(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    length = wav.shape[-1]
    if length < SR:
        return F.pad(wav, (0, SR - length))
    if length > SR:
        return wav[..., :SR]
    return wav


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle test on long audio")
    parser.add_argument("--audio", type=Path, default=Path("data/test/gsc_demo_20words.wav"))
    parser.add_argument("--timings", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--enroll", type=str,
                        default="yes,no,stop,go,up,down,left,right,on,off")
    parser.add_argument("--k", type=int, default=20, help="Enrollment samples per word")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--gsc-dir", type=Path, default=Path("data/gsc_v2"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.timings is None:
        args.timings = args.audio.with_suffix(".timings.json")

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DSCNN(model_size=cfg["model"]["architecture"][-1]).to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    extractor = MFCCExtractor()
    print(f"Checkpoint: {args.checkpoint} (epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('loss', '?'):.4f})")

    enroll_words = [w.strip() for w in args.enroll.split(",") if w.strip()]
    print(f"\nEnrolling {len(enroll_words)} words with k={args.k} samples each...")

    prototypes: dict[str, torch.Tensor] = {}
    for word in enroll_words:
        files = sorted((args.gsc_dir / word).glob("*.wav"))[:args.k]
        if not files:
            print(f"  {word}: SKIP (no wavs)")
            continue
        embs = []
        for path in files:
            wav = load_audio(path)
            wav = pad_or_trim_1s(wav)
            mfcc = extractor.extract(wav).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = F.normalize(encoder(mfcc), p=2, dim=-1)
            embs.append(emb.squeeze(0).cpu())
        prototypes[word] = torch.stack(embs).mean(0)
        print(f"  {word}: {len(files)} samples -> prototype OK")

    with open(args.timings, "r", encoding="utf-8") as f:
        timing_data = json.load(f)

    audio = load_audio(args.audio).squeeze(0)
    print(f"\nAudio: {args.audio} ({audio.shape[-1]/SR:.2f}s, {len(timing_data['words'])} words)")
    print(f"Threshold: {args.threshold}\n")
    print(f"{'#':>3}  {'Time':>11}  {'Expected':>10}  {'Predicted':>10}  {'Distance':>8}  Top-3")
    print("-" * 100)

    correct = 0
    total = 0
    rejected_when_should_match = 0
    proto_tensor = torch.stack(list(prototypes.values()))
    proto_labels = list(prototypes.keys())

    for idx, item in enumerate(timing_data["words"], start=1):
        start, end = item["start_sample"], item["end_sample"]
        clip = audio[start:end].unsqueeze(0)
        clip = pad_or_trim_1s(clip)

        mfcc = extractor.extract(clip).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = F.normalize(encoder(mfcc), p=2, dim=-1).squeeze(0).cpu()

        dists = torch.cdist(emb.unsqueeze(0), proto_tensor).squeeze(0)
        sorted_idx = torch.argsort(dists)
        best_idx = int(sorted_idx[0].item())
        best_word = proto_labels[best_idx]
        best_dist = float(dists[best_idx].item())

        if best_dist <= args.threshold:
            pred = best_word
        else:
            pred = "unknown"

        expected = item["label"]
        ok = pred == expected
        if ok:
            correct += 1
        elif pred == "unknown" and expected in proto_labels:
            rejected_when_should_match += 1
        total += 1

        top3 = ", ".join(
            f"{proto_labels[int(sorted_idx[i].item())]}:{float(dists[int(sorted_idx[i].item())].item()):.3f}"
            for i in range(min(3, len(sorted_idx)))
        )
        flag = "OK" if ok else "X "
        print(f"{idx:>3}  {item['start_sec']:5.2f}-{item['end_sec']:5.2f}s  "
              f"{expected:>10}  {pred:>10}  {best_dist:>8.4f}  {flag} [{top3}]")

    print("-" * 100)
    acc = correct / total if total else 0.0
    print(f"\nOracle accuracy: {acc:.2%} ({correct}/{total})")
    print(f"Rejected (unknown) when should match: {rejected_when_should_match}")
    print(f"Misclassified (wrong known label): {total - correct - rejected_when_should_match}")


if __name__ == "__main__":
    main()
