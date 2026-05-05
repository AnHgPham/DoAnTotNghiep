"""Compare 2x2 combinations of segmentation x scoring on long audio.

Aligns predicted segments to ground-truth word timings using temporal overlap,
so missed segments do not propagate alignment errors.

Usage:
    python scripts/compare_demo_methods.py
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torchaudio

import demo_web as dw

SR = dw.SR


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def enroll(words: list[str], k: int) -> None:
    dw.prototypes.clear()
    dw.sample_count.clear()
    for word in words:
        files = sorted(Path(f"data/gsc_v2/{word}").glob("*.wav"))[:k]
        if not files:
            continue
        embs = []
        for path in files:
            wav = load_wav(path)
            wav = dw.pad_or_trim_1s(wav)
            embs.append(dw.embed(wav))
        dw.prototypes[word] = torch.stack(embs).mean(0)
        dw.sample_count[word] = len(files)


def detect_segments(wav: torch.Tensor, seg_method: str, scoring: str,
                    threshold: float, min_dur_ms: int = 200) -> list[dict]:
    """Run segmentation + scoring; return list of {start, end, pred, score}."""
    if seg_method == "Silero VAD":
        segments = dw._word_segments_by_vad(wav, min_dur_ms)
        if not segments:
            segments = dw._word_segments_by_energy(wav, min_duration_ms=min_dur_ms)
    else:
        segments = dw._word_segments_by_energy(wav, min_duration_ms=min_dur_ms)

    out = []
    for start, end in segments:
        scored = dw._score_cluster(wav[..., start:end], threshold, scoring)
        out.append({
            "start": start / SR,
            "end": end / SR,
            "pred": scored["pred"],
            "score": scored["score"],
        })
    return out


def align_to_groundtruth(predicted: list[dict], expected_words: list[dict]) -> list[str]:
    """For each ground-truth word interval, pick the prediction that overlaps most.

    Returns one predicted label per ground-truth word (or 'missed' if nothing
    overlaps).
    """
    aligned = []
    for gt in expected_words:
        best_overlap = 0.0
        best_pred = "missed"
        for p in predicted:
            overlap = max(0.0, min(p["end"], gt["end_sec"]) - max(p["start"], gt["start_sec"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pred = p["pred"]
        aligned.append(best_pred)
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=Path("data/test/gsc_demo_diverse.wav"))
    parser.add_argument("--enroll", type=str,
                        default="yes,no,stop,happy,bird,dog,tree,marvin,four,learn")
    parser.add_argument("--k", type=int, default=5,
                        help="GSC enrollment samples per word (thesis: k=5)")
    parser.add_argument("--threshold-l2", type=float, default=0.85)
    parser.add_argument("--threshold-prob", type=float, default=0.20)
    args = parser.parse_args()

    dw.init()
    enroll_words = [w.strip() for w in args.enroll.split(",") if w.strip()]
    enroll(enroll_words, args.k)

    timings_path = args.audio.with_suffix(".timings.json")
    if not timings_path.exists():
        raise FileNotFoundError(f"Need timings: {timings_path}")
    timing_data = json.loads(timings_path.read_text(encoding="utf-8"))
    expected_segments = timing_data["words"]

    wav = load_wav(args.audio)

    print(f"\nAudio       : {args.audio} ({wav.shape[-1]/SR:.1f}s)")
    print(f"Expected    : {len(expected_segments)} words")
    print(f"Enrolled    : {len(enroll_words)} words, k={args.k}")
    print(f"Threshold L2={args.threshold_l2}, Prob={args.threshold_prob}\n")

    combos = [
        ("Energy", "L2", args.threshold_l2),
        ("Energy", "Probability", args.threshold_prob),
        ("Silero VAD", "L2", args.threshold_l2),
        ("Silero VAD", "Probability", args.threshold_prob),
    ]

    print(f"{'Segmentation':<14}  {'Scoring':<12}  {'thr':>5}  {'segs':>5}  "
          f"{'aligned':>8}  {'correct':>8}  {'accuracy':>9}")
    print("-" * 78)

    rows = []
    for seg, scoring, thr in combos:
        predicted = detect_segments(wav, seg, scoring, thr)
        aligned = align_to_groundtruth(predicted, expected_segments)
        correct = sum(1 for i, w in enumerate(expected_segments)
                      if aligned[i] == w["label"])
        accuracy = correct / len(expected_segments) if expected_segments else 0.0
        rows.append({
            "seg": seg, "scoring": scoring, "thr": thr,
            "n_segments": len(predicted), "aligned": aligned,
            "correct": correct, "total": len(expected_segments),
            "accuracy": accuracy,
        })
        print(f"{seg:<14}  {scoring:<12}  {thr:>5.2f}  {len(predicted):>5d}  "
              f"{len(aligned):>5d}/{len(expected_segments):<3d}  "
              f"{correct:>3d}/{len(expected_segments):<3d}  {accuracy:>9.2%}")

    print("-" * 78)
    best = max(rows, key=lambda r: r["accuracy"])
    print(f"\nBest combo: {best['seg']} + {best['scoring']} "
          f"(threshold={best['thr']:.2f}) -> {best['accuracy']:.2%}")
    print("\nPer-word predictions for best combo:")
    print(f"{'#':>3}  {'Expected':>10}  {'Aligned':>10}")
    for i, gt in enumerate(expected_segments, start=1):
        pred = best["aligned"][i - 1]
        flag = "OK" if pred == gt["label"] else "X "
        print(f"{i:>3}  {gt['label']:>10}  {pred:>10}  {flag}")


if __name__ == "__main__":
    main()
