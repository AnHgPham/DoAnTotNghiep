"""
Demo Few-Shot Open-Set KWS (terminal only, no web).

Flow:
  1. Load model (best.pt)
  2. Enroll keywords (5 samples each from GSC)
  3. Upload a long audio file -> slide 1-second window -> detect each second
"""

import sys
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import torchaudio

from src.models.dscnn import DSCNN
from src.features.mfcc import MFCCExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
DEFAULT_CKPT = Path("checkpoints/best.pt")


# ==================== Core functions ====================

def load_encoder(checkpoint_path: Path):
    enc = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
        enc.load_state_dict(ckpt["model_state_dict"])
        print(f"  Model   : {checkpoint_path}")
        print(f"  Epoch   : {ckpt.get('epoch', '?')}")
        print(f"  Loss    : {ckpt.get('loss', 0):.6f}")
    else:
        print(f"  WARNING : {checkpoint_path} not found, random weights!")
    enc = enc.to(DEVICE).eval()
    print(f"  Device  : {DEVICE}")
    print(f"  Params  : {sum(p.numel() for p in enc.parameters()):,}")
    return enc


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def get_embedding(enc, mfcc_ext, wav_1s: torch.Tensor) -> torch.Tensor:
    mfcc = mfcc_ext.extract(wav_1s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(enc(mfcc), p=2, dim=-1)
    return emb.squeeze(0).cpu()


def pad_or_trim_1s(wav: torch.Tensor) -> torch.Tensor:
    """Convert any segment to the 1-second input expected by DSCNN."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[-1] < SR:
        wav = F.pad(wav, (0, SR - wav.shape[-1]))
    return wav[..., :SR]


def score_segment(enc, mfcc_ext, prototypes: dict[str, torch.Tensor],
                  segment: torch.Tensor, threshold: float) -> dict:
    """Classify one already-cropped word segment."""
    emb = get_embedding(enc, mfcc_ext, pad_or_trim_1s(segment))
    dists = {
        word: torch.cdist(emb.unsqueeze(0), proto.unsqueeze(0)).item()
        for word, proto in prototypes.items()
    }
    sorted_d = sorted(dists.items(), key=lambda x: x[1])
    best_word, best_dist = sorted_d[0]
    pred = best_word if best_dist <= threshold else "unknown"
    return {
        "pred": pred,
        "dist": best_dist,
        "status": "MATCH" if pred != "unknown" else "REJECT",
        "top3": sorted_d[:3],
    }


def choose_final_prediction(results: list[dict]) -> tuple[str, float]:
    """Collapse many overlapping window predictions into one final label."""
    matches = [r for r in results if r["status"] == "MATCH" and r["pred"] != "unknown"]
    if not matches:
        return "unknown", float("inf")

    counts = Counter(str(r["pred"]) for r in matches)
    best_count = max(counts.values())
    candidate_labels = {label for label, count in counts.items() if count == best_count}
    best = min(
        (r for r in matches if r["pred"] in candidate_labels),
        key=lambda r: float(r["dist"]),
    )
    return str(best["pred"]), float(best["dist"])


def segment_speech_energy(
    wav: torch.Tensor,
    threshold_ratio: float = 0.15,
    min_word_ms: int = 180,
    merge_gap_ms: int = 180,
    pad_ms: int = 120,
) -> list[tuple[int, int]]:
    """Split long audio into likely word segments using short-time energy."""
    if wav.dim() == 2:
        wav = wav.mean(dim=0)

    total = wav.shape[-1]
    frame = int(SR * 0.03)
    hop = int(SR * 0.01)
    if total < frame:
        return [(0, total)] if total > 0 else []

    starts = list(range(0, total - frame + 1, hop))
    energies = []
    for start in starts:
        chunk = wav[start:start + frame]
        energies.append(float(torch.sqrt(torch.mean(chunk * chunk)).item()))

    max_energy = max(energies) if energies else 0.0
    if max_energy <= 1e-6:
        return []

    threshold = max_energy * threshold_ratio
    active_ranges = []
    current_start = None

    for start, energy in zip(starts, energies, strict=True):
        if energy >= threshold and current_start is None:
            current_start = start
        elif energy < threshold and current_start is not None:
            active_ranges.append((current_start, start + frame))
            current_start = None
    if current_start is not None:
        active_ranges.append((current_start, starts[-1] + frame))

    merge_gap = int(SR * merge_gap_ms / 1000)
    merged = []
    for start, end in active_ranges:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    min_len = int(SR * min_word_ms / 1000)
    pad = int(SR * pad_ms / 1000)
    return [
        (max(0, start - pad), min(total, end + pad))
        for start, end in merged
        if end - start >= min_len
    ]


# ==================== Step 1: Enroll ====================

def enroll(enc, mfcc_ext, words: list[str], n_samples: int = 5) -> dict[str, torch.Tensor]:
    prototypes = {}
    for word in words:
        word_dir = Path(f"data/gsc_v2/{word}")
        if not word_dir.exists():
            print(f"    {word:>10s} : SKIP (folder not found)")
            continue
        wav_files = sorted(word_dir.glob("*.wav"))[:n_samples]
        if len(wav_files) < 2:
            print(f"    {word:>10s} : SKIP (not enough files)")
            continue
        embs = []
        for f in wav_files:
            w = load_wav(f)
            if w.shape[-1] < SR:
                w = F.pad(w, (0, SR - w.shape[-1]))
            embs.append(get_embedding(enc, mfcc_ext, w[..., :SR]))
        prototypes[word] = torch.stack(embs).mean(0)
        print(f"    {word:>10s} : {len(wav_files)} samples -> prototype OK")
    return prototypes


# ==================== Step 2: Detect per second ====================

def detect_file(enc, mfcc_ext, prototypes: dict, audio_path: Path,
                threshold: float = 0.8, window_s: float = 1.0, stride_s: float = 0.5,
                expected_label: str | None = None):
    wav = load_wav(audio_path)
    total = wav.shape[-1]
    duration = total / SR
    window = int(SR * window_s)
    stride = int(SR * stride_s)

    print(f"\n  File    : {audio_path.name}")
    print(f"  Length  : {duration:.1f}s ({total} samples)")
    print(f"  Window  : {window_s}s, Stride: {stride_s}s")
    print(f"  Threshold: {threshold}")
    print()
    print(f"  {'Time':>10s}  {'Predicted':>10s}  {'Distance':>10s}  {'Status':>10s}  Top-3")
    print(f"  {'----':>10s}  {'--------':>10s}  {'--------':>10s}  {'------':>10s}  -----")

    pos = 0
    results = []
    while pos + window <= total:
        seg = wav[..., pos:pos + window]
        if seg.shape[-1] < SR:
            seg = F.pad(seg, (0, SR - seg.shape[-1]))
        seg = seg[..., :SR]

        emb = get_embedding(enc, mfcc_ext, seg)
        dists = {}
        for word, proto in prototypes.items():
            dists[word] = torch.cdist(emb.unsqueeze(0), proto.unsqueeze(0)).item()

        sorted_d = sorted(dists.items(), key=lambda x: x[1])
        best_word, best_dist = sorted_d[0]
        if best_dist > threshold:
            pred = "unknown"
            status = "REJECT"
        else:
            pred = best_word
            status = "MATCH"

        t_start = pos / SR
        t_end = (pos + window) / SR
        top3 = ", ".join(f"{w}:{d:.3f}" for w, d in sorted_d[:3])

        print(f"  {t_start:4.1f}-{t_end:4.1f}s  {pred:>10s}  {best_dist:>10.4f}  {status:>10s}  [{top3}]")
        results.append({"t": f"{t_start:.1f}-{t_end:.1f}s", "pred": pred, "dist": best_dist, "status": status})
        pos += stride

    detected = sum(1 for r in results if r["status"] == "MATCH")
    print(f"\n  Summary : {detected}/{len(results)} windows matched, {len(results)-detected} rejected")
    final_pred, final_dist = choose_final_prediction(results)
    votes = Counter(r["pred"] for r in results if r["status"] == "MATCH")
    vote_text = ", ".join(f"{word}:{count}" for word, count in votes.most_common()) or "none"
    print(f"  Final   : {final_pred} (best distance={final_dist:.4f}, votes={vote_text})")
    if expected_label:
        acc = 1.0 if final_pred == expected_label else 0.0
        print(f"  Accuracy: {acc:.2%} (expected={expected_label})")
    return results


def detect_word_segments(
    enc,
    mfcc_ext,
    prototypes: dict[str, torch.Tensor],
    audio_path: Path,
    threshold: float = 0.8,
    expected_labels: list[str] | None = None,
    energy_threshold: float = 0.15,
    min_word_ms: int = 180,
) -> list[dict]:
    """Cut a long utterance into word-like segments and classify each once."""
    wav = load_wav(audio_path)
    segments = segment_speech_energy(
        wav,
        threshold_ratio=energy_threshold,
        min_word_ms=min_word_ms,
    )

    print(f"\n  File    : {audio_path.name}")
    print(f"  Words   : {len(segments)} segments from energy-based cutting")
    print(f"  {'#':>3s}  {'Time':>10s}  {'Predicted':>10s}  {'Distance':>10s}  Top-3")
    print(f"  {'-':>3s}  {'----':>10s}  {'---------':>10s}  {'--------':>10s}  -----")

    results = []
    for idx, (start, end) in enumerate(segments, start=1):
        segment = wav[..., start:end]
        scored = score_segment(enc, mfcc_ext, prototypes, segment, threshold)
        t_start = start / SR
        t_end = end / SR
        top3 = ", ".join(f"{w}:{d:.3f}" for w, d in scored["top3"])
        print(f"  {idx:>3d}  {t_start:4.1f}-{t_end:4.1f}s  {scored['pred']:>10s}  {scored['dist']:>10.4f}  [{top3}]")
        results.append({
            "idx": idx,
            "t": f"{t_start:.1f}-{t_end:.1f}s",
            "pred": scored["pred"],
            "dist": scored["dist"],
        })

    if expected_labels:
        n = min(len(expected_labels), len(results))
        correct = sum(1 for i in range(n) if results[i]["pred"] == expected_labels[i])
        acc = correct / n if n else 0.0
        print(f"\n  Accuracy: {acc:.2%} ({correct}/{n} words)")
        if len(expected_labels) != len(results):
            print(f"  Note    : expected {len(expected_labels)} labels, detected {len(results)} segments")

    return results


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Few-Shot KWS Demo (terminal)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file to detect (WAV). If not given, uses GSC test samples.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT),
                        help="Checkpoint path, e.g. checkpoints/best.pt")
    parser.add_argument("--threshold", type=float, default=0.8, help="L2 distance threshold (default: 0.8)")
    parser.add_argument("--words", type=str, default="yes,no,stop,go,up,down,left,right,on,off",
                        help="Comma-separated keywords to enroll from GSC")
    parser.add_argument("--k", type=int, default=5, help="Samples per keyword for enrollment")
    parser.add_argument("--expected", type=str, default=None,
                        help="Expected final label, or comma-separated labels with --segment-words")
    parser.add_argument("--segment-words", action="store_true",
                        help="Cut long audio into word segments and classify each segment once")
    parser.add_argument("--energy-threshold", type=float, default=0.15,
                        help="Relative energy threshold for word cutting")
    parser.add_argument("--min-word-ms", type=int, default=180,
                        help="Minimum segment length for word cutting")
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  Few-Shot Open-Set Keyword Spotting - Terminal Demo")
    print("=" * 65)

    # --- Step 1: Load model ---
    print("\n[Step 1] Loading model...")
    enc = load_encoder(Path(args.checkpoint))
    mfcc_ext = MFCCExtractor(n_mfcc=40, num_features=10, sample_rate=SR)

    # --- Step 2: Enroll ---
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    print(f"\n[Step 2] Enrolling {len(words)} keywords ({args.k} samples each)...")
    prototypes = enroll(enc, mfcc_ext, words, n_samples=args.k)
    print(f"\n  Enrolled: {list(prototypes.keys())}")

    if not prototypes:
        print("  ERROR: No keywords enrolled. Check data/gsc_v2/ folder.")
        return

    # --- Step 3: Detect ---
    print(f"\n[Step 3] Detection (threshold={args.threshold})")

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"  ERROR: {audio_path} not found")
            return
        if args.segment_words:
            expected_labels = (
                [label.strip() for label in args.expected.split(",") if label.strip()]
                if args.expected else None
            )
            detect_word_segments(
                enc,
                mfcc_ext,
                prototypes,
                audio_path,
                threshold=args.threshold,
                expected_labels=expected_labels,
                energy_threshold=args.energy_threshold,
                min_word_ms=args.min_word_ms,
            )
        else:
            detect_file(
                enc,
                mfcc_ext,
                prototypes,
                audio_path,
                threshold=args.threshold,
                expected_label=args.expected,
            )
    else:
        # Demo with GSC samples: known words + unknown words
        print("\n  --- A. Known words (should MATCH) ---")
        for word in ["yes", "no", "stop", "go", "left"]:
            test_dir = Path(f"data/gsc_v2/{word}")
            files = sorted(test_dir.glob("*.wav"))[10:12]
            for f in files:
                detect_file(enc, mfcc_ext, prototypes, f, threshold=args.threshold)

        print("\n  --- B. Unknown words (should REJECT) ---")
        for word in ["cat", "dog", "bird", "house"]:
            test_dir = Path(f"data/gsc_v2/{word}")
            if not test_dir.exists():
                continue
            files = sorted(test_dir.glob("*.wav"))[:2]
            for f in files:
                detect_file(enc, mfcc_ext, prototypes, f, threshold=args.threshold)

    print(f"\n{'='*65}")
    print("  Done!")
    print("=" * 65)


if __name__ == "__main__":
    main()
