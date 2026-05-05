"""
Simple web demo: Enrollment first, then Detect.
Run: python -u demo_web.py
Open: http://127.0.0.1:7860
Requires: Gradio 6.x (``pip install -r requirements.txt``).

Features showcased:
  1. Few-shot enrollment (GSC samples or microphone)
  2. Direct L2 distance vs Probability-based scoring (ablation)
  3. Open-set rejection with adjustable threshold
  4. Energy-based or Silero VAD-based long-file segmentation
  5. Test-time augmentation (TTA) with majority voting
  6. Spectral-gating denoiser (EXT-1)
  7. Save/Load enrollment profiles to JSON
  8. Model info + latest evaluation results
"""

import json
import os
import socket
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
CKPT = Path("checkpoints/best.pt")
ENROLL_PROFILES_DIR = Path("data/enroll_profiles")

WORD_PRESETS: dict[str, str] = {
    "IoT (yes/no/...)": "yes,no,stop,go,up,down,left,right,on,off",
    "Diverse phonetic": "yes,no,stop,happy,bird,dog,tree,marvin,four,learn",
    "Numbers": "zero,one,two,three,four,five,six,seven,eight,nine",
    "Names + commands": "marvin,sheila,stop,go,yes,no,happy,wow",
}

KNOWN_GSC_WORDS = sorted([
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual",
])

encoder: DSCNN | None = None
mfcc_ext: MFCCExtractor | None = None
prototypes: dict[str, torch.Tensor] = {}
sample_count: dict[str, int] = {}
denoiser_instance = None


# ============================================================
# Initialization
# ============================================================

def init() -> None:
    global encoder, mfcc_ext
    mfcc_ext = MFCCExtractor(n_mfcc=40, num_features=10, sample_rate=SR)
    encoder = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))
    if CKPT.exists():
        ckpt = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"  Model : {CKPT} (epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss',0):.6f})")
    else:
        print(f"  WARNING: {CKPT} not found")
    encoder = encoder.to(DEVICE).eval()
    print(f"  Device: {DEVICE}, Params: {sum(p.numel() for p in encoder.parameters()):,}")


def get_denoiser():
    global denoiser_instance
    if denoiser_instance is None:
        try:
            from src.enhancements.denoiser import Denoiser
            denoiser_instance = Denoiser(backend="spectral_gate", enabled=True)
            print("  Denoiser: spectral_gate ready")
        except Exception as exc:
            print(f"  Denoiser unavailable: {exc}")
            denoiser_instance = False
    return denoiser_instance if denoiser_instance is not False else None


# ============================================================
# Audio helpers
# ============================================================

def to_wav(audio_input) -> torch.Tensor | None:
    if audio_input is None:
        return None
    sr, data = audio_input
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data).unsqueeze(0)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav


def maybe_denoise(wav: torch.Tensor, enable: bool) -> torch.Tensor:
    if not enable or wav is None:
        return wav
    dn = get_denoiser()
    if dn is None:
        return wav
    try:
        return dn.denoise(wav, sr=SR)
    except Exception as exc:
        print(f"Denoise failed: {exc}")
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


def to_wav_1s(audio_input, denoise: bool = False) -> torch.Tensor | None:
    wav = to_wav(audio_input)
    if wav is None:
        return None
    wav = maybe_denoise(wav, denoise)
    return pad_or_trim_1s(wav)


def embed(wav_1s: torch.Tensor) -> torch.Tensor:
    mfcc = mfcc_ext.extract(wav_1s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(encoder(mfcc), p=2, dim=-1)
    return emb.squeeze(0).cpu()


def status_md() -> str:
    if not prototypes:
        return "_No keywords enrolled._"
    items = [f"`{w}` ({sample_count.get(w, 0)})" for w in prototypes]
    return f"**{len(prototypes)} keywords enrolled:** " + ", ".join(items)


# ============================================================
# Enrollment (Tab 1)
# ============================================================

def enroll_gsc(words_text: str, k: int):
    words = [w.strip() for w in words_text.split(",") if w.strip()]
    if not words:
        return "Enter words separated by commas.", status_md()
    msgs = []
    for word in words:
        d = Path(f"data/gsc_v2/{word}")
        if not d.exists():
            msgs.append(f"`{word}`: not found")
            continue
        files = sorted(d.glob("*.wav"))[:k]
        if not files:
            msgs.append(f"`{word}`: no WAV files")
            continue
        embs = []
        for f in files:
            w, sr = torchaudio.load(str(f))
            if sr != SR:
                w = torchaudio.transforms.Resample(sr, SR)(w)
            if w.shape[-1] < SR:
                w = F.pad(w, (0, SR - w.shape[-1]))
            embs.append(embed(w[..., :SR]))
        prototypes[word] = torch.stack(embs).mean(0)
        sample_count[word] = len(files)
        msgs.append(f"`{word}`: OK ({len(files)} samples)")
    return "\n".join(msgs), status_md()


def enroll_mic(keyword: str, audio):
    if not keyword or not keyword.strip():
        return "Type keyword name first.", status_md()
    keyword = keyword.strip().lower()
    wav = to_wav_1s(audio)
    if wav is None:
        return "No audio.", status_md()
    e = embed(wav)
    if keyword in prototypes:
        n = sample_count.get(keyword, 1)
        prototypes[keyword] = (prototypes[keyword] * n + e) / (n + 1)
        sample_count[keyword] = n + 1
    else:
        prototypes[keyword] = e
        sample_count[keyword] = 1
    return f"Added `{keyword}` ({sample_count[keyword]} total)", status_md()


def use_preset(preset_name: str) -> str:
    return WORD_PRESETS.get(preset_name, "")


def clear_all():
    prototypes.clear()
    sample_count.clear()
    return "Cleared.", status_md()


def save_profile(profile_name: str):
    if not prototypes:
        return "Nothing to save."
    name = (profile_name or "").strip() or "default"
    ENROLL_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = ENROLL_PROFILES_DIR / f"{name}.json"
    payload = {
        "labels": list(prototypes.keys()),
        "sample_count": sample_count,
        "embeddings": {k: v.tolist() for k, v in prototypes.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return f"Saved profile to `{path}` ({len(prototypes)} keywords)."


def load_profile(profile_name: str):
    name = (profile_name or "").strip() or "default"
    path = ENROLL_PROFILES_DIR / f"{name}.json"
    if not path.exists():
        return f"Profile `{name}` not found.", status_md()
    payload = json.loads(path.read_text(encoding="utf-8"))
    prototypes.clear()
    sample_count.clear()
    for label, vec in payload.get("embeddings", {}).items():
        prototypes[label] = torch.tensor(vec, dtype=torch.float32)
        sample_count[label] = payload.get("sample_count", {}).get(label, 0)
    return f"Loaded profile `{name}` ({len(prototypes)} keywords).", status_md()


def list_profiles() -> list[str]:
    if not ENROLL_PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in ENROLL_PROFILES_DIR.glob("*.json"))


# ============================================================
# Single detection (Tab 2)
# ============================================================

def _l2_distances(emb: torch.Tensor) -> dict[str, float]:
    return {
        w: torch.cdist(emb.unsqueeze(0), p.unsqueeze(0)).item()
        for w, p in prototypes.items()
    }


def _scoring_softmax(dists: dict[str, float]) -> dict[str, float]:
    """Softmax over -distances giving per-class probability (Rusci baseline)."""
    keys = list(dists.keys())
    arr = np.array([dists[k] for k in keys], dtype=float)
    logits = -arr
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return dict(zip(keys, probs))


def detect_single(audio, threshold, denoise, scoring):
    wav = to_wav_1s(audio, denoise=denoise)
    if wav is None:
        return "_No audio._", None
    if not prototypes:
        return "Enroll keywords first (Tab 1).", None

    e = embed(wav)
    dists = _l2_distances(e)
    sd = sorted(dists.items(), key=lambda x: x[1])
    best_w, best_d = sd[0]

    if scoring == "Probability":
        probs = _scoring_softmax(dists)
        best_w = max(probs, key=probs.get)
        best_p = probs[best_w]
        score_label = "max prob"
        score_val = best_p
        accept = best_p >= threshold
        body = (
            f"### {best_w.upper() if accept else 'UNKNOWN (rejected)'}\n"
            f"Score (max prob): `{score_val:.3f}`  •  threshold: `{threshold:.2f}`\n"
            f"Distance to nearest: `{best_d:.4f}`"
        )
    else:
        accept = best_d <= threshold
        score_val = best_d
        score_label = "L2 distance"
        body = (
            f"### {best_w.upper() if accept else 'UNKNOWN (rejected)'}\n"
            f"Distance: `{best_d:.4f}`  •  threshold: `{threshold:.2f}`"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.0, len(sd) * 0.42 + 1.0)),
                             gridspec_kw={"width_ratios": [3, 2]})
    words = [w for w, _ in sd]
    if scoring == "Probability":
        probs = _scoring_softmax(dists)
        values = [probs[w] for w in words]
        bars = axes[0].barh(words, values, color=["#4CAF50" if v >= threshold else "#ef5350" for v in values])
        axes[0].axvline(threshold, color="orange", linestyle="--", lw=2, label=f"threshold={threshold:.2f}")
        axes[0].set_xlabel("Probability (softmax over -distances)")
        axes[0].set_xlim(0, 1)
        for bar, val in zip(bars, values):
            axes[0].text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=9)
    else:
        values = [d for _, d in sd]
        bars = axes[0].barh(words, values, color=["#4CAF50" if v <= threshold else "#ef5350" for v in values])
        axes[0].axvline(threshold, color="orange", linestyle="--", lw=2, label=f"threshold={threshold:.2f}")
        axes[0].set_xlabel("L2 distance (lower = more similar)")
        for bar, val in zip(bars, values):
            axes[0].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    axes[0].set_title(f"{score_label} per prototype")

    mfcc = mfcc_ext.extract(wav).squeeze(0).numpy()
    im = axes[1].imshow(mfcc.T, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("MFCC features (47 frames x 10 coef)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Coef")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    return body, fig


# ============================================================
# Long-file detection (Tab 3)
# ============================================================

def _word_segments_by_energy(
    wav: torch.Tensor,
    min_duration_ms: int,
    merge_gap_ms: int = 350,
    pad_ms: int = 120,
) -> list[tuple[int, int]]:
    mono = wav.mean(dim=0) if wav.dim() == 2 else wav
    total = mono.shape[-1]
    frame = int(SR * 0.03)
    hop = int(SR * 0.01)
    if total < frame:
        return [(0, total)] if total > 0 else []
    starts = list(range(0, total - frame + 1, hop))
    energies = [
        float(torch.sqrt(torch.mean(mono[start:start + frame] ** 2)).item())
        for start in starts
    ]
    max_energy = max(energies) if energies else 0.0
    if max_energy <= 1e-6:
        return []
    threshold = max(0.0005, max_energy * 0.08)
    active, current_start = [], None
    for start, energy in zip(starts, energies, strict=True):
        if energy >= threshold and current_start is None:
            current_start = start
        elif energy < threshold and current_start is not None:
            active.append((current_start, start + frame))
            current_start = None
    if current_start is not None:
        active.append((current_start, starts[-1] + frame))
    merge_gap = int(SR * merge_gap_ms / 1000)
    merged = []
    for start, end in active:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    min_len = int(SR * min_duration_ms / 1000)
    pad = int(SR * pad_ms / 1000)
    return [
        (max(0, start - pad), min(total, end + pad))
        for start, end in merged
        if min(total, end + pad) - max(0, start - pad) >= min_len
    ]


def _word_segments_by_vad(wav: torch.Tensor, min_duration_ms: int) -> list[tuple[int, int]]:
    try:
        from src.streaming.vad_engine import SileroVAD
    except Exception as exc:
        print(f"VAD unavailable: {exc}")
        return []
    try:
        vad = SileroVAD(threshold=0.5, min_speech_ms=min(min_duration_ms, 250), device=DEVICE)
        ts = vad.get_speech_timestamps(wav.squeeze(0) if wav.dim() == 2 else wav)
    except Exception as exc:
        print(f"VAD timestamps failed: {exc}")
        return []
    segments = []
    for item in ts:
        start = int(item["start"])
        end = int(item["end"])
        if end - start >= int(SR * min_duration_ms / 1000) // 2:
            segments.append((start, end))
    return segments


_TTA_SHIFT_COUNT = 5


def _tta_views(segment: torch.Tensor) -> list[torch.Tensor]:
    if segment.dim() == 1:
        segment = segment.unsqueeze(0)
    length = segment.shape[-1]
    if length >= SR:
        windows = [segment[..., :SR]]
        if length >= SR + SR // 4:
            offset = (length - SR) // 2
            windows.append(segment[..., offset:offset + SR])
            windows.append(segment[..., -SR:])
        return windows
    pad_total = SR - length
    fractions = [i / (_TTA_SHIFT_COUNT - 1) for i in range(_TTA_SHIFT_COUNT)]
    return [
        F.pad(segment, (int(round(pad_total * f)), pad_total - int(round(pad_total * f))))
        for f in fractions
    ]


def _score_cluster(segment: torch.Tensor, threshold: float, scoring: str) -> dict:
    views = _tta_views(segment)
    embeddings = [embed(view) for view in views]
    averaged = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=-1)
    dists = _l2_distances(averaged)
    sd = sorted(dists.items(), key=lambda x: x[1])
    if scoring == "Probability":
        probs = _scoring_softmax(dists)
        pred_label = max(probs, key=probs.get)
        score = probs[pred_label]
        accept = score >= threshold
    else:
        pred_label = sd[0][0]
        score = sd[0][1]
        accept = score <= threshold

    per_view_preds = []
    for emb_view in embeddings:
        view_dists = _l2_distances(emb_view)
        per_view_preds.append(min(view_dists, key=view_dists.get))
    vote_counts = Counter(per_view_preds)
    return {
        "pred": pred_label if accept else "unknown",
        "raw_pred": pred_label,
        "score": score,
        "dist": sd[0][1],
        "top3": ", ".join(f"{w}:{d:.3f}" for w, d in sd[:3]),
        "votes": ", ".join(f"{label}:{count}" for label, count in vote_counts.most_common()),
        "n_views": len(views),
    }


def _accuracy_line(preds: list[str], expected_text: str) -> str:
    expected = [w.strip().lower() for w in expected_text.split(",") if w.strip()]
    if not expected:
        return ""
    n = min(len(preds), len(expected))
    correct = sum(1 for i in range(n) if preds[i] == expected[i])
    acc = correct / len(expected) if expected else 0.0
    note = "" if len(preds) == len(expected) else f" (predicted {len(preds)}, expected {len(expected)})"
    return f"\n### Accuracy\n**{acc:.2%}** ({correct}/{len(expected)} words){note}"


def detect_long(audio, threshold, seg_method, denoise, scoring,
                min_duration_ms, expected_text):
    wav = to_wav(audio)
    if wav is None:
        return "_No audio._", None
    if not prototypes:
        return "Enroll keywords first (Tab 1).", None
    min_duration_ms = int(round(float(min_duration_ms)))
    min_duration_ms = max(80, min(5000, min_duration_ms))
    wav = maybe_denoise(wav, denoise)
    total = wav.shape[-1]

    if seg_method == "Silero VAD":
        segments = _word_segments_by_vad(wav, min_duration_ms)
        if not segments:
            segments = _word_segments_by_energy(wav, min_duration_ms=min_duration_ms)
            method_used = "Energy (VAD fallback)"
        else:
            method_used = "Silero VAD"
    else:
        segments = _word_segments_by_energy(wav, min_duration_ms=min_duration_ms)
        method_used = "Energy"

    merged = []
    for start, end in segments:
        scored = _score_cluster(wav[..., start:end], threshold, scoring)
        merged.append({
            "t0": start / SR,
            "t1": end / SR,
            "pred": scored["pred"],
            "raw_pred": scored["raw_pred"],
            "dist": scored["dist"],
            "score": scored["score"],
            "top3": scored["top3"],
            "votes": scored["votes"],
            "n_views": scored["n_views"],
        })

    preds = [m["pred"] for m in merged if m["pred"] != "unknown"]
    counts = Counter(preds)
    lines = [
        f"**{total/SR:.1f}s audio**, segmented into **{len(merged)} regions** "
        f"(method: {method_used}, scoring: {scoring}, denoise: {'on' if denoise else 'off'})\n"
    ]
    lines.append(f"### Final word sequence ({len(preds)} keywords)")
    lines.append(" -> ".join(f"**{p}**" for p in preds) if preds else "(no keyword detected)")
    if counts:
        vote_text = ", ".join(f"{word}:{count}" for word, count in counts.most_common())
        lines.append(f"\nVotes summary: {vote_text}")
    if expected_text:
        lines.append(_accuracy_line([m["pred"] for m in merged], expected_text))

    lines.append("\n### Per-segment detail")
    score_label = "Prob" if scoring == "Probability" else "Dist"
    lines.append(f"| # | Time | TTA | Final word | {score_label} | Votes | Top-3 |")
    lines.append("|---|------|-----|------------|-------|-------|-------|")
    for idx, m in enumerate(merged, start=1):
        label_md = f"**{m['pred']}**" if m["pred"] != "unknown" else "_unknown_"
        score_str = f"{m['score']:.3f}" if scoring == "Probability" else f"{m['dist']:.4f}"
        lines.append(
            f"| {idx} | {m['t0']:.1f}-{m['t1']:.1f}s | {m['n_views']} | "
            f"{label_md} | {score_str} | {m['votes']} | {m['top3']} |"
        )

    fig, ax = plt.subplots(figsize=(12, 3))
    for m in merged:
        color = "#4CAF50" if m["pred"] != "unknown" else "#ef5350"
        ax.barh(0, m["t1"] - m["t0"], left=m["t0"], height=0.6, color=color, edgecolor="white")
        label = m["pred"] if m["pred"] != "unknown" else "?"
        ax.text((m["t0"] + m["t1"]) / 2, 0, label, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Timeline ({method_used} segmentation)")
    ax.set_xlim(0, max(total / SR, 0.5))
    fig.tight_layout()
    return "\n".join(lines), fig


# ============================================================
# Open-set test (Tab 4)
# ============================================================

def open_set_test(unknown_words_text: str, k_each: int, threshold: float, denoise: bool, scoring: str):
    if not prototypes:
        return "Enroll keywords first (Tab 1).", None
    unknown_words = [w.strip().lower() for w in unknown_words_text.split(",") if w.strip()]
    unknown_words = [w for w in unknown_words if w not in prototypes]
    if not unknown_words:
        return "Enter a comma-separated list of GSC words NOT in your enrolled set.", None

    rows = []
    rejected_unknown, total_unknown = 0, 0
    accepted_known, total_known = 0, 0
    for word in list(prototypes.keys())[:5]:
        d = Path(f"data/gsc_v2/{word}")
        files = sorted(d.glob("*.wav"))[200:200 + k_each]
        for f in files:
            wav = _load_wav_safe(f, denoise)
            if wav is None:
                continue
            outcome = _classify_for_open_set(wav, threshold, scoring)
            total_known += 1
            if outcome["accept"] and outcome["pred"] == word:
                accepted_known += 1
            rows.append({"truth": word, "type": "known", **outcome})

    for word in unknown_words:
        d = Path(f"data/gsc_v2/{word}")
        if not d.exists():
            continue
        files = sorted(d.glob("*.wav"))[:k_each]
        for f in files:
            wav = _load_wav_safe(f, denoise)
            if wav is None:
                continue
            outcome = _classify_for_open_set(wav, threshold, scoring)
            total_unknown += 1
            if not outcome["accept"]:
                rejected_unknown += 1
            rows.append({"truth": word, "type": "unknown", **outcome})

    text = []
    text.append(f"### Open-set rejection test (threshold={threshold:.2f}, scoring={scoring})")
    if total_known:
        text.append(f"- **Known words**: {accepted_known}/{total_known} accepted "
                    f"({accepted_known/total_known:.1%}). Higher = better.")
    if total_unknown:
        text.append(f"- **Unknown words**: {rejected_unknown}/{total_unknown} correctly rejected "
                    f"({rejected_unknown/total_unknown:.1%}). Higher = better.")
    text.append("\n| Truth | Type | Predicted | Score | Accepted? |")
    text.append("|-------|------|-----------|-------|-----------|")
    for r in rows[:60]:
        status = "yes" if r["accept"] else "no"
        score_str = f"{r['score']:.3f}"
        text.append(f"| {r['truth']} | {r['type']} | {r['pred']} | {score_str} | {status} |")
    if len(rows) > 60:
        text.append(f"\n_Showing first 60 of {len(rows)} samples._")

    fig, ax = plt.subplots(figsize=(8, 4))
    knowns = [r["score"] for r in rows if r["type"] == "known"]
    unknowns = [r["score"] for r in rows if r["type"] == "unknown"]
    if knowns:
        ax.hist(knowns, bins=20, alpha=0.65, color="#4CAF50", label="known")
    if unknowns:
        ax.hist(unknowns, bins=20, alpha=0.65, color="#ef5350", label="unknown")
    ax.axvline(threshold, color="orange", linestyle="--", lw=2, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Score (probability)" if scoring == "Probability" else "L2 distance")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution: known vs unknown")
    ax.legend()
    fig.tight_layout()
    return "\n".join(text), fig


def _load_wav_safe(path: Path, denoise: bool) -> torch.Tensor | None:
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        return None
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    wav = maybe_denoise(wav, denoise)
    return pad_or_trim_1s(wav)


def _classify_for_open_set(wav: torch.Tensor, threshold: float, scoring: str) -> dict:
    e = embed(wav)
    dists = _l2_distances(e)
    if scoring == "Probability":
        probs = _scoring_softmax(dists)
        best_label = max(probs, key=probs.get)
        score = probs[best_label]
        accept = score >= threshold
    else:
        best_label = min(dists, key=dists.get)
        score = dists[best_label]
        accept = score <= threshold
    return {"pred": best_label, "score": score, "accept": accept}


# ============================================================
# Model info (Tab 5)
# ============================================================

def model_info_md() -> str:
    if encoder is None:
        return "_Encoder not loaded._"
    ckpt_info = "(no checkpoint info)"
    if CKPT.exists():
        ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
        ckpt_info = f"epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?'):.6f}"

    lines = [
        f"### Encoder: DSCNN-L (NORM)",
        f"- Checkpoint: `{CKPT}` ({ckpt_info})",
        f"- Parameters: **{sum(p.numel() for p in encoder.parameters()):,}**",
        f"- Embedding dim: 276 (L2-normalized)",
        f"- Device: {DEVICE}",
        f"- MFCC: 40 computed -> 10 used, 47 frames, n_fft=1024",
        "",
        "### Latest evaluation results",
    ]

    json_files = [
        ("results/gsc_fixed_results.json", "GSC Fixed (k=5)"),
        ("results/gsc_random_results.json", "GSC Random (k=5)"),
        ("results/gsc_fixed_k20_results.json", "GSC Fixed (k=20)"),
        ("results/kshot_ablation.json", "K-shot ablation"),
        ("results/denoiser_ablation.json", "EXT-1 Denoiser"),
        ("results/streaming_latency.json", "EXT-2 Streaming"),
    ]
    for rel, label in json_files:
        path = Path(rel)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rel.endswith("kshot_ablation.json"):
            lines.append(f"\n**{label}** (`{rel}`):")
            lines.append("| k | AUC | EER | ACC@5%FAR | KW-ACC | F1 |")
            lines.append("|---|-----|-----|-----------|--------|----|")
            for row in data:
                lines.append(
                    f"| {row['k']} | {row['auc']:.3f} | {row['eer']:.3f} | "
                    f"{row['open_set_acc_at_far']:.3f} | {row['keyword_acc']:.3f} | {row['f1']:.3f} |"
                )
        elif rel.endswith("denoiser_ablation.json"):
            lines.append(f"\n**{label}** (`{rel}`):")
            lines.append("| Condition | AUC | KW-ACC | F1 |")
            lines.append("|-----------|-----|--------|----|")
            for row in data:
                lines.append(
                    f"| SNR={row['snr_db']}, dn={row['denoiser']} | "
                    f"{row['auc']:.3f} | {row['keyword_acc']:.3f} | {row['f1']:.3f} |"
                )
        elif rel.endswith("streaming_latency.json"):
            lines.append(f"\n**{label}** (`{rel}`):")
            lines.append("| Step | Mean (ms) | RTF |")
            lines.append("|------|-----------|-----|")
            for row in data:
                rtf = f"{row.get('rtf', 0):.3f}" if "rtf" in row else "-"
                lines.append(f"| {row['step']} | {row.get('mean_ms', 0):.2f} | {rtf} |")
        else:
            if isinstance(data, dict):
                lines.append(
                    f"- {label}: AUC={data.get('auc', 0):.3f}, "
                    f"EER={data.get('eer', 0):.3f}, "
                    f"KW-ACC={data.get('keyword_acc', 0):.3f}, "
                    f"F1={data.get('f1', 0):.3f}"
                )
    return "\n".join(lines)


# ============================================================
# UI
# ============================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Few-Shot Open-Set KWS - Full Demo") as app:
        gr.Markdown(
            "# Few-Shot Open-Set Keyword Spotting\n"
            "End-to-end demo with denoising (EXT-1), VAD segmentation (EXT-2), "
            "Direct L2 vs Probability scoring, open-set rejection, and TTA voting. "
            "**Enrollment is fixed to k=5** samples per keyword (GSC), matching the evaluation protocol."
        )

        with gr.Tab("1. Enrollment"):
            gr.Markdown("Register keywords from the GSC dataset or your microphone.")
            with gr.Row():
                preset = gr.Dropdown(label="Quick preset", choices=list(WORD_PRESETS.keys()), value=None)
                gsc_words = gr.Textbox(value=WORD_PRESETS["IoT (yes/no/...)"],
                                       label="Words (comma-separated)")
                gsc_k = gr.Slider(
                    minimum=5, maximum=5, value=5, step=1, interactive=False,
                    label="Samples per word (k) = 5 (fixed; thesis / evaluation protocol)",
                )
                gsc_btn = gr.Button("Enroll from GSC", variant="primary")
            preset.change(use_preset, [preset], [gsc_words])
            gsc_msg = gr.Markdown()

            gr.Markdown("---\n**Add via microphone (custom keyword)**")
            with gr.Row():
                mic_name = gr.Textbox(label="Keyword name", placeholder="e.g. hello")
                mic_audio = gr.Audio(label="Record ~1 s", sources=["microphone", "upload"], type="numpy")
                mic_btn = gr.Button("Add Sample")
            mic_msg = gr.Markdown()

            gr.Markdown("---\n**Profiles** (save/load full enrollment to JSON)")
            with gr.Row():
                profile_name = gr.Textbox(label="Profile name", value="default")
                save_btn = gr.Button("Save profile")
                load_btn = gr.Button("Load profile")
                refresh_btn = gr.Button("Refresh list")
            existing_profiles = gr.Markdown(value=", ".join(list_profiles()) or "_(no profiles yet)_")
            profile_msg = gr.Markdown()

            gr.Markdown("---")
            with gr.Row():
                clear_btn = gr.Button("Clear all", variant="stop")
                enrolled = gr.Markdown(value=status_md())

            gsc_btn.click(enroll_gsc, [gsc_words, gsc_k], [gsc_msg, enrolled])
            mic_btn.click(enroll_mic, [mic_name, mic_audio], [mic_msg, enrolled])
            clear_btn.click(clear_all, [], [gsc_msg, enrolled])
            save_btn.click(save_profile, [profile_name], [profile_msg])
            load_btn.click(load_profile, [profile_name], [profile_msg, enrolled])
            refresh_btn.click(lambda: ", ".join(list_profiles()) or "_(no profiles yet)_",
                              [], [existing_profiles])

        with gr.Tab("2. Detect (single)"):
            gr.Markdown("Upload or record ~1 s of audio. Compare to enrolled prototypes.")
            with gr.Row():
                with gr.Column(scale=1):
                    det_audio = gr.Audio(label="Audio", sources=["upload", "microphone"], type="numpy")
                    det_th = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.6, step=0.01,
                        label="Threshold (lower=stricter for L2; higher=stricter for Prob)",
                    )
                    det_scoring = gr.Radio(["L2", "Probability"], value="L2",
                                           label="Scoring (L2 = proposed, Probability = Rusci baseline)")
                    det_denoise = gr.Checkbox(label="Apply denoiser (EXT-1)", value=False)
                    det_btn = gr.Button("Detect", variant="primary", size="lg")
                with gr.Column(scale=2):
                    det_result = gr.Markdown()
                    det_plot = gr.Plot()
            det_btn.click(detect_single,
                          [det_audio, det_th, det_denoise, det_scoring],
                          [det_result, det_plot])

        with gr.Tab("3. Detect (long file)"):
            gr.Markdown(
                "Upload a longer audio file (5-30 s). The system segments speech and "
                "predicts one keyword per word region. **Min duration** is adjustable from "
                "**80 ms to 5000 ms**. If the slider still caps at 1000, **restart** "
                "`python demo_web.py` (stale Gradio UI)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    long_audio = gr.Audio(label="Long audio", sources=["upload", "microphone"], type="numpy")
                    long_th = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.85, step=0.01, label="Threshold",
                    )
                    long_scoring = gr.Radio(["L2", "Probability"], value="L2", label="Scoring method")
                    long_seg = gr.Radio(["Energy", "Silero VAD"], value="Energy",
                                         label="Segmentation method")
                    long_denoise = gr.Checkbox(label="Apply denoiser before processing", value=False)
                    long_mindur = gr.Slider(
                        minimum=80,
                        maximum=5000,
                        value=200,
                        step=20,
                        label="Min keyword duration (ms)",
                    )
                    long_expected = gr.Textbox(label="Expected words for accuracy (optional)",
                                                placeholder="yes,no,stop,...")
                    long_btn = gr.Button("Detect Words", variant="primary", size="lg")
                with gr.Column(scale=2):
                    long_result = gr.Markdown()
                    long_plot = gr.Plot()
            long_btn.click(detect_long,
                            [long_audio, long_th, long_seg, long_denoise, long_scoring,
                             long_mindur, long_expected],
                            [long_result, long_plot])

        with gr.Tab("4. Open-set test"):
            gr.Markdown(
                "Verify that **non-enrolled words get rejected**. "
                "Enter GSC words that are NOT in your enrolled set; the system should classify them as `unknown`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    os_words = gr.Textbox(label="Unknown GSC words to test",
                                           value="cat,bed,house,wow,sheila")
                    os_k = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Test samples per word (use 5 to match k-shot enrollment)",
                    )
                    os_th = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.85, step=0.01, label="Threshold",
                    )
                    os_scoring = gr.Radio(["L2", "Probability"], value="L2", label="Scoring")
                    os_denoise = gr.Checkbox(label="Apply denoiser", value=False)
                    os_btn = gr.Button("Run open-set test", variant="primary")
                with gr.Column(scale=2):
                    os_result = gr.Markdown()
                    os_plot = gr.Plot()
            os_btn.click(open_set_test,
                          [os_words, os_k, os_th, os_denoise, os_scoring],
                          [os_result, os_plot])

        with gr.Tab("5. Model + Results"):
            info_md = gr.Markdown(value=model_info_md())
            refresh_info = gr.Button("Refresh info")
            refresh_info.click(model_info_md, [], [info_md])

    return app


def _choose_server_port() -> int:
    """Bind to 127.0.0.1: first free in 7860-7879, or ``GRADIO_SERVER_PORT`` if set."""
    env = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    if env.isdigit():
        return int(env)
    for port in range(7860, 7880):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    msg = (
        "No free port 7860-7879. Stop the other Python/Gradio demo "
        "(Task Manager: end process on that port) or set GRADIO_SERVER_PORT=7880"
    )
    raise RuntimeError(msg)


if __name__ == "__main__":
    print("=" * 60)
    print("  Few-Shot KWS - Full Web Demo")
    print("=" * 60)
    init()
    print("\n  Starting web server...")
    app = build_ui()
    _port = _choose_server_port()
    print(f"  Local URL: http://127.0.0.1:{_port}")
    app.launch(server_name="127.0.0.1", server_port=_port, show_error=True)
