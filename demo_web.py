"""
Simple web demo: Enrollment first, then Detect.
Run: python -u demo_web.py
Open: http://127.0.0.1:7860
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.dscnn import DSCNN
from src.features.mfcc import MFCCExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
CKPT = Path("checkpoints/best.pt")

encoder = None
mfcc_ext = None
prototypes: dict[str, torch.Tensor] = {}
sample_count: dict[str, int] = {}


def init():
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


def to_wav_1s(audio_input) -> torch.Tensor | None:
    wav = to_wav(audio_input)
    if wav is None:
        return None
    if wav.shape[-1] < SR:
        wav = F.pad(wav, (0, SR - wav.shape[-1]))
    return wav[..., :SR]


def embed(wav_1s: torch.Tensor) -> torch.Tensor:
    mfcc = mfcc_ext.extract(wav_1s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(encoder(mfcc), p=2, dim=-1)
    return emb.squeeze(0).cpu()


def status_md():
    if not prototypes:
        return "No keywords enrolled yet."
    lines = []
    for w in prototypes:
        lines.append(f"**{w}** ({sample_count.get(w,0)} samples)")
    return "Enrolled: " + ", ".join(lines)


# ============ Enrollment ============

def enroll_gsc(words_text: str, k: int):
    words = [w.strip() for w in words_text.split(",") if w.strip()]
    if not words:
        return "Enter words separated by commas.", status_md()
    msgs = []
    for word in words:
        d = Path(f"data/gsc_v2/{word}")
        if not d.exists():
            msgs.append(f"{word}: not found")
            continue
        files = sorted(d.glob("*.wav"))[:k]
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
        msgs.append(f"{word}: OK ({len(files)} samples)")
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
    return f"Added '{keyword}' ({sample_count[keyword]} total)", status_md()


def clear_all():
    prototypes.clear()
    sample_count.clear()
    return "Cleared.", status_md()


# ============ Detection ============

def detect_single(audio, threshold):
    wav = to_wav_1s(audio)
    if wav is None:
        return "No audio.", None
    if not prototypes:
        return "Enroll keywords first! (Tab 1)", None

    e = embed(wav)
    dists = {w: torch.cdist(e.unsqueeze(0), p.unsqueeze(0)).item() for w, p in prototypes.items()}
    sd = sorted(dists.items(), key=lambda x: x[1])
    best_w, best_d = sd[0]
    pred = best_w if best_d <= threshold else "unknown"

    if pred == "unknown":
        txt = f"### UNKNOWN (rejected)\nNearest: {best_w} ({best_d:.4f} > {threshold})"
    else:
        txt = f"### {pred.upper()}\nDistance: {best_d:.4f}"

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(sd) * 0.4)))
    words = [w for w, _ in sd]
    ds = [d for _, d in sd]
    colors = ["#4CAF50" if d <= threshold else "#ef5350" for d in ds]
    ax.barh(words, ds, color=colors)
    ax.axvline(threshold, color="orange", linestyle="--", lw=2, label=f"threshold={threshold}")
    ax.set_xlabel("L2 Distance")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    return txt, fig


def _pad_or_center_1s(wav: torch.Tensor) -> torch.Tensor:
    """Match MFCCExtractor preprocessing: right-pad short clips, right-trim long clips."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    length = wav.shape[-1]
    if length < SR:
        return F.pad(wav, (0, SR - length))
    if length > SR:
        return wav[..., :SR]
    return wav


def _word_segments_by_energy(
    wav: torch.Tensor,
    min_duration_ms: int,
    merge_gap_ms: int = 350,
    pad_ms: int = 120,
) -> list[tuple[int, int]]:
    """Cut long audio into word-like regions using short-time energy."""
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
    active = []
    current_start = None
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

    # Spoken keywords can have a short high-energy core. Keep short cores, then
    # pad them to feed a stable 1-second model input.
    min_len = int(SR * min(120, min_duration_ms) / 1000)
    pad = int(SR * pad_ms / 1000)
    return [
        (max(0, start - pad), min(total, end + pad))
        for start, end in merged
        if min(total, end + pad) - max(0, start - pad) >= min_len
    ]


def _fallback_segments_by_window_energy(
    wav: torch.Tensor,
    window_ms: int,
    stride_ms: int,
) -> list[tuple[int, int]]:
    """Fallback grouping from active sliding windows when energy cutting is empty."""
    mono = wav.mean(dim=0) if wav.dim() == 2 else wav
    total = mono.shape[-1]
    win = int(SR * window_ms / 1000)
    stride = int(SR * stride_ms / 1000)
    if total <= 0 or win <= 0 or stride <= 0:
        return []

    rows = []
    pos = 0
    while pos < total:
        end = min(total, pos + win)
        seg = mono[pos:end]
        energy = float(seg.abs().mean().item())
        rows.append((pos, end, energy))
        if end == total:
            break
        pos += stride

    max_energy = max((energy for _, _, energy in rows), default=0.0)
    if max_energy <= 1e-6:
        return []
    threshold = max(0.0005, max_energy * 0.15)

    active = []
    for start, end, energy in rows:
        if energy >= threshold:
            if active and start - active[-1][1] <= stride:
                active[-1] = (active[-1][0], end)
            else:
                active.append((start, end))

    return active


def _score_segment(segment: torch.Tensor, threshold: float) -> dict:
    """Predict one label for a single word segment."""
    e = embed(_pad_or_center_1s(segment))
    dists = {
        w: torch.cdist(e.unsqueeze(0), p.unsqueeze(0)).item()
        for w, p in prototypes.items()
    }
    sd = sorted(dists.items(), key=lambda x: x[1])
    best_w, best_d = sd[0]
    pred = best_w if best_d <= threshold else "unknown"
    return {
        "pred": pred,
        "dist": best_d,
        "top3": ", ".join(f"{w}:{d:.3f}" for w, d in sd[:3]),
    }


def _score_window(segment: torch.Tensor, threshold: float) -> dict:
    """Predict one sliding window inside a word cluster."""
    e = embed(_pad_or_center_1s(segment))
    dists = {
        w: torch.cdist(e.unsqueeze(0), p.unsqueeze(0)).item()
        for w, p in prototypes.items()
    }
    sd = sorted(dists.items(), key=lambda x: x[1])
    best_w, best_d = sd[0]
    return {
        "raw_pred": best_w,
        "pred": best_w if best_d <= threshold else "unknown",
        "dist": best_d,
        "top3": ", ".join(f"{w}:{d:.3f}" for w, d in sd[:3]),
    }


_TTA_SHIFT_COUNT = 5


def _tta_views(segment: torch.Tensor) -> list[torch.Tensor]:
    """Generate _TTA_SHIFT_COUNT padded 1-second views of the segment."""
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
    views = []
    for frac in fractions:
        left = int(round(pad_total * frac))
        right = pad_total - left
        views.append(F.pad(segment, (left, right)))
    return views


def _score_cluster_tta(segment: torch.Tensor, threshold: float) -> dict:
    """TTA-based scoring: average embeddings from multiple shifts + vote."""
    views = _tta_views(segment)
    embeddings = [embed(view) for view in views]
    averaged_emb = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=-1)

    dists = {
        w: torch.cdist(averaged_emb.unsqueeze(0), p.unsqueeze(0)).item()
        for w, p in prototypes.items()
    }
    sd = sorted(dists.items(), key=lambda x: x[1])
    best_w, best_d = sd[0]

    per_view_preds = []
    for emb_view in embeddings:
        view_dists = {
            w: torch.cdist(emb_view.unsqueeze(0), p.unsqueeze(0)).item()
            for w, p in prototypes.items()
        }
        per_view_preds.append(min(view_dists, key=view_dists.get))

    vote_counts = Counter(per_view_preds)
    vote_text = ", ".join(f"{label}:{count}" for label, count in vote_counts.most_common())
    return {
        "pred": best_w if best_d <= threshold else "unknown",
        "dist": best_d,
        "top3": ", ".join(f"{w}:{d:.3f}" for w, d in sd[:3]),
        "n_views": len(views),
        "votes": vote_text,
    }


def _score_cluster_by_windows(
    wav: torch.Tensor,
    start: int,
    end: int,
    threshold: float,
    window_ms: int,
    stride_ms: int,
) -> dict:
    """Collapse many predictions inside one speech cluster into one word.

    Short clusters (shorter than the window) use TTA: average embeddings
    from 3 shift positions within a 1-second frame, then classify once.
    Long clusters use sliding-window voting.
    """
    win = int(SR * window_ms / 1000)
    stride = int(SR * stride_ms / 1000)
    cluster_len = end - start
    if cluster_len <= 0:
        return {"pred": "unknown", "dist": float("inf"), "top3": "-", "votes": "-", "n_windows": 0}

    if cluster_len <= win:
        scored = _score_cluster_tta(wav[..., start:end], threshold)
        return {
            "pred": scored["pred"],
            "dist": scored["dist"],
            "top3": scored["top3"],
            "votes": f"tta-{scored['n_views']} {scored['votes']}",
            "n_windows": scored["n_views"],
        }

    window_results = []
    pos = start
    while pos + win <= end:
        window_results.append(_score_window(wav[..., pos:pos + win], threshold))
        pos += stride
    if not window_results or pos < end:
        window_results.append(_score_window(wav[..., max(start, end - win):end], threshold))

    accepted = [r for r in window_results if r["pred"] != "unknown"]
    vote_pool = accepted if accepted else window_results
    vote_counts = Counter(str(r["pred"]) for r in vote_pool)
    best_count = max(vote_counts.values())
    candidate_labels = {label for label, count in vote_counts.items() if count == best_count}
    best = min(
        (r for r in vote_pool if r["pred"] in candidate_labels),
        key=lambda r: float(r["dist"]),
    )
    votes = ", ".join(f"{label}:{count}" for label, count in vote_counts.most_common())
    return {
        "pred": str(best["pred"]),
        "dist": float(best["dist"]),
        "top3": best["top3"],
        "votes": votes,
        "n_windows": len(window_results),
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


def detect_long(audio, threshold, window_ms, stride_ms, min_duration_ms=250, expected_text=""):
    wav = to_wav(audio)
    if wav is None:
        return "No audio.", None
    if not prototypes:
        return "Enroll keywords first! (Tab 1)", None

    total = wav.shape[-1]
    segments = _word_segments_by_energy(wav, min_duration_ms=min_duration_ms)
    if not segments:
        segments = _fallback_segments_by_window_energy(wav, window_ms, stride_ms)

    merged = []
    for start, end in segments:
        scored = _score_cluster_by_windows(
            wav,
            start,
            end,
            threshold,
            window_ms,
            stride_ms,
        )
        merged.append({
            "t0": start / SR,
            "t1": end / SR,
            "pred": scored["pred"],
            "dist": scored["dist"],
            "top3": scored["top3"],
            "votes": scored["votes"],
            "n_windows": scored["n_windows"],
        })

    preds = [m["pred"] for m in merged if m["pred"] != "unknown"]
    counts = Counter(preds)
    lines = [
        f"**{total/SR:.1f}s** audio, cut into **{len(merged)} word segments** "
        f"(window={window_ms}ms, stride={stride_ms}ms kept for reference)\n"
    ]
    lines.append(f"### Final word sequence ({len(preds)} keywords)")
    lines.append(" -> ".join(f"**{p}**" for p in preds) if preds else "(no keyword detected)")
    if counts:
        vote_text = ", ".join(f"{word}:{count}" for word, count in counts.most_common())
        lines.append(f"\nVotes: {vote_text}")
    if expected_text:
        lines.append(_accuracy_line([m["pred"] for m in merged], expected_text))

    lines.append("\n### Detail (clustered windows -> one word)")
    lines.append("| # | Time | Windows | Final word | Distance | Votes | Top-3 |")
    lines.append("|---|------|---------|------------|----------|-------|-------|")
    for idx, m in enumerate(merged, start=1):
        if m["pred"] == "unknown":
            lines.append(
                f"| {idx} | {m['t0']:.1f}-{m['t1']:.1f}s | {m['n_windows']} | "
                f"unknown | {m['dist']:.4f} | {m['votes']} | {m['top3']} |"
            )
        else:
            lines.append(
                f"| {idx} | {m['t0']:.1f}-{m['t1']:.1f}s | {m['n_windows']} | "
                f"**{m['pred']}** | {m['dist']:.4f} | {m['votes']} | {m['top3']} |"
            )

    fig, ax = plt.subplots(figsize=(12, 3))
    for m in merged:
        if m["pred"] == "unknown":
            color, label = "#ef5350", "?"
        else:
            color, label = "#4CAF50", m["pred"]
        ax.barh(0, m["t1"] - m["t0"], left=m["t0"], height=0.6,
                color=color, edgecolor="white", linewidth=1)
        if label:
            ax.text((m["t0"] + m["t1"]) / 2, 0, label,
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white" if color != "#e0e0e0" else "black")
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Timeline: one block = one detected word segment")
    ax.set_xlim(0, total / SR)
    fig.tight_layout()
    return "\n".join(lines), fig


# ============ UI ============

with gr.Blocks(title="Few-Shot KWS") as app:
    gr.Markdown("# Few-Shot Open-Set Keyword Spotting Demo")

    # --- Tab 1: Enrollment (FIRST) ---
    with gr.Tab("1. Enrollment"):
        gr.Markdown("### Enroll keywords before detection.\n"
                     "Option A: auto-enroll from GSC dataset. Option B: record with mic.")

        gr.Markdown("**Option A: From GSC dataset**")
        with gr.Row():
            gsc_words = gr.Textbox(value="yes,no,stop,go,up,down,left,right,on,off",
                                   label="Words (comma-separated)")
            gsc_k = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="Samples per word (more = better)")
            gsc_btn = gr.Button("Enroll from GSC", variant="primary")
        gsc_msg = gr.Markdown()

        gr.Markdown("---\n**Option B: Record with microphone**")
        with gr.Row():
            mic_name = gr.Textbox(label="Keyword name", placeholder="e.g. hello")
            mic_audio = gr.Audio(label="Record 1 sample (~1s)", sources=["microphone", "upload"], type="numpy")
            mic_btn = gr.Button("Add Sample")
        mic_msg = gr.Markdown()

        gr.Markdown("---")
        with gr.Row():
            clear_btn = gr.Button("Clear All", variant="stop")
            enrolled = gr.Markdown(value="No keywords enrolled yet.")

        gsc_btn.click(enroll_gsc, [gsc_words, gsc_k], [gsc_msg, enrolled])
        mic_btn.click(enroll_mic, [mic_name, mic_audio], [mic_msg, enrolled])
        clear_btn.click(clear_all, [], [gsc_msg, enrolled])

    # --- Tab 2: Detect single ---
    with gr.Tab("2. Detect (single)"):
        gr.Markdown("Upload or record **~1 second** audio. Compare to enrolled prototypes.")
        with gr.Row():
            with gr.Column(scale=1):
                det_audio = gr.Audio(label="Audio", sources=["upload", "microphone"], type="numpy")
                det_th = gr.Slider(0.0, 2.0, value=0.6, step=0.01, label="Threshold (lower = stricter)")
                det_btn = gr.Button("Detect", variant="primary", size="lg")
            with gr.Column(scale=2):
                det_result = gr.Markdown()
                det_plot = gr.Plot()
        det_btn.click(detect_single, [det_audio, det_th], [det_result, det_plot])

    # --- Tab 3: Detect long file ---
    with gr.Tab("3. Detect (long file)"):
        gr.Markdown("Upload a **longer audio file** (e.g. 5-30s). "
                     "System cuts speech into word segments and returns **one prediction per word**.")
        with gr.Row():
            with gr.Column(scale=1):
                long_audio = gr.Audio(label="Long audio file", sources=["upload", "microphone"], type="numpy")
                long_th = gr.Slider(0.0, 2.0, value=0.6, step=0.01, label="Threshold (lower = stricter)")
                with gr.Row():
                    long_win = gr.Slider(500, 2000, value=1000, step=100, label="Window (ms)")
                    long_stride = gr.Slider(100, 1000, value=250, step=50, label="Stride (ms)")
                long_mindur = gr.Slider(100, 1000, value=400, step=50,
                                        label="Min keyword duration (ms) - filter false positives")
                long_expected = gr.Textbox(
                    label="Expected words for accuracy (optional)",
                    placeholder="yes,no,stop,go,yes",
                )
                long_btn = gr.Button("Detect Words", variant="primary", size="lg")
            with gr.Column(scale=2):
                long_result = gr.Markdown()
                long_plot = gr.Plot()
        long_btn.click(detect_long, [long_audio, long_th, long_win, long_stride, long_mindur, long_expected],
                       [long_result, long_plot])


if __name__ == "__main__":
    print("=" * 50)
    print("  Few-Shot KWS - Simple Web Demo")
    print("=" * 50)
    init()
    print("\n  Starting web server...")
    app.launch(server_name="127.0.0.1", show_error=True)
