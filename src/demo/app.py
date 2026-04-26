"""Gradio demo for Few-Shot Open-Set Keyword Spotting.

Four tabs:
  1. Offline Detection  – upload/record audio -> detect keyword
  2. Enrollment         – record 3-5 samples to register new keywords (few-shot)
  3. Streaming + VAD    – upload long audio -> sliding window + VAD detection
  4. Settings           – threshold, denoising toggle, speaker gate, model info
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

# ─────────────────────────── Globals ────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pt"

encoder: DSCNN | None = None
mfcc_extractor: MFCCExtractor | None = None

enrolled_prototypes: dict[str, torch.Tensor] = {}
enrolled_samples: dict[str, list[torch.Tensor]] = {}

DEFAULT_THRESHOLD = 0.8

# Optional components (loaded lazily)
_denoiser = None
_speaker_gate = None
_vad = None
_denoiser_enabled = False
_speaker_gate_enabled = False


# ─────────────────────── Model Loading ──────────────────────────

def load_model():
    """Load DSCNN encoder from checkpoint."""
    global encoder, mfcc_extractor

    mfcc_extractor = MFCCExtractor(
        n_mfcc=40, num_features=10, sample_rate=SAMPLE_RATE,
    )
    encoder = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["model_state_dict"])
        elif "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if "loss" in checkpoint:
                print(f"   Loss:  {checkpoint['loss']:.6f}")
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        print("   Model will use random weights (for testing UI only)")

    encoder = encoder.to(DEVICE)
    encoder.eval()
    print(f"   Device: {DEVICE}")
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()):,}")


def _get_denoiser():
    """Lazy-load denoiser."""
    global _denoiser
    if _denoiser is None:
        try:
            from src.enhancements.denoiser import Denoiser
            _denoiser = Denoiser(backend="spectral_gate", enabled=True)
        except ImportError:
            _denoiser = None
    return _denoiser


def _get_speaker_gate():
    """Lazy-load speaker gate."""
    global _speaker_gate
    if _speaker_gate is None:
        try:
            from src.enhancements.speaker_verify import SpeakerGate
            _speaker_gate = SpeakerGate(enabled=False, device=DEVICE)
        except ImportError:
            _speaker_gate = None
    return _speaker_gate


def _get_vad():
    """Lazy-load Silero VAD."""
    global _vad
    if _vad is None:
        try:
            from src.streaming.vad_engine import SileroVAD
            _vad = SileroVAD(threshold=0.5, device=DEVICE)
        except Exception:
            _vad = None
    return _vad


# ─────────────────── Audio Preprocessing ────────────────────────

def preprocess_audio(audio_input, apply_denoise: bool = False) -> torch.Tensor | None:
    """Convert Gradio audio input to (1, 16000) waveform tensor."""
    if audio_input is None:
        return None

    sr, audio_np = audio_input

    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    elif audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    waveform = torch.from_numpy(audio_np).unsqueeze(0)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    if apply_denoise and _denoiser_enabled:
        denoiser = _get_denoiser()
        if denoiser is not None:
            waveform = denoiser.denoise(waveform)

    if waveform.shape[-1] < SAMPLE_RATE:
        pad = SAMPLE_RATE - waveform.shape[-1]
        waveform = F.pad(waveform, (0, pad))
    else:
        waveform = waveform[..., :SAMPLE_RATE]

    return waveform


def preprocess_audio_long(audio_input, apply_denoise: bool = False) -> torch.Tensor | None:
    """Convert Gradio audio to full-length (1, T) waveform (no trim to 1s)."""
    if audio_input is None:
        return None

    sr, audio_np = audio_input

    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    elif audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    waveform = torch.from_numpy(audio_np).unsqueeze(0)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    if apply_denoise and _denoiser_enabled:
        denoiser = _get_denoiser()
        if denoiser is not None:
            waveform = denoiser.denoise(waveform)

    return waveform


def extract_embedding(waveform: torch.Tensor) -> torch.Tensor:
    """Extract L2-normalized embedding from waveform."""
    mfcc = mfcc_extractor.extract(waveform)
    mfcc = mfcc.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = encoder(mfcc)
        embedding = F.normalize(embedding, p=2, dim=-1)

    return embedding.squeeze(0).cpu()


# ─────────────────── Visualization ──────────────────────────────

def plot_mfcc(waveform: torch.Tensor) -> plt.Figure:
    """Plot MFCC spectrogram."""
    mfcc = mfcc_extractor.extract(waveform)
    mfcc_np = mfcc.squeeze(0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    im = ax.imshow(mfcc_np.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MFCC Coefficient")
    ax.set_title("MFCC Features (10 coefficients x 47 frames)")
    fig.colorbar(im, ax=ax, label="Value")
    fig.tight_layout()
    return fig


def plot_distances(distances: dict[str, float], threshold: float) -> plt.Figure:
    """Bar chart of L2 distances to all prototypes."""
    if not distances:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No keywords enrolled yet",
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.tight_layout()
        return fig

    labels = list(distances.keys())
    dists = list(distances.values())

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5 + 1)))
    colors = ["#4CAF50" if d <= threshold else "#f44336" for d in dists]
    bars = ax.barh(labels, dists, color=colors, edgecolor="white", height=0.6)
    ax.axvline(x=threshold, color="#FF9800", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("L2 Distance")
    ax.set_title("Distance to Enrolled Keywords")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    for bar, dist in zip(bars, dists):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{dist:.4f}", va="center", fontsize=10)
    fig.tight_layout()
    return fig


def plot_det_enrolled() -> plt.Figure | None:
    """Plot DET curve for currently enrolled keywords (if enough data)."""
    if len(enrolled_samples) < 1 or not any(len(v) >= 3 for v in enrolled_samples.values()):
        return None

    from src.evaluation.metrics import compute_det_curve
    y_true_all = []
    scores_all = []

    for label, samples in enrolled_samples.items():
        if len(samples) < 3:
            continue
        proto = enrolled_prototypes[label]
        for emb in samples:
            d = torch.dist(emb, proto, p=2).item()
            y_true_all.append(1)
            scores_all.append(-d)
        for other_label, other_proto in enrolled_prototypes.items():
            if other_label != label:
                d = torch.dist(samples[0], other_proto, p=2).item()
                y_true_all.append(0)
                scores_all.append(-d)

    if len(set(y_true_all)) < 2:
        return None

    y_true = np.array(y_true_all)
    scores = np.array(scores_all)
    far, frr = compute_det_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    mask = (far > 0) & (frr > 0)
    if mask.any():
        ax.plot(far[mask] * 100, frr[mask] * 100, "b-", linewidth=2, label="Enrolled KWs")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curve (Enrolled Keywords)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


# ──────────────────── Tab 1: Offline Detection ──────────────────

def detect_keyword(audio_input, threshold: float, denoise: bool):
    """Detect keyword from uploaded/recorded audio."""
    global _denoiser_enabled
    _denoiser_enabled = denoise

    waveform = preprocess_audio(audio_input, apply_denoise=denoise)
    if waveform is None:
        return "No audio provided", None, None

    if not enrolled_prototypes:
        mfcc_fig = plot_mfcc(waveform)
        return "No keywords enrolled. Go to Enrollment tab first.", mfcc_fig, None

    query_emb = extract_embedding(waveform)

    # Speaker gate check
    speaker_ok = True
    speaker_sim = 1.0
    if _speaker_gate_enabled:
        gate = _get_speaker_gate()
        if gate is not None:
            speaker_ok, speaker_sim = gate.check(waveform)

    distances = {}
    for label, proto in enrolled_prototypes.items():
        dist = torch.dist(query_emb, proto, p=2).item()
        distances[label] = dist

    closest_label = min(distances, key=distances.get)
    closest_dist = distances[closest_label]

    if closest_dist <= threshold and speaker_ok:
        result = f"### Keyword Detected: **{closest_label.upper()}**\n\n"
        result += f"Distance: `{closest_dist:.4f}` (threshold: `{threshold:.2f}`)"
        if _speaker_gate_enabled:
            result += f"\nSpeaker similarity: `{speaker_sim:.3f}`"
    elif not speaker_ok:
        result = f"### REJECTED (speaker mismatch)\n\n"
        result += f"Closest: `{closest_label}` at distance `{closest_dist:.4f}`\n"
        result += f"Speaker similarity: `{speaker_sim:.3f}` (below threshold)"
    else:
        result = f"### REJECTED (unknown)\n\n"
        result += f"Closest: `{closest_label}` at distance `{closest_dist:.4f}`\n"
        result += f"Threshold: `{threshold:.2f}`"

    if denoise:
        result += "\n\n*Denoising: ON*"

    mfcc_fig = plot_mfcc(waveform)
    dist_fig = plot_distances(distances, threshold)
    return result, mfcc_fig, dist_fig


# ──────────────────── Tab 2: Enrollment ─────────────────────────

def enroll_sample(keyword_name: str, audio_input):
    """Add one audio sample for a keyword."""
    if not keyword_name or not keyword_name.strip():
        return "Please enter a keyword name.", get_enrollment_status(), None

    keyword = keyword_name.strip().lower()
    waveform = preprocess_audio(audio_input)
    if waveform is None:
        return "No audio provided.", get_enrollment_status(), None

    emb = extract_embedding(waveform)

    if keyword not in enrolled_samples:
        enrolled_samples[keyword] = []
    enrolled_samples[keyword].append(emb)

    stacked = torch.stack(enrolled_samples[keyword])
    prototype = F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
    enrolled_prototypes[keyword] = prototype

    n = len(enrolled_samples[keyword])
    msg = f"Added sample #{n} for **'{keyword}'**"
    if n < 3:
        msg += f"\n\nNeed at least 3 samples for reliable detection. ({n}/3)"
    else:
        msg += f"\n\nPrototype ready! ({n} samples)"

    det_fig = plot_det_enrolled()
    return msg, get_enrollment_status(), det_fig


def remove_keyword(keyword_name: str):
    """Remove all samples for a keyword."""
    keyword = keyword_name.strip().lower()
    if keyword in enrolled_samples:
        del enrolled_samples[keyword]
    if keyword in enrolled_prototypes:
        del enrolled_prototypes[keyword]
    return f"Removed keyword '{keyword}'", get_enrollment_status(), None


def clear_all_keywords():
    """Remove all enrolled keywords."""
    enrolled_samples.clear()
    enrolled_prototypes.clear()
    return "All keywords cleared.", get_enrollment_status(), None


def get_enrollment_status() -> str:
    """Get markdown table of enrolled keywords."""
    if not enrolled_prototypes:
        return "No keywords enrolled yet. Record samples above to get started."

    lines = ["| Keyword | Samples | Status |",
             "|---------|---------|--------|"]
    for keyword in sorted(enrolled_prototypes.keys()):
        n = len(enrolled_samples.get(keyword, []))
        status = "Ready" if n >= 3 else f"Need {3-n} more"
        lines.append(f"| {keyword} | {n} | {status} |")

    if len(enrolled_prototypes) >= 2:
        lines.append("")
        lines.append("### Prototype Distances")
        lines.append("| Pair | L2 Distance |")
        lines.append("|------|-------------|")
        keywords = sorted(enrolled_prototypes.keys())
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                d = torch.dist(enrolled_prototypes[keywords[i]],
                               enrolled_prototypes[keywords[j]], p=2).item()
                lines.append(f"| {keywords[i]} - {keywords[j]} | {d:.4f} |")

    return "\n".join(lines)


# ──────────────────── Tab 3: Streaming + VAD ────────────────────

def streaming_detect(audio_input, threshold: float, window_ms: int, stride_ms: int,
                     use_vad: bool, denoise: bool):
    """Streaming detection with optional VAD on a longer audio file."""
    global _denoiser_enabled
    _denoiser_enabled = denoise

    waveform = preprocess_audio_long(audio_input, apply_denoise=denoise)
    if waveform is None:
        return "No audio provided", None

    if not enrolled_prototypes:
        return "No keywords enrolled. Go to Enrollment tab first.", None

    total_samples = waveform.shape[-1]
    window_size = int(SAMPLE_RATE * window_ms / 1000)
    stride = int(SAMPLE_RATE * stride_ms / 1000)

    vad = _get_vad() if use_vad else None
    if vad is not None:
        vad.reset_states()

    results = []
    pos = 0
    vad_skipped = 0

    while pos + window_size <= total_samples:
        segment = waveform[..., pos : pos + window_size]
        t_start = pos / SAMPLE_RATE
        t_end = (pos + window_size) / SAMPLE_RATE

        speech_prob = 1.0
        is_speech = True
        if vad is not None:
            seg_for_vad = segment.squeeze(0) if segment.dim() == 2 else segment
            if seg_for_vad.shape[-1] < SAMPLE_RATE:
                seg_for_vad = F.pad(seg_for_vad, (0, SAMPLE_RATE - seg_for_vad.shape[-1]))
            is_speech, speech_prob = vad.is_speech(seg_for_vad)

        if not is_speech:
            vad_skipped += 1
            results.append({
                "t_start": t_start, "t_end": t_end,
                "label": "(silence)", "dist": float("inf"),
                "detected": False, "speech_prob": speech_prob,
            })
            pos += stride
            continue

        if segment.shape[-1] < SAMPLE_RATE:
            segment = F.pad(segment, (0, SAMPLE_RATE - segment.shape[-1]))

        emb = extract_embedding(segment)
        min_dist = float("inf")
        min_label = "unknown"
        for label, proto in enrolled_prototypes.items():
            d = torch.dist(emb, proto, p=2).item()
            if d < min_dist:
                min_dist = d
                min_label = label

        detected = min_dist <= threshold
        results.append({
            "t_start": t_start, "t_end": t_end,
            "label": min_label if detected else "---",
            "dist": min_dist, "detected": detected,
            "speech_prob": speech_prob,
        })
        pos += stride

    # Build results text
    lines = ["### Streaming Results\n"]
    lines.append(f"Audio: {total_samples/SAMPLE_RATE:.1f}s | "
                 f"Window: {window_ms}ms | Stride: {stride_ms}ms | "
                 f"Segments: {len(results)}")
    if use_vad:
        lines.append(f"\nVAD: ON | Skipped {vad_skipped}/{len(results)} silent segments")
    if denoise:
        lines.append("Denoising: ON")
    lines.append("")
    lines.append("| Time | Keyword | Distance | Speech Prob | Status |")
    lines.append("|------|---------|----------|-------------|--------|")
    for r in results:
        status = "DETECTED" if r["detected"] else ("silence" if r["label"] == "(silence)" else "---")
        lines.append(
            f"| {r['t_start']:.1f}-{r['t_end']:.1f}s | {r['label']} | "
            f"{r['dist']:.4f} | {r['speech_prob']:.2f} | {status} |"
        )

    # Timeline plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), height_ratios=[2, 1])

    for r in results:
        if r["detected"]:
            color = "#4CAF50"
        elif r["label"] == "(silence)":
            color = "#e0e0e0"
        else:
            color = "#FFCDD2"
        axes[0].barh(0, r["t_end"] - r["t_start"], left=r["t_start"],
                     height=0.5, color=color, edgecolor="white")
        if r["detected"]:
            axes[0].text((r["t_start"] + r["t_end"]) / 2, 0,
                         r["label"], ha="center", va="center",
                         fontsize=8, fontweight="bold", color="white")
    axes[0].set_yticks([])
    axes[0].set_title("Detection Timeline (green=detected, red=rejected, gray=silence)")
    axes[0].set_xlim(0, total_samples / SAMPLE_RATE)

    # Speech probability plot
    times = [(r["t_start"] + r["t_end"]) / 2 for r in results]
    probs = [r["speech_prob"] for r in results]
    axes[1].fill_between(times, probs, alpha=0.3, color="blue")
    axes[1].plot(times, probs, "b-", linewidth=1)
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="VAD threshold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Speech Prob")
    axes[1].set_xlim(0, total_samples / SAMPLE_RATE)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    return "\n".join(lines), fig


# ──────────────────── Tab 4: Settings ───────────────────────────

def get_model_info() -> str:
    """Get model information."""
    info = "### Model Information\n\n"
    info += f"| Property | Value |\n|---|---|\n"
    info += f"| Architecture | DSCNN-L |\n"
    info += f"| Parameters | {sum(p.numel() for p in encoder.parameters()):,} |\n"
    info += f"| Embedding dim | {encoder.embedding_dim} |\n"
    info += f"| Input shape | (1, 47, 10) |\n"
    info += f"| Device | {DEVICE} |\n"
    info += f"| Checkpoint | {CHECKPOINT_PATH.name} |\n"

    if CHECKPOINT_PATH.exists():
        size_mb = CHECKPOINT_PATH.stat().st_size / 1024 / 1024
        info += f"| Checkpoint size | {size_mb:.1f} MB |\n"

    info += "\n### Extensions\n\n"
    info += f"| Extension | Available |\n|---|---|\n"

    denoise_ok = False
    try:
        import noisereduce  # noqa: F401
        denoise_ok = True
    except ImportError:
        pass
    info += f"| EXT-1: Denoising | {'Yes' if denoise_ok else 'No (install noisereduce)'} |\n"

    vad_ok = False
    try:
        torch.hub.list("snakers4/silero-vad", trust_repo=True)
        vad_ok = True
    except Exception:
        pass
    info += f"| EXT-2: VAD (Silero) | {'Yes' if vad_ok else 'No'} |\n"

    sb_ok = False
    try:
        import speechbrain  # noqa: F401
        sb_ok = True
    except ImportError:
        pass
    info += f"| Speaker Verify (ECAPA) | {'Yes' if sb_ok else 'No (install speechbrain)'} |\n"

    return info


def toggle_denoiser(enabled: bool):
    global _denoiser_enabled
    _denoiser_enabled = enabled
    return f"Denoising: {'ON' if enabled else 'OFF'}"


def toggle_speaker_gate(enabled: bool):
    global _speaker_gate_enabled
    _speaker_gate_enabled = enabled
    gate = _get_speaker_gate()
    if gate is not None:
        gate.set_enabled(enabled)
    return f"Speaker gate: {'ON' if enabled else 'OFF'}"


def enroll_speaker(audio_input):
    """Enroll speaker sample for speaker gate."""
    waveform = preprocess_audio(audio_input)
    if waveform is None:
        return "No audio provided."
    gate = _get_speaker_gate()
    if gate is None:
        return "Speaker verification not available. Install speechbrain."
    n = gate.enroll(waveform)
    return f"Speaker enrolled ({n} samples). Gate will verify speaker identity."


def clear_speaker():
    """Clear enrolled speaker."""
    gate = _get_speaker_gate()
    if gate is not None:
        gate.clear()
    return "Speaker enrollment cleared."


# ──────────────────── Gradio UI ─────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio app with 4 tabs."""

    with gr.Blocks(
        title="Few-Shot KWS Demo",
    ) as app:

        gr.Markdown(
            """
            # Few-Shot Open-Set Keyword Spotting
            ### DSCNN-L Encoder + Prototypical Network + Open NCM Classifier

            Enroll custom keywords with just **3-5 audio samples**, then detect them in real-time.
            """
        )

        # ─── Tab 1: Offline Detection ───
        with gr.Tab("Offline Detection"):
            gr.Markdown("Upload or record a **1-second** audio clip to detect enrolled keywords.")

            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Audio Input",
                        sources=["upload", "microphone"],
                        type="numpy",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.1, maximum=2.0, value=DEFAULT_THRESHOLD,
                        step=0.05, label="Detection Threshold (L2 distance)"
                    )
                    denoise_check = gr.Checkbox(label="Enable Denoising (EXT-1)", value=False)
                    detect_btn = gr.Button("Detect Keyword", variant="primary", size="lg")

                with gr.Column(scale=2):
                    result_text = gr.Markdown(label="Result")
                    mfcc_plot = gr.Plot(label="MFCC Spectrogram")
                    dist_plot = gr.Plot(label="Distances to Prototypes")

            detect_btn.click(
                fn=detect_keyword,
                inputs=[audio_input, threshold_slider, denoise_check],
                outputs=[result_text, mfcc_plot, dist_plot],
            )

        # ─── Tab 2: Enrollment ───
        with gr.Tab("Enrollment"):
            gr.Markdown(
                "Record **3-5 samples** per keyword to enroll it. "
                "Say the keyword clearly, ~1 second per sample."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    keyword_input = gr.Textbox(
                        label="Keyword Name",
                        placeholder="e.g. yes, no, stop, hey_jarvis...",
                    )
                    enroll_audio = gr.Audio(
                        label="Record Sample",
                        sources=["upload", "microphone"],
                        type="numpy",
                    )
                    enroll_btn = gr.Button("Add Sample", variant="primary")

                    gr.Markdown("---")
                    remove_input = gr.Textbox(label="Remove Keyword", placeholder="keyword name")
                    with gr.Row():
                        remove_btn = gr.Button("Remove", variant="secondary")
                        clear_btn = gr.Button("Clear All", variant="stop")

                with gr.Column(scale=2):
                    enroll_msg = gr.Markdown(label="Status")
                    enroll_status = gr.Markdown(
                        value=get_enrollment_status,
                        label="Enrolled Keywords",
                    )
                    det_plot = gr.Plot(label="DET Curve (Enrolled)")

            enroll_btn.click(
                fn=enroll_sample,
                inputs=[keyword_input, enroll_audio],
                outputs=[enroll_msg, enroll_status, det_plot],
            )
            remove_btn.click(
                fn=remove_keyword,
                inputs=[remove_input],
                outputs=[enroll_msg, enroll_status, det_plot],
            )
            clear_btn.click(
                fn=clear_all_keywords,
                inputs=[],
                outputs=[enroll_msg, enroll_status, det_plot],
            )

        # ─── Tab 3: Streaming + VAD ───
        with gr.Tab("Streaming + VAD"):
            gr.Markdown(
                "Upload a **longer audio file** (5-30s) to test sliding-window detection "
                "with optional Voice Activity Detection (Silero VAD)."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    stream_audio = gr.Audio(
                        label="Long Audio File",
                        sources=["upload", "microphone"],
                        type="numpy",
                    )
                    stream_threshold = gr.Slider(
                        minimum=0.1, maximum=2.0, value=DEFAULT_THRESHOLD,
                        step=0.05, label="Threshold"
                    )
                    with gr.Row():
                        window_slider = gr.Slider(
                            minimum=500, maximum=2000, value=1000,
                            step=100, label="Window (ms)"
                        )
                        stride_slider = gr.Slider(
                            minimum=100, maximum=1000, value=500,
                            step=100, label="Stride (ms)"
                        )
                    vad_check = gr.Checkbox(label="Enable VAD (EXT-2)", value=True)
                    stream_denoise_check = gr.Checkbox(label="Enable Denoising (EXT-1)", value=False)
                    stream_btn = gr.Button("Run Streaming Detection", variant="primary")

                with gr.Column(scale=2):
                    stream_result = gr.Markdown()
                    stream_plot = gr.Plot(label="Timeline + VAD")

            stream_btn.click(
                fn=streaming_detect,
                inputs=[stream_audio, stream_threshold, window_slider, stride_slider,
                        vad_check, stream_denoise_check],
                outputs=[stream_result, stream_plot],
            )

        # ─── Tab 4: Settings ───
        with gr.Tab("Settings"):
            with gr.Row():
                with gr.Column():
                    model_info = gr.Markdown(value=get_model_info)

                with gr.Column():
                    gr.Markdown("### Extension Controls")

                    with gr.Group():
                        gr.Markdown("**EXT-1: Denoising**")
                        denoise_toggle = gr.Checkbox(label="Enable denoising globally", value=False)
                        denoise_status = gr.Markdown("Denoising: OFF")
                        denoise_toggle.change(
                            fn=toggle_denoiser,
                            inputs=[denoise_toggle],
                            outputs=[denoise_status],
                        )

                    with gr.Group():
                        gr.Markdown("**Speaker Verification (Optional)**")
                        spk_toggle = gr.Checkbox(label="Enable speaker gate", value=False)
                        spk_status = gr.Markdown("Speaker gate: OFF")
                        spk_audio = gr.Audio(
                            label="Enroll Speaker Voice",
                            sources=["upload", "microphone"],
                            type="numpy",
                        )
                        with gr.Row():
                            spk_enroll_btn = gr.Button("Enroll Speaker", variant="primary")
                            spk_clear_btn = gr.Button("Clear Speaker", variant="secondary")
                        spk_result = gr.Markdown()

                        spk_toggle.change(
                            fn=toggle_speaker_gate,
                            inputs=[spk_toggle],
                            outputs=[spk_status],
                        )
                        spk_enroll_btn.click(
                            fn=enroll_speaker,
                            inputs=[spk_audio],
                            outputs=[spk_result],
                        )
                        spk_clear_btn.click(
                            fn=clear_speaker,
                            inputs=[],
                            outputs=[spk_result],
                        )

    return app


# ─────────────────────── Main ───────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Few-Shot Open-Set Keyword Spotting - Demo")
    print("=" * 60)

    load_model()

    print("\nStarting Gradio server...")
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
    )
