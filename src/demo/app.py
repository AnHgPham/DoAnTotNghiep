"""Gradio demo for Few-Shot Open-Set Keyword Spotting.

Three tabs:
  1. Offline Detection  – upload/record audio → detect keyword
  2. Enrollment         – record 3-5 samples to register new keywords (few-shot)
  3. Settings            – threshold slider, model info
"""

import sys
import os
from pathlib import Path

# Add project root to path
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

# Model & feature extractor (loaded once)
encoder: DSCNN | None = None
mfcc_extractor: MFCCExtractor | None = None

# Enrollment state
enrolled_prototypes: dict[str, torch.Tensor] = {}   # label -> (embedding_dim,)
enrolled_samples: dict[str, list[torch.Tensor]] = {}  # label -> list of embeddings

# Default threshold
DEFAULT_THRESHOLD = 0.8


# ─────────────────────── Model Loading ──────────────────────────

def load_model():
    """Load DSCNN encoder from checkpoint."""
    global encoder, mfcc_extractor

    mfcc_extractor = MFCCExtractor(
        n_mfcc=40,
        num_features=10,
        sample_rate=SAMPLE_RATE,
    )

    encoder = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["model_state_dict"])
        elif "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        else:
            # Try loading directly as state dict
            encoder.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")
        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
            if "loss" in checkpoint:
                print(f"   Loss:  {checkpoint['loss']:.6f}")
    else:
        print(f"⚠️  Checkpoint not found at {CHECKPOINT_PATH}")
        print("   Model will use random weights (for testing UI only)")

    encoder = encoder.to(DEVICE)
    encoder.eval()
    print(f"   Device: {DEVICE}")
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()):,}")


# ─────────────────── Audio Preprocessing ────────────────────────

def preprocess_audio(audio_input) -> torch.Tensor | None:
    """Convert Gradio audio input to (1, 16000) waveform tensor.

    Args:
        audio_input: tuple (sample_rate, numpy_array) from Gradio.

    Returns:
        (1, 16000) tensor or None if invalid.
    """
    if audio_input is None:
        return None

    sr, audio_np = audio_input

    # Convert to float32
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    elif audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    # Convert to mono
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Pad/trim to 1 second
    if waveform.shape[-1] < SAMPLE_RATE:
        pad = SAMPLE_RATE - waveform.shape[-1]
        waveform = F.pad(waveform, (0, pad))
    else:
        waveform = waveform[..., :SAMPLE_RATE]

    return waveform


def extract_embedding(waveform: torch.Tensor) -> torch.Tensor:
    """Extract L2-normalized embedding from waveform.

    Args:
        waveform: (1, 16000) tensor.

    Returns:
        (embedding_dim,) normalized embedding.
    """
    mfcc = mfcc_extractor.extract(waveform)  # (1, 47, 10)
    mfcc = mfcc.unsqueeze(0).to(DEVICE)      # (1, 1, 47, 10)

    with torch.no_grad():
        embedding = encoder(mfcc)             # (1, 276)
        embedding = F.normalize(embedding, p=2, dim=-1)

    return embedding.squeeze(0).cpu()         # (276,)


# ─────────────────── MFCC Visualization ─────────────────────────

def plot_mfcc(waveform: torch.Tensor) -> plt.Figure:
    """Plot MFCC spectrogram."""
    mfcc = mfcc_extractor.extract(waveform)  # (1, 47, 10)
    mfcc_np = mfcc.squeeze(0).numpy()        # (47, 10)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    im = ax.imshow(mfcc_np.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MFCC Coefficient")
    ax.set_title("MFCC Features (10 coefficients × 47 frames)")
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

    # Add distance values on bars
    for bar, dist in zip(bars, dists):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{dist:.4f}", va="center", fontsize=10)

    fig.tight_layout()
    return fig


# ──────────────────── Tab 1: Offline Detection ──────────────────

def detect_keyword(audio_input, threshold: float):
    """Detect keyword from uploaded/recorded audio."""
    waveform = preprocess_audio(audio_input)
    if waveform is None:
        return "❌ No audio provided", None, None

    if not enrolled_prototypes:
        mfcc_fig = plot_mfcc(waveform)
        return "⚠️ No keywords enrolled. Go to Tab 2 to enroll keywords first.", mfcc_fig, None

    # Extract embedding
    query_emb = extract_embedding(waveform)

    # Compute distances to all prototypes
    distances = {}
    for label, proto in enrolled_prototypes.items():
        dist = torch.dist(query_emb, proto, p=2).item()
        distances[label] = dist

    # Find closest
    closest_label = min(distances, key=distances.get)
    closest_dist = distances[closest_label]

    # Decision
    if closest_dist <= threshold:
        result = f"✅ Keyword Detected: **{closest_label.upper()}**\n\n"
        result += f"Distance: `{closest_dist:.4f}` (threshold: `{threshold:.2f}`)"
    else:
        result = f"❌ **REJECTED** (unknown)\n\n"
        result += f"Closest: `{closest_label}` at distance `{closest_dist:.4f}`\n"
        result += f"Threshold: `{threshold:.2f}`"

    mfcc_fig = plot_mfcc(waveform)
    dist_fig = plot_distances(distances, threshold)

    return result, mfcc_fig, dist_fig


# ──────────────────── Tab 2: Enrollment ─────────────────────────

def enroll_sample(keyword_name: str, audio_input):
    """Add one audio sample for a keyword."""
    if not keyword_name or not keyword_name.strip():
        return "❌ Please enter a keyword name.", get_enrollment_status()

    keyword = keyword_name.strip().lower()
    waveform = preprocess_audio(audio_input)

    if waveform is None:
        return "❌ No audio provided.", get_enrollment_status()

    # Extract embedding
    emb = extract_embedding(waveform)

    # Store
    if keyword not in enrolled_samples:
        enrolled_samples[keyword] = []
    enrolled_samples[keyword].append(emb)

    # Update prototype (mean of all samples)
    stacked = torch.stack(enrolled_samples[keyword])
    prototype = F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
    enrolled_prototypes[keyword] = prototype

    n = len(enrolled_samples[keyword])
    msg = f"✅ Added sample #{n} for **'{keyword}'**"
    if n < 3:
        msg += f"\n\n⚠️ Need at least 3 samples for reliable detection. ({n}/3)"
    else:
        msg += f"\n\n🎯 Prototype ready! ({n} samples)"

    return msg, get_enrollment_status()


def remove_keyword(keyword_name: str):
    """Remove all samples for a keyword."""
    keyword = keyword_name.strip().lower()
    if keyword in enrolled_samples:
        del enrolled_samples[keyword]
    if keyword in enrolled_prototypes:
        del enrolled_prototypes[keyword]
    return f"🗑️ Removed keyword '{keyword}'", get_enrollment_status()


def clear_all_keywords():
    """Remove all enrolled keywords."""
    enrolled_samples.clear()
    enrolled_prototypes.clear()
    return "🗑️ All keywords cleared.", get_enrollment_status()


def get_enrollment_status() -> str:
    """Get markdown table of enrolled keywords."""
    if not enrolled_prototypes:
        return "No keywords enrolled yet. Record samples above to get started."

    lines = ["| Keyword | Samples | Status |",
             "|---------|---------|--------|"]

    for keyword in sorted(enrolled_prototypes.keys()):
        n = len(enrolled_samples.get(keyword, []))
        status = "✅ Ready" if n >= 3 else f"⚠️ Need {3-n} more"
        lines.append(f"| {keyword} | {n} | {status} |")

    # Add distances between prototypes
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
                lines.append(f"| {keywords[i]} ↔ {keywords[j]} | {d:.4f} |")

    return "\n".join(lines)


# ──────────────────── Tab 3: Settings ───────────────────────────

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

    return info


def streaming_detect(audio_input, threshold: float, window_ms: int = 1000, stride_ms: int = 500):
    """Simulated streaming detection on a longer audio file."""
    waveform = preprocess_audio(audio_input)
    if waveform is None:
        return "❌ No audio provided", None

    if not enrolled_prototypes:
        return "⚠️ No keywords enrolled. Go to Tab 2 first.", None

    # For streaming, re-process without 1s limit
    sr, audio_np = audio_input
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    full_waveform = torch.from_numpy(audio_np).unsqueeze(0)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        full_waveform = resampler(full_waveform)

    total_samples = full_waveform.shape[-1]
    window_size = int(SAMPLE_RATE * window_ms / 1000)
    stride = int(SAMPLE_RATE * stride_ms / 1000)

    results = []
    times = []

    pos = 0
    while pos + window_size <= total_samples:
        segment = full_waveform[..., pos:pos + window_size]
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

        t_start = pos / SAMPLE_RATE
        t_end = (pos + window_size) / SAMPLE_RATE
        detected = min_dist <= threshold
        results.append({
            "t_start": t_start,
            "t_end": t_end,
            "label": min_label if detected else "—",
            "dist": min_dist,
            "detected": detected,
        })
        times.append(t_start)
        pos += stride

    # Build results text
    lines = ["### Streaming Results\n"]
    lines.append(f"Audio length: {total_samples/SAMPLE_RATE:.1f}s | "
                 f"Window: {window_ms}ms | Stride: {stride_ms}ms | "
                 f"Segments: {len(results)}\n")
    lines.append("| Time | Keyword | Distance | Status |")
    lines.append("|------|---------|----------|--------|")
    for r in results:
        status = "✅" if r["detected"] else "—"
        lines.append(f"| {r['t_start']:.1f}–{r['t_end']:.1f}s | {r['label']} | "
                     f"{r['dist']:.4f} | {status} |")

    # Timeline plot
    fig, ax = plt.subplots(figsize=(10, 3))
    for r in results:
        color = "#4CAF50" if r["detected"] else "#e0e0e0"
        ax.barh(0, r["t_end"] - r["t_start"], left=r["t_start"],
                height=0.5, color=color, edgecolor="white")
        if r["detected"]:
            ax.text((r["t_start"] + r["t_end"]) / 2, 0,
                    r["label"], ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    ax.set_title("Streaming Detection Timeline")
    ax.set_xlim(0, total_samples / SAMPLE_RATE)
    fig.tight_layout()

    return "\n".join(lines), fig


# ──────────────────── Gradio UI ─────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio app with 3 tabs."""

    with gr.Blocks(
        title="Few-Shot KWS Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange",
        ),
    ) as app:

        gr.Markdown(
            """
            # 🎤 Few-Shot Open-Set Keyword Spotting
            ### DSCNN-L Encoder + Prototypical Network + Open NCM Classifier

            Enroll custom keywords with just **3–5 audio samples**, then detect them in real-time.
            """
        )

        # ─── Tab 1: Offline Detection ───
        with gr.Tab("🔍 Offline Detection"):
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
                    detect_btn = gr.Button("🔍 Detect Keyword", variant="primary", size="lg")

                with gr.Column(scale=2):
                    result_text = gr.Markdown(label="Result")
                    mfcc_plot = gr.Plot(label="MFCC Spectrogram")
                    dist_plot = gr.Plot(label="Distances to Prototypes")

            detect_btn.click(
                fn=detect_keyword,
                inputs=[audio_input, threshold_slider],
                outputs=[result_text, mfcc_plot, dist_plot],
            )

        # ─── Tab 2: Enrollment ───
        with gr.Tab("📝 Enrollment"):
            gr.Markdown(
                """
                Record **3–5 samples** per keyword to enroll it.
                Say the keyword clearly, ~1 second per sample.
                """
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
                    enroll_btn = gr.Button("➕ Add Sample", variant="primary")

                    gr.Markdown("---")
                    remove_input = gr.Textbox(label="Remove Keyword", placeholder="keyword name")
                    with gr.Row():
                        remove_btn = gr.Button("🗑️ Remove", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear All", variant="stop")

                with gr.Column(scale=2):
                    enroll_msg = gr.Markdown(label="Status")
                    enroll_status = gr.Markdown(
                        value=get_enrollment_status,
                        label="Enrolled Keywords",
                    )

            enroll_btn.click(
                fn=enroll_sample,
                inputs=[keyword_input, enroll_audio],
                outputs=[enroll_msg, enroll_status],
            )
            remove_btn.click(
                fn=remove_keyword,
                inputs=[remove_input],
                outputs=[enroll_msg, enroll_status],
            )
            clear_btn.click(
                fn=clear_all_keywords,
                inputs=[],
                outputs=[enroll_msg, enroll_status],
            )

        # ─── Tab 3: Settings + Streaming ───
        with gr.Tab("⚙️ Settings & Streaming"):
            with gr.Row():
                with gr.Column():
                    model_info = gr.Markdown(value=get_model_info)

                with gr.Column():
                    gr.Markdown("### Simulated Streaming Test")
                    gr.Markdown("Upload a **longer audio file** (5–30s) to test sliding-window detection.")

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
                    stream_btn = gr.Button("▶️ Run Streaming Detection", variant="primary")

                    stream_result = gr.Markdown()
                    stream_plot = gr.Plot(label="Timeline")

            stream_btn.click(
                fn=streaming_detect,
                inputs=[stream_audio, stream_threshold, window_slider, stride_slider],
                outputs=[stream_result, stream_plot],
            )

    return app


# ─────────────────────── Main ───────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Few-Shot Open-Set Keyword Spotting — Demo")
    print("=" * 60)

    load_model()

    print("\n🚀 Starting Gradio server...")
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
