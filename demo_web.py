"""Gradio web demo for Few-Shot Open-Set KWS.

Usage:
    python demo_web.py
    -> Opens browser at http://localhost:7860

Loads ``checkpoints/best.pt`` when present (same format as Colab notebook / ``scripts/train.py``).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import gradio as gr

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.classifiers.open_ncm import OpenNCMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
extractor = MFCCExtractor()
encoder = DSCNN(model_size="L", feature_mode="NORM")
if CHECKPOINT_PATH.exists():
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {CHECKPOINT_PATH} (epoch {ckpt.get('epoch', '?')})")
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        encoder.load_state_dict(ckpt)
        print(f"Loaded state dict: {CHECKPOINT_PATH}")
else:
    print(f"No checkpoint at {CHECKPOINT_PATH} — using random weights (UI test only)")
encoder = encoder.to(DEVICE)
encoder.eval()
classifier = OpenNCMClassifier()

enrolled_data: dict[str, torch.Tensor] = {}
param_count = sum(p.numel() for p in encoder.parameters())
_checkpoint_loaded = CHECKPOINT_PATH.exists()
print(f"DSCNN-L loaded: {param_count:,} params, embedding_dim={encoder.embedding_dim}")


def load_audio(filepath: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(filepath)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def make_mfcc_plot(mfcc: torch.Tensor) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    data = mfcc.squeeze().numpy().T
    ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("MFCC Coeff")
    ax.set_title(f"MFCC ({data.shape[0]} x {data.shape[1]})")
    plt.tight_layout()
    return fig


def make_dist_plot(distances: dict[str, float], threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, max(2.5, len(distances) * 0.6)))
    labels = list(distances.keys())
    dists = list(distances.values())
    colors = ["#27AE60" if d <= threshold else "#E74C3C" for d in dists]
    ax.barh(labels, dists, color=colors)
    ax.axvline(x=threshold, color="gray", linestyle="--", linewidth=2)
    ax.set_xlabel("L2 Distance")
    ax.set_title(f"Distances (threshold={threshold:.2f})")
    for i, d in enumerate(dists):
        ax.text(d + 0.01, i, f"{d:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def enroll(keyword_name: str, *audio_files):
    if not keyword_name or not keyword_name.strip():
        return "Enter a keyword name."

    files = [f for f in audio_files if f is not None]
    if not files:
        return "Upload at least 1 audio file."

    keyword_name = keyword_name.strip().lower()
    embeddings = []
    for fpath in files:
        wav = load_audio(fpath)
        mfcc = extractor.extract(wav).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = F.normalize(encoder(mfcc), p=2, dim=-1).squeeze(0).cpu()
        embeddings.append(emb)

    prototype = torch.stack(embeddings).mean(dim=0)
    enrolled_data[keyword_name] = prototype

    proto_tensor = torch.stack(list(enrolled_data.values()))
    classifier.set_prototypes(proto_tensor, list(enrolled_data.keys()))
    if classifier.threshold is None:
        classifier.threshold = 1.4

    kw_list = ", ".join(enrolled_data.keys())
    return f"Enrolled '{keyword_name}' ({len(embeddings)} samples). Keywords: [{kw_list}]"


def detect(audio_file, threshold):
    if audio_file is None:
        return "Upload an audio file.", None, None
    if not enrolled_data:
        return "No keywords enrolled yet! Go to Enrollment tab.", None, None

    classifier.threshold = threshold
    wav = load_audio(audio_file)
    mfcc = extractor.extract(wav)
    mfcc_fig = make_mfcc_plot(mfcc)

    with torch.no_grad():
        emb = F.normalize(encoder(mfcc.unsqueeze(0).to(DEVICE)), p=2, dim=-1).squeeze(0).cpu()

    pred, dist = classifier.predict(emb)
    distances = classifier.get_distances(emb)
    dist_fig = make_dist_plot(distances, threshold)

    if pred == "unknown":
        result = f"REJECTED (unknown)\nMin distance: {dist:.4f} > threshold {threshold:.2f}"
    else:
        result = f"DETECTED: '{pred}'\nDistance: {dist:.4f}"

    return result, mfcc_fig, dist_fig


_model_line = (
    f"Trained weights: `{CHECKPOINT_PATH.name}`"
    if _checkpoint_loaded
    else f"Random weights (place `{CHECKPOINT_PATH.name}` in `checkpoints/`)"
)

with gr.Blocks(title="Few-Shot KWS") as app:
    gr.Markdown("# Few-Shot Open-Set Keyword Spotting Demo\n"
                f"DSCNN-L ({param_count:,} params) | {_model_line} | Upload WAV to test")

    with gr.Tab("Enrollment"):
        gr.Markdown("### Upload 1-5 audio samples per keyword")
        kw_name = gr.Textbox(label="Keyword Name", placeholder="yes, stop, hello...")
        with gr.Row():
            a1 = gr.Audio(label="Sample 1", type="filepath", sources=["upload", "microphone"])
            a2 = gr.Audio(label="Sample 2", type="filepath", sources=["upload", "microphone"])
            a3 = gr.Audio(label="Sample 3", type="filepath", sources=["upload", "microphone"])
        enroll_btn = gr.Button("Enroll", variant="primary")
        enroll_out = gr.Textbox(label="Result", lines=2)
        enroll_btn.click(fn=enroll, inputs=[kw_name, a1, a2, a3], outputs=[enroll_out])

    with gr.Tab("Detection"):
        gr.Markdown("### Upload audio to detect keyword")
        with gr.Row():
            det_audio = gr.Audio(label="Audio", type="filepath", sources=["upload", "microphone"])
            threshold = gr.Slider(0.1, 3.0, value=1.4, step=0.05, label="Threshold")
        det_btn = gr.Button("Detect", variant="primary")
        det_result = gr.Textbox(label="Result", lines=3)
        with gr.Row():
            mfcc_out = gr.Plot(label="MFCC")
            dist_out = gr.Plot(label="Distances")
        det_btn.click(fn=detect, inputs=[det_audio, threshold], outputs=[det_result, mfcc_out, dist_out])

    with gr.Tab("About"):
        gr.Markdown(f"""
**Pipeline:** Audio (16kHz) -> MFCC (47x10) -> DSCNN-L ({param_count:,} params, 276-dim) -> L2 norm -> Direct L2 distance

**Model:** Loads `checkpoints/best.pt` if present (Colab / `scripts/train.py` format). Otherwise random weights.

**How to use:**
1. Go to **Enrollment** tab, type a keyword name, upload/record 1-3 samples, click Enroll
2. Repeat for more keywords
3. Go to **Detection** tab, upload/record audio, click Detect
4. Adjust **Threshold** slider (lower = stricter rejection)
        """)

print("Starting server...")
app.launch(server_name="127.0.0.1", server_port=7860, share=False, quiet=False)
