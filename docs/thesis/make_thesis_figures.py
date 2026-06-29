"""Generate thesis-standard illustration figures from REAL project data.

Outputs (docs/thesis/assets/):
  - fig_audio_features.png : waveform -> log-mel -> MFCC(10) of a real GSC "yes"
  - fig_vocab_scale.png    : training-vocabulary scale (log) across regimes
  - fig_data_saturation.png: cross-corpus ACC@5%FAR vs training-clip volume

All numbers in the saturation/scale figures are the project's measured values.
Run:  python docs/thesis/make_thesis_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.audio_io import load_waveform              # noqa: E402
from src.features.mfcc import MFCCExtractor          # noqa: E402
from src.features.mel import MelSpectrogramExtractor # noqa: E402

ASSETS = ROOT / "docs" / "thesis" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Consistent thesis styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

SR = 16000


def pick_sample() -> Path:
    cand_dir = ROOT / "data" / "gsc_v2" / "yes"
    wavs = sorted(cand_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No wav found in {cand_dir}")
    # a mid-list file tends to be a clean, centered utterance
    return wavs[len(wavs) // 3]


def fig_audio_features() -> None:
    wav_path = pick_sample()
    wav = load_waveform(wav_path, sample_rate=SR, mono=True, target_length=SR)  # (1,T)
    t = np.arange(wav.shape[-1]) / SR

    mel_ext = MelSpectrogramExtractor()
    mfcc_ext = MFCCExtractor()
    mel = mel_ext.extract(wav).squeeze(0).numpy()          # (40,101)
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-8))
    mfcc = mfcc_ext.extract(wav).squeeze(0).numpy()        # (47,10)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.25, 1.25], hspace=0.62)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, wav.squeeze(0).numpy(), lw=0.6, color="#1f3b73")
    ax0.set_xlim(0, 1.0)
    ax0.set_title('(a) Raw waveform of the spoken word "yes" (1 s, 16 kHz)')
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(log_mel, origin="lower", aspect="auto",
                     extent=[0, 1.0, 0, 40], cmap="magma")
    ax1.set_title("(b) Log-mel spectrogram (40 bands)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mel band")
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.01)
    cb1.set_label("dB")

    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(mfcc.T, origin="lower", aspect="auto",
                     extent=[0, 1.0, 0, 10], cmap="viridis")
    ax2.set_title("(c) MFCC feature map fed to DSCNN: first 10 coefficients, "
                  r"$T{=}47$ frames")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("MFCC index")
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.01)
    cb2.set_label("coeff.")

    out = ASSETS / "fig_audio_features.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, "from", wav_path.name)


def fig_vocab_scale() -> None:
    regimes = ["Micro-set", "Top-500", "Full MSWC-EN"]
    counts = [31, 500, 37387]
    colors = ["#9ecae1", "#4292c6", "#08519c"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.bar(regimes, counts, color=colors, width=0.6, edgecolor="black", lw=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Number of training words (log)")
    ax.set_title("Training-vocabulary scale across regimes")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c * 1.15, f"{c:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(10, 1e5)
    out = ASSETS / "fig_vocab_scale.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_data_saturation() -> None:
    clips = [0.53, 0.94, 2.05, 2.99]            # millions
    dscnn = [86.05, 84.68, 88.23, 88.56]        # ACC@5%FAR, GSC test100
    edge = [83.06, 82.24, 86.03, 86.01]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(clips, dscnn, "-o", color="#08519c", lw=1.8, label="DSCNN-L + PCEN + GE2E")
    ax.plot(clips, edge, "-s", color="#d94801", lw=1.8, label="EdgeSpot-Full + PCEN + GE2E")
    ax.axvspan(2.05, 2.99, color="gray", alpha=0.10)
    ax.text(2.5, 83.4, "saturation\nregion", ha="center", fontsize=8, color="gray")
    for x, y in zip(clips, dscnn):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7.5, color="#08519c")
    for x, y in zip(clips, edge):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=7.5, color="#d94801")
    ax.set_xlabel("Training clips (millions)")
    ax.set_ylabel("ACC@5%FAR on GSC test100 (%)")
    ax.set_title("Cross-corpus accuracy vs. training-data volume")
    ax.set_xticks(clips)
    ax.set_xticklabels(["0.53\n(cap20)", "0.94\n(cap50)", "2.05\n(cap220)", "2.99\n(cap620)"])
    ax.set_ylim(81, 90)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    out = ASSETS / "fig_data_saturation.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _logmel(wav: torch.Tensor) -> np.ndarray:
    mel = MelSpectrogramExtractor().extract(wav).squeeze(0).numpy()
    return 10.0 * np.log10(np.maximum(mel, 1e-8))


def fig_data_examples() -> None:
    """Grid of log-mel spectrograms for several real GSC words (the data 'looks like')."""
    words = ["yes", "no", "stop", "go", "left", "right"]
    base = ROOT / "data" / "gsc_v2"
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.6))
    for ax, w in zip(axes.ravel(), words):
        wavs = sorted((base / w).glob("*.wav"))
        if not wavs:
            ax.set_visible(False)
            continue
        wav = load_waveform(wavs[len(wavs) // 3], sample_rate=SR, mono=True, target_length=SR)
        lm = _logmel(wav)
        ax.imshow(lm, origin="lower", aspect="auto", extent=[0, 1, 0, 40], cmap="magma")
        ax.set_title(f'"{w}"', fontsize=10)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 20, 40])
    fig.suptitle("Log-mel spectrograms of six Google Speech Commands words",
                 fontsize=11)
    fig.supxlabel("Time (s)", fontsize=9)
    fig.supylabel("Mel band", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    out = ASSETS / "fig_data_examples.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_augmentation() -> None:
    """Clean vs +background noise (real GSC noise, 5 dB) vs +SpecAugment, on log-mel."""
    base = ROOT / "data" / "gsc_v2"
    wavs = sorted((base / "yes").glob("*.wav"))
    clean = load_waveform(wavs[len(wavs) // 3], sample_rate=SR, mono=True, target_length=SR)

    # real ambient noise from GSC _background_noise_
    noise_files = sorted((base / "_background_noise_").glob("*.wav"))
    if noise_files:
        ns = load_waveform(noise_files[0], sample_rate=SR, mono=True)
        if ns.shape[-1] >= SR:
            ns = ns[..., :SR]
        else:
            ns = torch.nn.functional.pad(ns, (0, SR - ns.shape[-1]))
    else:
        ns = torch.randn_like(clean) * 0.01
    rms_c = torch.sqrt(torch.mean(clean ** 2) + 1e-8)
    rms_n = torch.sqrt(torch.mean(ns ** 2) + 1e-8)
    scale = rms_c / (rms_n * 10 ** (5.0 / 20.0))   # SNR = 5 dB
    noisy = clean + scale * ns

    lm_clean = _logmel(clean)
    lm_noisy = _logmel(noisy)
    lm_spec = lm_clean.copy()
    rng = np.random.default_rng(0)
    f0 = rng.integers(0, 40 - 6); lm_spec[f0:f0 + 6, :] = lm_spec.min()
    t0 = rng.integers(0, 101 - 8); lm_spec[:, t0:t0 + 8] = lm_spec.min()

    titles = ["(a) Clean", "(b) + background noise (5 dB SNR)", "(c) + SpecAugment masks"]
    mats = [lm_clean, lm_noisy, lm_spec]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, m, t in zip(axes, mats, titles):
        ax.imshow(m, origin="lower", aspect="auto", extent=[0, 1, 0, 40], cmap="magma")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Mel band")
    fig.suptitle('Data augmentation applied to the word "yes"', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS / "fig_augmentation.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_mel_filterbank() -> None:
    """The 40 triangular mel filters and the Hz->mel warping curve (background)."""
    ext = MelSpectrogramExtractor()
    fb = ext.mel_filterbank.numpy()                 # (40, n_freq)
    n_freq = fb.shape[1]
    freqs = np.linspace(0, SR / 2, n_freq)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for i in range(fb.shape[0]):
        ax0.plot(freqs, fb[i], lw=0.8)
    ax0.set_title("(a) 40 triangular mel filters")
    ax0.set_xlabel("Frequency (Hz)")
    ax0.set_ylabel("Filter gain")
    ax0.set_xlim(0, SR / 2)
    f = np.linspace(0, SR / 2, 400)
    m = 2595 * np.log10(1 + f / 700)
    ax1.plot(f, m, color="#08519c", lw=2)
    ax1.set_title(r"(b) Mel scale: $m=2595\log_{10}(1+f/700)$")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Mel")
    ax1.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    out = ASSETS / "fig_mel_filterbank.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_model_tradeoff() -> None:
    """Accuracy vs parameter-count trade-off (the 'model size comparison' analogue)."""
    # (params in M, ACC@1%FAR test100, label)
    pts = [
        (0.413, 86.36, "DSCNN-L\n+PCEN+GE2E", "#08519c", "o"),
        (0.131, 82.87, "EdgeSpot-Full\n+PCEN+GE2E", "#d94801", "s"),
        (0.131, 80.74, "EdgeSpot-Full\n+KD (cap50)", "#fd8d3c", "^"),
        (0.128, 82.00, "EdgeSpot-4\n(reference)", "#6a51a3", "D"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for x, y, lab, c, mk in pts:
        ax.scatter(x, y, s=120, color=c, marker=mk, edgecolor="black", zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=8.5)
    ax.set_xlabel("Parameters (millions) — lower is smaller")
    ax.set_ylabel("ACC@1%FAR on GSC test100 (%)")
    ax.set_title("Accuracy vs. model size trade-off")
    ax.set_xlim(0.10, 0.46)
    ax.set_ylim(79, 88)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    out = ASSETS / "fig_model_tradeoff.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_audio_features()
    fig_vocab_scale()
    fig_data_saturation()
    fig_data_examples()
    fig_augmentation()
    fig_mel_filterbank()
    fig_model_tradeoff()
    print("done.")
