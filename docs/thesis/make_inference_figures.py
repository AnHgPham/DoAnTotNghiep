"""Generate checkpoint-based thesis figures (REAL model, REAL audio).

Outputs (docs/thesis/assets/):
  - fig_embedding_space.png : t-SNE of GSC embeddings (known commands vs unknown)
                              with per-class prototypes -> the "model output" view.
  - fig_long_inference.png  : nearest-prototype inference on a real long audio file,
                              predicted vs ground-truth keyword per segment.

Uses the best local checkpoint:
  checkpoints/dscnn_pcen_ge2e_accdev_ep300_composite_colab_mswc.pt  (DSCNN-L, mel+PCEN)

Run:  python docs/thesis/make_inference_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.dscnn import DSCNN                       # noqa: E402
from src.features.mel import MelSpectrogramExtractor      # noqa: E402
from src.audio_io import load_waveform, pad_or_trim       # noqa: E402
from src.evaluation.gsc import GSCFewShotProvider         # noqa: E402

ASSETS = ROOT / "docs" / "thesis" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "checkpoints" / "dscnn_pcen_ge2e_accdev_ep300_composite_colab_mswc.pt"
GSC = ROOT / "data" / "gsc_v2"
SR = 16000
DEV = torch.device("cpu")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def load_encoder() -> DSCNN:
    ck = torch.load(str(CKPT), map_location=DEV, weights_only=False)
    enc = DSCNN(model_size="L", feature_mode="NORM",
                input_shape=(40, 101), use_pcen=True)
    enc.load_state_dict(ck["model_state_dict"])
    enc.eval().to(DEV)
    print(f"loaded {CKPT.name} (epoch={ck.get('epoch')}, val_auc={ck.get('val_auc')})")
    return enc


@torch.no_grad()
def embed(enc: DSCNN, feats: torch.Tensor) -> torch.Tensor:
    e = enc(feats.to(DEV))
    return F.normalize(e, p=2, dim=-1).cpu()


def fig_embedding_space(enc: DSCNN) -> None:
    prov = GSCFewShotProvider(GSC, feature_type="mel", query_split="test")
    known = ["yes", "no", "up", "down", "stop", "go"]
    unknown = ["bed", "bird", "cat"]
    words = known + unknown
    per_word = 45

    all_emb, all_lab = [], []
    for w in words:
        try:
            feats, _ = prov.get_query_samples(w, max_samples=per_word)
        except Exception as exc:           # noqa: BLE001
            print("skip", w, exc); continue
        if feats.numel() == 0:
            continue
        e = embed(enc, feats).numpy()
        all_emb.append(e)
        all_lab += [w] * len(e)
    X = np.concatenate(all_emb, 0)
    labs = np.array(all_lab)
    print("embedding matrix", X.shape)

    Y = TSNE(n_components=2, perplexity=30, init="pca",
             random_state=42, learning_rate="auto").fit_transform(X)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    cmap = plt.get_cmap("tab10")
    known_colors = {w: cmap(i) for i, w in enumerate(known)}
    # known clusters
    for w in known:
        m = labs == w
        ax.scatter(Y[m, 0], Y[m, 1], s=22, color=known_colors[w],
                   alpha=0.75, edgecolor="none", label=f'known: "{w}"')
        cx, cy = Y[m, 0].mean(), Y[m, 1].mean()
        ax.scatter(cx, cy, marker="*", s=320, color=known_colors[w],
                   edgecolor="black", linewidth=1.0, zorder=5)
    # unknown words pooled
    mu = np.isin(labs, unknown)
    ax.scatter(Y[mu, 0], Y[mu, 1], s=20, color="0.6", marker="x",
               alpha=0.7, label="unknown words")
    ax.set_title("Embedding space (t-SNE) of DSCNN-L + PCEN + GE2E\n"
                 "stars = class prototypes; grey crosses = out-of-vocabulary")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9)
    out = ASSETS / "fig_embedding_space.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


@torch.no_grad()
def fig_long_inference(enc: DSCNN) -> None:
    mel = MelSpectrogramExtractor()
    prov = GSCFewShotProvider(GSC, feature_type="mel", query_split="test")
    long_wav = (ROOT / "data" / "test" / "generated" / "long_audio"
                / "gsc_demo_17kw_one_100words.wav")
    timings = json.loads((long_wav.with_suffix(".timings.json")).read_text())
    words = sorted({w["label"] for w in timings["words"]})

    # enroll prototypes (10 shots from GSC val)
    protos, plabels = [], []
    for w in words:
        try:
            feats, _ = prov.get_support_samples(w, 10, seed=42)
        except Exception:                  # noqa: BLE001
            continue
        protos.append(embed(enc, feats).mean(0))
        plabels.append(w)
    P = torch.stack(protos)

    wav = load_waveform(long_wav, sample_rate=SR, mono=True).squeeze(0)
    segs = timings["words"][:16]           # first 16 segments for legibility
    times, dists, ok, gts, preds = [], [], [], [], []
    for s in segs:
        a, b = int(s["start_sample"]), int(s["end_sample"])
        clip = pad_or_trim(wav[a:b].unsqueeze(0), SR)
        feat = mel.extract(clip).unsqueeze(0)
        e = embed(enc, feat)[0]
        d = torch.cdist(e.unsqueeze(0), P).squeeze(0)
        j = int(d.argmin())
        times.append((a / SR + b / SR) / 2)
        dists.append(float(d[j]))
        preds.append(plabels[j]); gts.append(s["label"])
        ok.append(plabels[j] == s["label"])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.6, 5.0),
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    last = segs[-1]["end_sec"] + 0.5
    n = min(int(last * SR), len(wav))
    tt = np.arange(n) / SR
    ax0.plot(tt, wav[:n].numpy(), lw=0.4, color="#1f3b73")
    ax0.set_xlim(0, last); ax0.set_ylabel("Amp.")
    ax0.set_title("Nearest-prototype inference on a real long recording "
                  "(17 enrolled keywords)")
    for s, o in zip(segs, ok):
        ax0.axvspan(s["start_sec"], s["end_sec"],
                    color=("#2ca02c" if o else "#d62728"), alpha=0.12)

    ax1.scatter(times, dists, c=["#2ca02c" if o else "#d62728" for o in ok],
                s=60, edgecolor="black", zorder=3)
    for t, d, p, o in zip(times, dists, preds, ok):
        ax1.annotate(p, (t, d), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7.5,
                     color=("#2ca02c" if o else "#d62728"))
    ax1.set_xlim(0, last)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("min prototype distance")
    ax1.grid(True, ls=":", alpha=0.4)
    acc = 100 * np.mean(ok)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
                  markersize=9, label="correct"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                  markersize=9, label="wrong")]
    ax1.legend(handles=leg, fontsize=8, loc="upper right",
               title=f"segment acc = {acc:.0f}%")
    out = ASSETS / "fig_long_inference.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out, f"(acc={acc:.0f}%)")


if __name__ == "__main__":
    enc = load_encoder()
    try:
        fig_embedding_space(enc)
    except Exception as exc:               # noqa: BLE001
        print("EMBEDDING FIG FAILED:", exc)
    try:
        fig_long_inference(enc)
    except Exception as exc:               # noqa: BLE001
        print("INFERENCE FIG FAILED:", exc)
    print("done.")
