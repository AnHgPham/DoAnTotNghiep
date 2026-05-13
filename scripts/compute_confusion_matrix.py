"""Compute confusion matrix for closed-set keyword classification on GSC test.

Identifies which keyword pairs the encoder confuses most often. Uses 5-shot
enrollment per word and evaluates top-1 prediction on remaining test samples.

Output:
  - Console: confusion matrix + top confused pairs sorted by error rate
  - results/confusion_matrix_best_v2.json: machine-readable report
  - results/confusion_matrix_best_v2.png: heatmap visualisation
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
import os
_CKPT_NAME = os.environ.get("CKPT_NAME", "best_v2_margin1.0_colab")
CKPT = PROJECT_ROOT / f"checkpoints/triplet/{_CKPT_NAME}.pt"
GSC = PROJECT_ROOT / "data/gsc_v2"
OUT_JSON = PROJECT_ROOT / f"results/confusion_matrix_{_CKPT_NAME}.json"
OUT_PNG = PROJECT_ROOT / f"results/confusion_matrix_{_CKPT_NAME}.png"

KEYWORDS = [
    "yes", "no", "stop", "go", "up", "down", "left", "right", "on", "off",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual",
]
N_ENROLL = 5
MAX_TEST = 100


def load_wav(path: Path) -> torch.Tensor:
    w, sr = torchaudio.load(str(path))
    if sr != SR:
        w = torchaudio.transforms.Resample(sr, SR)(w)
    if w.shape[-1] < SR:
        w = F.pad(w, (0, SR - w.shape[-1]))
    return w[..., :SR]


def main() -> None:
    print(f"Loading model: {CKPT.name}")
    encoder = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))
    ckpt = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder = encoder.to(DEVICE).eval()
    mfcc_ext = MFCCExtractor(n_mfcc=40, num_features=10, sample_rate=SR)
    print(f"  epoch={ckpt.get('epoch','?')}, val_auc={ckpt.get('val_auc',0):.4f}")

    @torch.no_grad()
    def embed(wav: torch.Tensor) -> torch.Tensor:
        m = mfcc_ext.extract(wav).unsqueeze(0).to(DEVICE)
        return F.normalize(encoder(m), p=2, dim=-1).squeeze(0).cpu()

    available = []
    for w in KEYWORDS:
        d = GSC / w
        if d.exists():
            files = sorted(d.glob("*.wav"))
            if len(files) >= N_ENROLL + 10:
                available.append((w, files))
    words = [w for w, _ in available]
    print(f"\n{len(words)} keywords available: {words}")

    print(f"\nEnrolling {N_ENROLL} samples / keyword...")
    prototypes = []
    test_pool: dict[str, list[torch.Tensor]] = {}
    for w, files in available:
        embs = [embed(load_wav(f)) for f in files[:N_ENROLL]]
        prototypes.append(torch.stack(embs).mean(0))
        test_files = files[N_ENROLL : N_ENROLL + MAX_TEST]
        test_pool[w] = [embed(load_wav(f)) for f in test_files]
        print(f"  {w:10s}: enroll {N_ENROLL}, test {len(test_files)}")

    proto_tensor = torch.stack(prototypes)

    n = len(words)
    cm = np.zeros((n, n), dtype=int)
    word_to_idx = {w: i for i, w in enumerate(words)}
    confused_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)

    print("\nClassifying test samples...")
    for true_idx, true_w in enumerate(words):
        for emb in test_pool[true_w]:
            dists = torch.cdist(emb.unsqueeze(0), proto_tensor).squeeze(0)
            pred_idx = int(dists.argmin().item())
            cm[true_idx, pred_idx] += 1
            if pred_idx != true_idx:
                pred_w = words[pred_idx]
                confused_pairs[(true_w, pred_w)].append(float(dists[pred_idx].item()))

    diag = np.diag(cm)
    per_class_total = cm.sum(axis=1)
    per_class_acc = diag / np.maximum(per_class_total, 1)
    overall_acc = diag.sum() / cm.sum()

    print("\n" + "=" * 60)
    print(f"Overall closed-set top-1 accuracy: {overall_acc:.4f}")
    print("=" * 60)
    print(f"\n{'Word':>10s}  {'Acc':>6s}  {'N':>4s}  Most confused -> ...")
    print("-" * 70)

    sorted_words = sorted(zip(words, per_class_acc), key=lambda x: x[1])
    for w, acc in sorted_words:
        i = word_to_idx[w]
        wrong_idx = np.argsort(cm[i])[::-1]
        confs = []
        for j in wrong_idx:
            if j == i:
                continue
            if cm[i, j] == 0:
                continue
            confs.append(f"{words[j]}:{cm[i, j]}")
            if len(confs) >= 3:
                break
        print(f"{w:>10s}  {acc:6.3f}  {per_class_total[i]:>4d}  {', '.join(confs)}")

    print("\nTop 15 confused pairs (sorted by count):")
    print("-" * 70)
    pair_counts = sorted(
        ((p, len(d)) for p, d in confused_pairs.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:15]
    for (true_w, pred_w), cnt in pair_counts:
        avg_d = float(np.mean(confused_pairs[(true_w, pred_w)]))
        print(f"  {true_w:>10s} -> {pred_w:<10s}  count={cnt:>3d}  avg_dist={avg_d:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "checkpoint": str(CKPT.relative_to(PROJECT_ROOT)),
        "epoch": int(ckpt.get("epoch", -1)),
        "val_auc": float(ckpt.get("val_auc", 0.0)),
        "n_keywords": n,
        "n_enroll": N_ENROLL,
        "max_test_per_word": MAX_TEST,
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": {w: float(a) for w, a in zip(words, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "labels": words,
        "top_confused_pairs": [
            {"true": t, "pred": p, "count": c,
             "avg_dist": float(np.mean(confused_pairs[(t, p)]))}
            for (t, p), c in pair_counts
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nReport saved: {OUT_JSON}")

    fig, ax = plt.subplots(figsize=(11, 9))
    cm_norm = cm / np.maximum(per_class_total[:, None], 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(words, rotation=70, fontsize=8)
    ax.set_yticklabels(words, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Closed-set confusion matrix (best_v2, acc={overall_acc:.3f})")
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"Heatmap saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
