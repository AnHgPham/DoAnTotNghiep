"""Plot REAL training dynamics from local TensorBoard event files.

Reads the Microset anchor run (EdgeSpot-Full + SCAF+GE2E) and produces:
  - fig_training_curves.png : (a) training loss, (b) validation AUC + GSC-dev ACC@1%FAR

No re-training needed; uses logs already on disk.
Run:  python docs/thesis/make_training_curves.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "thesis" / "assets"
RUN = (ROOT / "server" / "DoAnTotNghiep_output-20260522T014622Z-3-001"
       / "DoAnTotNghiep_output" / "checkpoints"
       / "edgespot_full_t4_scaf_ge2e_microset_en_v1" / "runs")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def series(ea: EventAccumulator, tag: str):
    """Return the LAST monotonic-step segment (handles restarts that merge
    several runs into one event dir)."""
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    ev = ea.Scalars(tag)
    steps = np.array([e.step for e in ev])
    vals = np.array([e.value for e in ev])
    # find the last index where step resets (step <= previous step)
    start = 0
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            start = i
    return steps[start:], vals[start:]


def main() -> None:
    ea = EventAccumulator(str(RUN)); ea.Reload()
    ep_loss, loss = series(ea, "train/loss")
    ep_auc, auc = series(ea, "val/auc")
    ep_acc, acc = series(ea, "gsc_dev/acc_at_1far")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax0.plot(ep_loss, loss, color="#08519c", lw=1.8)
    ax0.set_title("(a) Training loss")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("SCAF+GE2E loss")
    ax0.grid(True, ls=":", alpha=0.4)

    ax1.plot(ep_auc, auc * 100, "-o", ms=3, color="#08519c", label="val AUC")
    if acc.size:
        sc = acc * 100 if acc.max() <= 1.5 else acc
        ax1.plot(ep_acc, sc, "-s", ms=4, color="#d94801",
                 label="GSC-dev ACC@1%FAR")
    ax1.set_title("(b) Validation AUC and GSC-dev accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Percent (%)")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(fontsize=8, loc="lower right")

    fig.suptitle("Training dynamics of the Microset anchor "
                 "(EdgeSpot-Full + SCAF+GE2E)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS / "fig_training_curves.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out, "| epochs:", len(loss))


if __name__ == "__main__":
    main()
