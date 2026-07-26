"""Plot the SCAF collapse SIGNATURE from REAL final metrics (no retraining).

The per-epoch TensorBoard logs of the full-vocabulary SCAF+GE2E run are NOT in
this repo (they live on Google Drive). What we DO have locally is the final
cap-620 16-pipeline GSC test100 @1% FAR table -- the exact same source as the
thesis Table `tab:matrix`. This script visualizes the collapse from those real
numbers, so the figure is fully reproducible offline and cannot contradict the
text.

Source of every number below (DSCNN-L + PCEN branch):
  docs/thesis/cap620_16_pipeline_scientific_chapter_vi_2026_06_12.md  (Sec. 6.1)
  docs/reports/cap620_16_pipeline_test100_far1_compact_table_vi.md

Run:  python docs/thesis/make_collapse_figure.py
Output: docs/thesis/assets/scaf_collapse.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- REAL cap-620 GSC test100 @1% FAR numbers (DSCNN-L + PCEN) ----------------
# config            AUC     F1     KeywordACC  FRR@1%FAR  ACC@1%FAR
DATA = {
    "GE2E\n(healthy)":      dict(auc=92.42, f1=77.75, kw=88.81, frr=54.55, acc=82.34),
    "SCAF\n(collapse)":     dict(auc=50.00, f1=0.00,  kw=9.09,  frr=100.0, acc=69.44),
    "SCAF+GE2E\n(collapse)":dict(auc=50.00, f1=0.00,  kw=9.09,  frr=100.0, acc=69.44),
}
CONFIGS = list(DATA.keys())
COLORS = ["#08519c", "#cb181d", "#fb6a4a"]  # healthy blue, collapsed reds

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def bars(ax, metrics, labels, title, refs=None):
    x = np.arange(len(metrics))
    w = 0.26
    for i, cfg in enumerate(CONFIGS):
        vals = [DATA[cfg][m] for m in metrics]
        b = ax.bar(x + (i - 1) * w, vals, w, label=cfg.replace("\n", " "),
                   color=COLORS[i], edgecolor="black", linewidth=0.4)
        ax.bar_label(b, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Percent (%)"); ax.set_ylim(0, 109)
    ax.set_title(title); ax.grid(True, axis="y", ls=":", alpha=0.4)
    if refs:
        for y, txt in refs:
            ax.axhline(y, ls="--", lw=1.0, color="black", alpha=0.6)
            ax.text(-0.45, y + 1.8, txt, fontsize=7,
                    ha="left", va="bottom", color="black", alpha=0.8)


def main() -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.7))

    # (a) discrimination metrics: higher = better; collapse floors them out
    bars(ax0, ["auc", "f1", "kw"], ["AUC", "F1", "Keyword\nACC"],
         "(a) Discrimination collapses (higher better)",
         refs=[(50.0, "AUC chance = 50"), (9.09, "keyword chance = 9.09")])
    ax0.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    # (b) the misleading-ACC trap: collapsed runs reject everything (FRR=100),
    #     yet open-set ACC stays ~69% against the many negatives.
    bars(ax1, ["acc", "frr"], ["ACC@1%FAR", "FRR@1%FAR"],
         "(b) Degenerate operating point",
         refs=[(100.0, "reject-all FRR = 100")])
    ax1.set_ylim(0, 118)

    fig.suptitle("SCAF collapse signature at full 37,387-class vocabulary "
                 "(cap-620, GSC test100 @ 1% FAR)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ASSETS / "scaf_collapse.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
