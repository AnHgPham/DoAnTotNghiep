"""Plot DET curves from saved evaluation JSON files.

Usage:
    python scripts/plot_det_curves.py results/gsc_fixed_results.json \
        results/gsc_random_results.json --output results/det_curves.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def mean_det_curve(result: dict, n_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Average per-run DET curves on a common FAR grid."""
    common_far = np.linspace(0, 1, n_points)
    frr_curves = []

    for run in result.get("per_run", []):
        curve = run.get("det_curve", {})
        far = np.asarray(curve.get("far", []), dtype=float)
        frr = np.asarray(curve.get("frr", []), dtype=float)
        if far.size == 0 or frr.size == 0:
            continue
        frr_curves.append(np.interp(common_far, far, frr))

    if not frr_curves:
        return common_far, np.ones_like(common_far)

    return common_far, np.mean(frr_curves, axis=0)


def label_for(path: Path, result: dict) -> str:
    """Build a readable curve label from result metadata."""
    dataset = result.get("dataset")
    mode = result.get("mode")
    if dataset and mode:
        return f"{dataset.upper()} {str(mode).title()}"
    return path.stem


def _approx_eer(far: np.ndarray, frr: np.ndarray) -> float:
    """Approximate EER from a mean DET curve."""
    idx = int(np.nanargmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0)


def _frr_at_far(far: np.ndarray, frr: np.ndarray, target_far: float) -> float:
    """Interpolate FRR at a target FAR."""
    return float(np.interp(target_far, far, frr))


def plot_enhanced_det(
    entries: list[tuple[str, dict, tuple[np.ndarray, np.ndarray]]],
    output: Path,
    target_far: float = 0.05,
) -> None:
    """Plot DET curves with operating point annotations for reports."""
    fig, ax = plt.subplots(figsize=(9, 7))
    summary_lines = []

    for label, result, (far, frr) in entries:
        auc = result.get("auc")
        acc = result.get("open_set_acc_at_far", result.get("open_set_acc_at_5far"))
        frr_target = result.get("frr_at_far", _frr_at_far(far, frr, target_far))
        eer = result.get("eer", _approx_eer(far, frr))

        legend = (
            f"{label} | AUC={auc:.3f}, EER={eer*100:.1f}%, "
            f"ACC@5%FAR={acc*100:.1f}%"
        )
        mask = (far > 0) & (frr > 0)
        (line,) = ax.plot(far[mask] * 100, frr[mask] * 100, linewidth=2.3, label=legend)

        target_frr_from_curve = _frr_at_far(far, frr, target_far)
        ax.scatter(
            [target_far * 100],
            [target_frr_from_curve * 100],
            color=line.get_color(),
            edgecolor="black",
            zorder=5,
        )
        ax.annotate(
            f"{target_frr_from_curve * 100:.1f}%",
            xy=(target_far * 100, target_frr_from_curve * 100),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

        summary_lines.append(
            f"{label}: FRR@5%FAR={frr_target*100:.1f}%, "
            f"ACC={acc*100:.1f}%, AUC={auc:.3f}"
        )

    ax.axvline(target_far * 100, color="black", linestyle="--", linewidth=1.4, alpha=0.75)
    ax.text(
        target_far * 100,
        0.65,
        "5% FAR operating point",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=9,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Acceptance Rate, FAR (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate, FRR (%)", fontsize=12)
    ax.set_title("Detection Error Tradeoff (DET) Curve on GSC v2", fontsize=14)
    tick_values = [0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    ax.set_xticklabels([str(v) for v in tick_values])
    ax.set_yticklabels([str(v) for v in tick_values])
    ax.set_xlim(0.5, 50)
    ax.set_ylim(0.5, 50)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(fontsize=8, loc="upper right")

    ax.text(
        0.02,
        0.03,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DET curves from result JSON files")
    parser.add_argument("results", nargs="+", type=Path, help="Evaluation JSON files")
    parser.add_argument("--output", type=Path, default=Path("results/det_curves.png"))
    args = parser.parse_args()

    entries = []
    for result_path in args.results:
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        entries.append((label_for(result_path, result), result, mean_det_curve(result)))

    plot_enhanced_det(entries, args.output)
    print(f"Saved DET curves to {args.output}")


if __name__ == "__main__":
    main()
