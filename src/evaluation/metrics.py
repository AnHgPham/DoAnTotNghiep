"""Evaluation metrics for few-shot open-set keyword spotting.

Provides DET curve computation, AUC, ACC@FAR, FRR@FAR, and DET plotting.
All metrics operate on binary (keyword vs non-keyword) decisions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_det_curve(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Detection Error Tradeoff (DET) curve.

    Args:
        y_true: Binary labels (1=keyword, 0=non-keyword/unknown).
        scores: Confidence scores (higher = more likely keyword).
            For L2 distance: use negative distance so closer = higher score.

    Returns:
        (far, frr) arrays where FAR = False Acceptance Rate, FRR = False Rejection Rate.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    far = fpr
    frr = 1.0 - tpr
    return far, frr


def compute_mean_det(
    y_true_per_class: list[np.ndarray],
    scores_per_class: list[np.ndarray],
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Average DET curves across multiple keyword classes.

    Interpolates each class's DET curve to common FAR points, then averages FRR.

    Args:
        y_true_per_class: List of binary label arrays, one per class.
        scores_per_class: List of score arrays, one per class.
        n_points: Number of interpolation points.

    Returns:
        (mean_far, mean_frr) arrays.
    """
    common_far = np.linspace(0, 1, n_points)
    frr_interpolated = []

    for y_true, scores in zip(y_true_per_class, scores_per_class):
        if len(np.unique(y_true)) < 2:
            continue
        far, frr = compute_det_curve(y_true, scores)
        frr_interp = np.interp(common_far, far, frr)
        frr_interpolated.append(frr_interp)

    if not frr_interpolated:
        return common_far, np.ones_like(common_far)

    mean_frr = np.mean(frr_interpolated, axis=0)
    return common_far, mean_frr


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Args:
        y_true: Binary labels (1=keyword, 0=non-keyword).
        scores: Confidence scores (higher = more likely keyword).

    Returns:
        AUC value in [0, 1].
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, scores))


def compute_acc_at_far(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    target_far: float = 0.05,
) -> float:
    """Classification accuracy at a specified FAR operating point.

    Finds the threshold where FAR <= target_far, then computes accuracy
    of predictions at that threshold.

    Args:
        y_true: True binary labels.
        y_pred: Predicted class labels (integer).
        scores: Confidence scores for threshold determination.
        target_far: Target FAR (default 5%).

    Returns:
        Accuracy at the operating point.
    """
    far, frr = compute_det_curve(y_true, scores)

    valid = far <= target_far
    if not np.any(valid):
        return 0.0

    idx = np.where(valid)[0][-1]

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    if idx >= len(thresholds):
        idx = len(thresholds) - 1
    threshold = thresholds[idx]

    accepted = scores >= threshold
    correct = (y_pred == y_true) & accepted
    rejected_correct = (~accepted) & (y_true == 0)

    total_correct = np.sum(correct) + np.sum(rejected_correct)
    return float(total_correct / len(y_true))


def compute_frr_at_far(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_far: float = 0.05,
) -> float:
    """False Rejection Rate at a specified FAR operating point.

    Args:
        y_true: Binary labels (1=keyword, 0=non-keyword).
        scores: Confidence scores.
        target_far: Target FAR (default 5%).

    Returns:
        FRR value at the operating point.
    """
    far, frr = compute_det_curve(y_true, scores)

    valid = far <= target_far
    if not np.any(valid):
        return 1.0

    idx = np.where(valid)[0][-1]
    return float(frr[idx])


def plot_det_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None,
    title: str = "Detection Error Tradeoff (DET) Curve",
) -> None:
    """Plot multiple DET curves on log-scale axes.

    Args:
        curves: Dict mapping label to (far, frr) arrays.
        save_path: If provided, save figure to this path.
        title: Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for label, (far, frr) in curves.items():
        mask = (far > 0) & (frr > 0)
        ax.plot(far[mask] * 100, frr[mask] * 100, label=label, linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14)

    tick_values = [1, 2, 5, 10, 20, 50]
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    ax.set_xticklabels([str(v) for v in tick_values])
    ax.set_yticklabels([str(v) for v in tick_values])

    ax.set_xlim(0.5, 50)
    ax.set_ylim(0.5, 50)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
