"""Tests for evaluation metrics."""

import tempfile
from pathlib import Path

import numpy as np

from src.evaluation.metrics import (
    compute_auc,
    compute_det_curve,
    compute_frr_at_far,
    compute_mean_det,
    plot_det_curves,
)


def test_det_curve_shape():
    y_true = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.7, 0.3, 0.2, 0.8])
    far, frr = compute_det_curve(y_true, scores)
    assert len(far) == len(frr)
    assert far[0] <= far[-1]


def test_det_curve_perfect():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    far, frr = compute_det_curve(y_true, scores)
    assert far[0] == 0.0
    assert frr[-1] == 0.0


def test_auc_perfect():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    assert compute_auc(y_true, scores) == 1.0


def test_auc_random():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 1000)
    scores = rng.rand(1000)
    auc = compute_auc(y_true, scores)
    assert 0.4 < auc < 0.6  # near random


def test_auc_single_class():
    y_true = np.array([1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7])
    assert compute_auc(y_true, scores) == 0.0


def test_frr_at_far_perfect():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    frr = compute_frr_at_far(y_true, scores, target_far=0.05)
    assert frr == 0.0


def test_frr_at_far_random():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 500)
    scores = rng.rand(500)
    frr = compute_frr_at_far(y_true, scores, target_far=0.05)
    assert 0.0 <= frr <= 1.0


def test_mean_det():
    y1 = np.array([1, 1, 0, 0])
    s1 = np.array([0.9, 0.8, 0.1, 0.2])
    y2 = np.array([1, 1, 0, 0])
    s2 = np.array([0.7, 0.6, 0.3, 0.4])

    far, frr = compute_mean_det([y1, y2], [s1, s2])
    assert len(far) == 1000
    assert len(frr) == 1000
    assert far[0] == 0.0
    assert far[-1] == 1.0


def test_plot_det_curves():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    far, frr = compute_det_curve(y_true, scores)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "det.png"
        plot_det_curves({"test": (far, frr)}, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
