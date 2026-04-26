"""Tests for EER and Precision/Recall/F1 metrics."""

import numpy as np
import pytest


class TestEER:
    """Tests for compute_eer."""

    def test_perfect_separation(self):
        from src.evaluation.metrics import compute_eer
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        eer, thr = compute_eer(y_true, scores)
        assert eer < 0.1  # near-perfect separation -> low EER

    def test_random_scores(self):
        from src.evaluation.metrics import compute_eer
        np.random.seed(42)
        y_true = np.array([1] * 50 + [0] * 50)
        scores = np.random.rand(100)
        eer, thr = compute_eer(y_true, scores)
        assert 0.0 <= eer <= 1.0
        assert isinstance(thr, float)

    def test_single_class(self):
        from src.evaluation.metrics import compute_eer
        y_true = np.array([1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7])
        eer, thr = compute_eer(y_true, scores)
        assert eer == 1.0  # degenerate case

    def test_eer_symmetry(self):
        from src.evaluation.metrics import compute_eer
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.6, 0.7, 0.3, 0.4])
        eer, _ = compute_eer(y_true, scores)
        assert 0.0 <= eer <= 0.5


class TestPrecisionRecallF1:
    """Tests for compute_precision_recall_f1."""

    def test_perfect_prediction(self):
        from src.evaluation.metrics import compute_precision_recall_f1
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        result = compute_precision_recall_f1(y_true, y_pred)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self):
        from src.evaluation.metrics import compute_precision_recall_f1
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_precision_recall_f1(y_true, y_pred)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_partial_correct(self):
        from src.evaluation.metrics import compute_precision_recall_f1
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        result = compute_precision_recall_f1(y_true, y_pred)
        assert 0.0 < result["precision"] < 1.0
        assert 0.0 < result["recall"] < 1.0
        assert 0.0 < result["f1"] < 1.0

    def test_keys_present(self):
        from src.evaluation.metrics import compute_precision_recall_f1
        y_true = np.array([1, 0])
        y_pred = np.array([1, 0])
        result = compute_precision_recall_f1(y_true, y_pred)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
