"""Tests for OpenMAXClassifier (Weibull tail prototypical OpenMAX)."""

import numpy as np
import pytest
import torch

from src.classifiers.openmax import OpenMAXClassifier


def _support_distances(prototypes: torch.Tensor, labels: list[str], jitter: float = 0.05, k: int = 8, seed: int = 0) -> dict[str, list[float]]:
    """Synthesize per-class support-to-prototype distances from jittered protos."""
    rng = np.random.default_rng(seed)
    distances: dict[str, list[float]] = {}
    for i, label in enumerate(labels):
        proto = prototypes[i].numpy()
        sup = proto + rng.normal(scale=jitter, size=(k, proto.size))
        d = np.linalg.norm(sup - proto, axis=1)
        distances[label] = d.tolist()
    return distances


def test_known_keyword():
    clf = OpenMAXClassifier(threshold=0.1, tail_size=4)
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    clf.fit_weibull(_support_distances(protos, ["yes", "no"]))

    pred, score = clf.predict(torch.tensor([0.99, 0.01, 0.0]))
    assert pred == "yes"
    assert 0.0 <= score <= 1.0


def test_unknown_rejection_far_query():
    clf = OpenMAXClassifier(threshold=0.5, tail_size=4)
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    clf.fit_weibull(_support_distances(protos, ["yes", "no"], jitter=0.02))

    pred, _ = clf.predict(torch.tensor([5.0, 5.0, 5.0]))
    assert pred == "unknown"


def test_predict_batch():
    clf = OpenMAXClassifier(threshold=0.1, tail_size=4)
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    clf.fit_weibull(_support_distances(protos, ["yes", "no"]))

    queries = torch.tensor([[0.95, 0.05, 0.0], [0.05, 0.95, 0.0]])
    results = clf.predict_batch(queries)
    assert len(results) == 2
    assert results[0][0] == "yes"
    assert results[1][0] == "no"


def test_get_distances_matches_l2():
    clf = OpenMAXClassifier(threshold=0.1, tail_size=4)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    dists = clf.get_distances(torch.tensor([0.9, 0.1]))
    assert dists["yes"] < dists["no"]
    expected_yes = float(torch.dist(torch.tensor([0.9, 0.1]), protos[0]).item())
    assert abs(dists["yes"] - expected_yes) < 1e-6


def test_get_scores_decay_with_distance():
    clf = OpenMAXClassifier(threshold=0.0, tail_size=4)
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    clf.fit_weibull(_support_distances(protos, ["yes", "no"], jitter=0.02))

    near = clf.get_scores(torch.tensor([0.99, 0.01, 0.0]))["yes"]
    far = clf.get_scores(torch.tensor([3.0, 0.0, 0.0]))["yes"]
    assert near > far
    assert 0.0 <= far <= near <= 1.0


def test_no_prototypes_error():
    clf = OpenMAXClassifier(threshold=0.5, tail_size=4)
    with pytest.raises(RuntimeError):
        clf.get_distances(torch.tensor([0.5, 0.5]))


def test_no_weibull_error():
    clf = OpenMAXClassifier(threshold=0.5, tail_size=4)
    protos = torch.tensor([[1.0, 0.0]])
    clf.set_prototypes(protos, ["yes"])
    with pytest.raises(RuntimeError):
        clf.get_scores(torch.tensor([0.9, 0.1]))


def test_no_threshold_error():
    clf = OpenMAXClassifier(threshold=None, tail_size=4)
    protos = torch.tensor([[1.0, 0.0]])
    clf.set_prototypes(protos, ["yes"])
    clf.fit_weibull({"yes": [0.05, 0.06, 0.07, 0.08]})
    with pytest.raises(RuntimeError):
        clf.predict(torch.tensor([0.9, 0.1]))


def test_calibrate_sets_threshold_in_unit_interval():
    clf = OpenMAXClassifier(tail_size=4)
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    clf.fit_weibull(_support_distances(protos, ["yes", "no"], jitter=0.02))

    val_emb = torch.tensor([
        [0.95, 0.05, 0.0],
        [0.05, 0.95, 0.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
    ])
    val_labels = ["yes", "no", "unknown", "unknown", "unknown"]
    threshold = clf.calibrate(val_emb, val_labels, target_far=0.05)

    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0
    assert clf.threshold == threshold


def test_calibrate_then_predict_rejects_unknown():
    clf = OpenMAXClassifier(tail_size=4)
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    clf.fit_weibull(_support_distances(protos, ["yes", "no"], jitter=0.02))

    val_emb = torch.tensor([
        [0.95, 0.05, 0.0], [0.05, 0.95, 0.0],
        [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0],
    ])
    val_labels = ["yes", "no", "unknown", "unknown", "unknown"]
    clf.calibrate(val_emb, val_labels, target_far=0.05)

    pred_known, _ = clf.predict(torch.tensor([0.97, 0.03, 0.0]))
    assert pred_known == "yes"
    pred_unknown, _ = clf.predict(torch.tensor([7.0, 7.0, 7.0]))
    assert pred_unknown == "unknown"


def test_prototype_label_mismatch():
    clf = OpenMAXClassifier(threshold=0.5, tail_size=4)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        clf.set_prototypes(protos, ["yes"])


def test_276dim_embeddings_smoke():
    """End-to-end smoke test on realistic 276-dim embeddings."""
    torch.manual_seed(0)
    protos = torch.randn(5, 276)
    labels = ["yes", "no", "stop", "go", "up"]
    clf = OpenMAXClassifier(tail_size=10)
    clf.set_prototypes(protos, labels)
    clf.fit_weibull(_support_distances(protos, labels, jitter=0.1, k=12))
    clf.threshold = 0.05

    query = protos[0] + torch.randn(276) * 0.01
    pred, _ = clf.predict(query)
    assert pred == "yes"
