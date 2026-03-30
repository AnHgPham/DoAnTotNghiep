"""Tests for OpenNCMClassifier."""

import torch

from src.classifiers.open_ncm import OpenNCMClassifier


def test_known_keyword():
    clf = OpenNCMClassifier(threshold=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    pred, dist = clf.predict(torch.tensor([0.9, 0.1]))
    assert pred == "yes"


def test_unknown_rejection():
    clf = OpenNCMClassifier(threshold=0.5)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    pred, dist = clf.predict(torch.tensor([0.5, 0.5]))
    assert pred == "unknown"


def test_predict_batch():
    clf = OpenNCMClassifier(threshold=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    queries = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    results = clf.predict_batch(queries)
    assert len(results) == 3
    assert results[0][0] == "yes"
    assert results[1][0] == "no"


def test_get_distances():
    clf = OpenNCMClassifier(threshold=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    dists = clf.get_distances(torch.tensor([0.9, 0.1]))
    assert "yes" in dists
    assert "no" in dists
    assert dists["yes"] < dists["no"]


def test_no_prototypes_error():
    clf = OpenNCMClassifier(threshold=1.0)
    try:
        clf.predict(torch.tensor([0.5, 0.5]))
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass


def test_no_threshold_error():
    clf = OpenNCMClassifier(threshold=None)
    protos = torch.tensor([[1.0, 0.0]])
    clf.set_prototypes(protos, ["yes"])
    try:
        clf.predict(torch.tensor([0.5, 0.5]))
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass


def test_calibrate():
    clf = OpenNCMClassifier()
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])

    val_emb = torch.tensor([
        [0.9, 0.1, 0.0],   # close to "yes"
        [0.1, 0.9, 0.0],   # close to "no"
        [0.3, 0.3, 0.8],   # unknown (far from both)
        [0.2, 0.2, 0.9],   # unknown
        [0.4, 0.4, 0.7],   # unknown
    ])
    val_labels = ["yes", "no", "unknown", "unknown", "unknown"]

    threshold = clf.calibrate(val_emb, val_labels, target_far=0.05)
    assert isinstance(threshold, float)
    assert threshold > 0
    assert clf.threshold == threshold


def test_calibrate_then_predict():
    clf = OpenNCMClassifier()
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])

    val_emb = torch.tensor([
        [0.9, 0.1], [0.1, 0.9],
        [5.0, 5.0], [6.0, 6.0], [7.0, 7.0],
    ])
    val_labels = ["yes", "no", "unknown", "unknown", "unknown"]
    clf.calibrate(val_emb, val_labels)

    pred_known, _ = clf.predict(torch.tensor([0.95, 0.05]))
    assert pred_known == "yes"


def test_prototype_label_mismatch():
    clf = OpenNCMClassifier(threshold=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    try:
        clf.set_prototypes(protos, ["yes"])  # 2 protos, 1 label
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_276dim_embeddings():
    clf = OpenNCMClassifier(threshold=2.0)
    protos = torch.randn(5, 276)
    labels = ["yes", "no", "stop", "go", "up"]
    clf.set_prototypes(protos, labels)

    query = protos[0] + torch.randn(276) * 0.01  # close to "yes"
    pred, dist = clf.predict(query)
    assert pred == "yes"
    assert dist < 2.0
