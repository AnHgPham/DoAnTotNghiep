"""Tests for EnergyOODClassifier (Energy-based OOD on prototypes)."""

import math

import pytest
import torch

from src.classifiers.energy_ood import EnergyOODClassifier


def test_known_keyword():
    clf = EnergyOODClassifier(threshold=-2.0, temperature=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    pred, energy = clf.predict(torch.tensor([0.99, 0.01]))
    assert pred == "yes"
    assert isinstance(energy, float)


def test_unknown_rejection_far_query():
    clf = EnergyOODClassifier(threshold=-1.0, temperature=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    pred, _ = clf.predict(torch.tensor([5.0, 5.0]))
    assert pred == "unknown"


def test_predict_batch():
    clf = EnergyOODClassifier(threshold=-2.0, temperature=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    queries = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
    results = clf.predict_batch(queries)
    assert len(results) == 2
    assert results[0][0] == "yes"
    assert results[1][0] == "no"


def test_get_distances_matches_l2():
    clf = EnergyOODClassifier(threshold=-2.0, temperature=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    dists = clf.get_distances(torch.tensor([0.9, 0.1]))
    assert dists["yes"] < dists["no"]


def test_energy_decreases_with_distance():
    clf = EnergyOODClassifier(threshold=-2.0, temperature=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    near = clf.get_energy(torch.tensor([0.99, 0.01]))
    far = clf.get_energy(torch.tensor([5.0, 5.0]))
    assert near > far


def test_low_temperature_approaches_neg_min_distance():
    """As T -> 0, energy approximates -min(d). Verified at T=0.05."""
    clf = EnergyOODClassifier(threshold=-2.0, temperature=0.05)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    q = torch.tensor([0.5, 0.0])
    dists = clf.get_distances(q)
    energy = clf.get_energy(q)
    assert energy == pytest.approx(-min(dists.values()), abs=0.05)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        EnergyOODClassifier(temperature=0.0)
    with pytest.raises(ValueError):
        EnergyOODClassifier(temperature=-1.0)


def test_no_prototypes_error():
    clf = EnergyOODClassifier(threshold=-1.0)
    with pytest.raises(RuntimeError):
        clf.get_energy(torch.tensor([0.5, 0.5]))


def test_no_threshold_error():
    clf = EnergyOODClassifier(threshold=None)
    protos = torch.tensor([[1.0, 0.0]])
    clf.set_prototypes(protos, ["yes"])
    with pytest.raises(RuntimeError):
        clf.predict(torch.tensor([0.9, 0.1]))


def test_calibrate_then_predict_rejects_unknown():
    clf = EnergyOODClassifier(temperature=1.0)
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    clf.set_prototypes(protos, ["yes", "no"])

    val_emb = torch.tensor([
        [0.95, 0.05, 0.0], [0.05, 0.95, 0.0],
        [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0],
    ])
    val_labels = ["yes", "no", "unknown", "unknown", "unknown"]
    threshold = clf.calibrate(val_emb, val_labels, target_far=0.05)

    assert isinstance(threshold, float)
    assert math.isfinite(threshold)
    pred_known, _ = clf.predict(torch.tensor([0.97, 0.03, 0.0]))
    assert pred_known == "yes"
    pred_unknown, _ = clf.predict(torch.tensor([7.0, 7.0, 7.0]))
    assert pred_unknown == "unknown"


def test_prototype_label_mismatch():
    clf = EnergyOODClassifier(threshold=-1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        clf.set_prototypes(protos, ["yes"])


def test_276dim_smoke():
    torch.manual_seed(0)
    protos = torch.randn(5, 276)
    labels = ["yes", "no", "stop", "go", "up"]
    clf = EnergyOODClassifier(temperature=1.0, threshold=None)
    clf.set_prototypes(protos, labels)

    # Calibrate on synthetic val: knowns near each proto, unknowns random.
    val_emb = torch.cat([
        protos + torch.randn_like(protos) * 0.01,
        torch.randn(10, 276) * 5,  # far OOD
    ])
    val_labels = labels + ["unknown"] * 10
    clf.calibrate(val_emb, val_labels, target_far=0.05)

    query_known = protos[0] + torch.randn(276) * 0.01
    pred, _ = clf.predict(query_known)
    assert pred == "yes"
