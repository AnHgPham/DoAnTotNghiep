"""Tests for prototypical training components."""

import torch

from src.models.prototypical import EpisodicBatchSampler, TripletLoss


def _make_labels(n_classes: int = 100, n_per_class: int = 50) -> list[int]:
    """Create a label list with n_classes classes, each having n_per_class samples."""
    labels = []
    for c in range(n_classes):
        labels.extend([c] * n_per_class)
    return labels


def test_episodic_sampler_length():
    labels = _make_labels(100, 50)
    sampler = EpisodicBatchSampler(labels, n_classes=80, n_samples=20, n_episodes=10)
    assert len(sampler) == 10


def test_episodic_sampler_batch_size():
    labels = _make_labels(100, 50)
    sampler = EpisodicBatchSampler(labels, n_classes=80, n_samples=20, n_episodes=5)
    for batch in sampler:
        assert len(batch) == 80 * 20


def test_episodic_sampler_class_diversity():
    labels = _make_labels(100, 50)
    sampler = EpisodicBatchSampler(labels, n_classes=80, n_samples=20, n_episodes=3)
    for batch in sampler:
        batch_labels = [labels[i] for i in batch]
        unique = set(batch_labels)
        assert len(unique) == 80


def test_episodic_sampler_insufficient_classes():
    labels = _make_labels(10, 50)
    try:
        EpisodicBatchSampler(labels, n_classes=80, n_samples=20)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_triplet_loss_basic():
    loss_fn = TripletLoss(margin=0.5)
    embeddings = torch.randn(20, 276)
    labels = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    loss = loss_fn(embeddings, labels)
    assert loss.item() >= 0


def test_triplet_loss_perfect_separation():
    loss_fn = TripletLoss(margin=0.5)
    e0 = torch.zeros(5, 4)
    e0[:, 0] = 10.0
    e1 = torch.zeros(5, 4)
    e1[:, 1] = 10.0
    embeddings = torch.cat([e0, e1], dim=0)
    labels = torch.tensor([0] * 5 + [1] * 5)
    loss = loss_fn(embeddings, labels)
    assert loss.item() == 0.0  # well-separated clusters -> zero loss


def test_triplet_loss_zero_margin():
    loss_fn = TripletLoss(margin=0.0)
    embeddings = torch.randn(10, 276)
    labels = torch.tensor([0] * 5 + [1] * 5)
    loss = loss_fn(embeddings, labels)
    assert loss.item() >= 0


def test_triplet_loss_requires_grad():
    loss_fn = TripletLoss(margin=0.5)
    embeddings = torch.randn(20, 276, requires_grad=True)
    labels = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    loss = loss_fn(embeddings, labels)
    loss.backward()
    assert embeddings.grad is not None
