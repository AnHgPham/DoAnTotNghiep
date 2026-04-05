"""Prototypical Network training with Triplet Loss.

Episodic training: sample N classes x K samples per batch, mine triplets,
optimize embedding space so same-class samples are close and different-class
samples are far apart.
"""

import random
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class EpisodicBatchSampler(Sampler[list[int]]):
    """Sample episodic batches: N classes x K samples per class.

    Args:
        labels: Integer labels for the full dataset.
        n_classes: Number of classes per episode.
        n_samples: Number of samples per class per episode.
        n_episodes: Number of episodes per epoch.
    """

    def __init__(
        self,
        labels: list[int] | torch.Tensor,
        n_classes: int = 80,
        n_samples: int = 20,
        n_episodes: int = 400,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_episodes = n_episodes

        self.class_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            self.class_to_indices.setdefault(label, []).append(idx)

        self.available_classes = [
            c for c, indices in self.class_to_indices.items()
            if len(indices) >= n_samples
        ]

        if len(self.available_classes) < n_classes:
            raise ValueError(
                f"Need at least {n_classes} classes with >={n_samples} samples each, "
                f"but only {len(self.available_classes)} classes qualify."
            )

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.n_episodes):
            selected_classes = random.sample(self.available_classes, self.n_classes)
            batch_indices = []
            for cls in selected_classes:
                indices = random.sample(self.class_to_indices[cls], self.n_samples)
                batch_indices.extend(indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.n_episodes


class TripletLoss(nn.Module):
    """Triplet loss with random negative mining.

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Args:
        margin: Margin for triplet loss.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss over a batch (vectorized).

        Args:
            embeddings: (N, D) L2-normalized embeddings.
            labels: (N,) integer class labels.

        Returns:
            Scalar loss value.
        """
        device = embeddings.device
        N = embeddings.shape[0]

        # Pairwise distance matrix: (N, N)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Masks: (N, N)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # same class
        labels_neq = ~labels_eq  # different class

        # For each anchor i, pick random positive and random negative
        # Positive: same class, different index
        pos_mask = labels_eq.clone()
        pos_mask.fill_diagonal_(False)  # exclude self

        # Check valid anchors (at least 1 positive and 1 negative)
        valid_anchor = pos_mask.any(dim=1) & labels_neq.any(dim=1)
        if not valid_anchor.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Random positive: sample one per anchor using Gumbel trick
        pos_scores = torch.where(pos_mask, torch.rand(N, N, device=device), torch.tensor(-1.0, device=device))
        pos_idx = pos_scores.argmax(dim=1)  # random positive index

        neg_scores = torch.where(labels_neq, torch.rand(N, N, device=device), torch.tensor(-1.0, device=device))
        neg_idx = neg_scores.argmax(dim=1)  # random negative index

        d_pos = dist_matrix[torch.arange(N, device=device), pos_idx]
        d_neg = dist_matrix[torch.arange(N, device=device), neg_idx]

        losses = F.relu(d_pos - d_neg + self.margin)
        losses = losses[valid_anchor]

        return losses.mean() if losses.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)


def train_one_epoch(
    encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TripletLoss,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train encoder for one epoch of episodic batches.

    Args:
        encoder: DSCNN model.
        dataloader: DataLoader with EpisodicBatchSampler.
        optimizer: Adam optimizer.
        loss_fn: TripletLoss instance.
        device: Training device.

    Returns:
        Dict with 'loss' (mean) and 'num_episodes'.
    """
    encoder.train()
    total_loss = 0.0
    num_episodes = 0

    for batch_mfcc, batch_labels in dataloader:
        batch_mfcc = batch_mfcc.to(device)
        batch_labels = batch_labels.to(device)

        embeddings = encoder(batch_mfcc)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        loss = loss_fn(embeddings, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_episodes += 1

    mean_loss = total_loss / max(num_episodes, 1)
    return {"loss": mean_loss, "num_episodes": num_episodes}
