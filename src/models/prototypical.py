"""Prototypical Network training with Triplet Loss.

Episodic training: sample N classes x K samples per batch, mine triplets,
optimize embedding space so same-class samples are close and different-class
samples are far apart.

Mining strategies:
    - "random": pick a random positive and random negative (weakest gradient)
    - "hard":   pick hardest negative (smallest d_neg) per anchor
    - "semi_hard": pick negative satisfying d_pos < d_neg < d_pos + margin
                   (FaceNet-style). Falls back to hard if no semi-hard exists.
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
    """Triplet loss with selectable negative mining.

    Loss = mean( max(0, d(anchor, positive) - d(anchor, negative) + margin) )

    Args:
        margin: Margin for triplet loss (recommended 1.0 for L2-normalized
            embeddings -- max possible L2 distance is 2.0).
        mining: One of "random", "hard", "semi_hard". For "semi_hard",
            anchors with no semi-hard negative fall back to the hardest
            negative (combined strategy).
    """

    VALID_MINING = ("random", "hard", "semi_hard")

    def __init__(self, margin: float = 1.0, mining: str = "semi_hard"):
        super().__init__()
        if mining not in self.VALID_MINING:
            raise ValueError(
                f"mining must be one of {self.VALID_MINING}, got {mining!r}"
            )
        self.margin = margin
        self.mining = mining
        # Stats from the most recent forward() pass (for logging).
        self.last_stats: dict[str, float] = {}

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

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_neq = ~labels_eq

        pos_mask = labels_eq.clone()
        pos_mask.fill_diagonal_(False)

        valid_anchor = pos_mask.any(dim=1) & labels_neq.any(dim=1)
        if not valid_anchor.any():
            zero = torch.zeros((), device=device, requires_grad=True)
            self.last_stats = {"n_active": 0.0, "n_semi_hard": 0.0, "n_hard_fallback": 0.0}
            return zero

        # Positive: hardest positive per anchor (largest d_pos same-class).
        # Hardest positive provides a stronger learning signal than a random one.
        NEG_INF = torch.tensor(float("-inf"), device=device)
        pos_dists = torch.where(pos_mask, dist_matrix, NEG_INF)
        d_pos, pos_idx = pos_dists.max(dim=1)

        # Negative mining
        POS_INF = torch.tensor(float("inf"), device=device)
        neg_dists_only = torch.where(labels_neq, dist_matrix, POS_INF)

        n_semi_hard = 0
        n_hard_fallback = 0

        if self.mining == "random":
            neg_scores = torch.where(
                labels_neq, torch.rand(N, N, device=device),
                torch.tensor(-1.0, device=device),
            )
            neg_idx = neg_scores.argmax(dim=1)

        elif self.mining == "hard":
            # Hardest negative = smallest d_neg
            _, neg_idx = neg_dists_only.min(dim=1)

        else:  # semi_hard
            # Semi-hard: d_pos < d_neg < d_pos + margin
            d_pos_expand = d_pos.unsqueeze(1)
            semi_hard_mask = (
                labels_neq
                & (dist_matrix > d_pos_expand)
                & (dist_matrix < d_pos_expand + self.margin)
            )
            has_semi_hard = semi_hard_mask.any(dim=1)
            n_semi_hard = int(has_semi_hard.sum().item())
            n_hard_fallback = int((~has_semi_hard & valid_anchor).sum().item())

            # Among semi-hard, pick the hardest one (smallest d_neg) for stable signal.
            semi_dists = torch.where(semi_hard_mask, dist_matrix, POS_INF)
            _, semi_idx = semi_dists.min(dim=1)

            # Fallback: hardest negative when no semi-hard exists
            _, hard_idx = neg_dists_only.min(dim=1)

            neg_idx = torch.where(has_semi_hard, semi_idx, hard_idx)

        d_neg = dist_matrix[torch.arange(N, device=device), neg_idx]

        losses = F.relu(d_pos - d_neg + self.margin)
        losses = losses[valid_anchor]
        n_active = int((losses > 0).sum().item())

        self.last_stats = {
            "n_active": float(n_active),
            "n_semi_hard": float(n_semi_hard),
            "n_hard_fallback": float(n_hard_fallback),
            "n_valid": float(int(valid_anchor.sum().item())),
            "mean_d_pos": float(d_pos[valid_anchor].mean().item()),
            "mean_d_neg": float(d_neg[valid_anchor].mean().item()),
        }

        if losses.numel() == 0:
            return torch.zeros((), device=device, requires_grad=True)
        return losses.mean()


def train_one_epoch(
    encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TripletLoss,
    device: torch.device = torch.device("cpu"),
    grad_clip: float = 0.0,
) -> dict:
    """Train encoder for one epoch of episodic batches.

    Args:
        encoder: DSCNN model.
        dataloader: DataLoader with EpisodicBatchSampler.
        optimizer: Adam optimizer.
        loss_fn: TripletLoss instance.
        device: Training device.
        grad_clip: If > 0, clip gradient L2-norm to this value.

    Returns:
        Dict with 'loss', 'num_episodes', and aggregated mining stats.
    """
    encoder.train()
    total_loss = 0.0
    num_episodes = 0
    sum_active = 0.0
    sum_semi_hard = 0.0
    sum_hard_fallback = 0.0
    sum_valid = 0.0
    sum_d_pos = 0.0
    sum_d_neg = 0.0
    sum_grad_norm = 0.0

    for batch_mfcc, batch_labels in dataloader:
        batch_mfcc = batch_mfcc.to(device)
        batch_labels = batch_labels.to(device)

        embeddings = encoder(batch_mfcc)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        loss = loss_fn(embeddings, batch_labels)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                encoder.parameters(), max_norm=grad_clip,
            )
            sum_grad_norm += float(grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_episodes += 1

        stats = getattr(loss_fn, "last_stats", {})
        sum_active += stats.get("n_active", 0.0)
        sum_semi_hard += stats.get("n_semi_hard", 0.0)
        sum_hard_fallback += stats.get("n_hard_fallback", 0.0)
        sum_valid += stats.get("n_valid", 0.0)
        sum_d_pos += stats.get("mean_d_pos", 0.0)
        sum_d_neg += stats.get("mean_d_neg", 0.0)

    n = max(num_episodes, 1)
    return {
        "loss": total_loss / n,
        "num_episodes": num_episodes,
        "active_per_ep": sum_active / n,
        "semi_hard_per_ep": sum_semi_hard / n,
        "hard_fallback_per_ep": sum_hard_fallback / n,
        "valid_per_ep": sum_valid / n,
        "mean_d_pos": sum_d_pos / n,
        "mean_d_neg": sum_d_neg / n,
        "mean_grad_norm": sum_grad_norm / n if grad_clip > 0 else 0.0,
    }
