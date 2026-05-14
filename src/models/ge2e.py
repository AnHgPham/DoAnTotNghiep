"""GE2E loss for few-shot keyword spotting.

The loss mirrors deployment: each class in an episode is split into enrollment
support samples and query samples; query embeddings are classified by cosine
similarity to support centroids.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    """Generalized end-to-end centroid loss.

    Args:
        init_scale: Initial positive scale for cosine logits.
        init_bias: Initial bias for cosine logits.
        support_fraction: Fraction of each class episode used to form centroids.
    """

    def __init__(
        self,
        init_scale: float = 10.0,
        init_bias: float = -5.0,
        support_fraction: float = 0.5,
    ):
        super().__init__()
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(init_scale))))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.support_fraction = float(support_fraction)
        self.last_stats: dict[str, float] = {}

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute GE2E centroid classification loss.

        Args:
            embeddings: L2-normalized embeddings, shape ``(B, D)``.
            labels: Integer labels, shape ``(B,)``.
        """
        unique = labels.unique(sorted=True)
        centroids = []
        query_embs = []
        query_targets = []
        support_counts = []
        query_counts = []

        for class_idx, label in enumerate(unique):
            class_embs = embeddings[labels == label]
            if class_embs.shape[0] < 2:
                continue
            n_support = int(round(class_embs.shape[0] * self.support_fraction))
            n_support = max(1, min(n_support, class_embs.shape[0] - 1))
            support = class_embs[:n_support]
            query = class_embs[n_support:]
            centroids.append(F.normalize(support.mean(dim=0), p=2, dim=0))
            query_embs.append(query)
            query_targets.extend([len(centroids) - 1] * query.shape[0])
            support_counts.append(n_support)
            query_counts.append(query.shape[0])

        if not centroids or not query_embs:
            self.last_stats = {"n_classes": 0.0, "n_queries": 0.0, "acc": 0.0}
            return embeddings.sum() * 0.0

        centroid_tensor = torch.stack(centroids)
        query_tensor = torch.cat(query_embs, dim=0)
        targets = torch.tensor(query_targets, dtype=torch.long, device=embeddings.device)
        scale = self.log_scale.exp().clamp(min=1e-3, max=100.0)
        logits = scale * F.linear(query_tensor, centroid_tensor) + self.bias
        loss = F.cross_entropy(logits, targets)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == targets).float().mean().item()
            self.last_stats = {
                "n_classes": float(len(centroids)),
                "n_queries": float(query_tensor.shape[0]),
                "support_per_class": float(sum(support_counts) / max(len(support_counts), 1)),
                "query_per_class": float(sum(query_counts) / max(len(query_counts), 1)),
                "acc": float(acc),
                "scale": float(scale.item()),
                "bias": float(self.bias.item()),
            }
        return loss
