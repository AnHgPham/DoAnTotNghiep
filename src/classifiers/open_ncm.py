"""Open Nearest Class Mean classifier with Direct L2 distance.

Uses raw L2 distances (NOT softmax probability) for classification.
Threshold acts as an acceptance radius around each prototype.
"""

import torch
import numpy as np


class OpenNCMClassifier:
    """Open-set keyword classifier using prototype distances.

    Args:
        threshold: L2 distance cutoff for open-set rejection.
            If None, must be set via calibrate() before prediction.
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold
        self.prototypes: torch.Tensor | None = None
        self.labels: list[str] = []
        self.thresholds_per_prototype: dict[str, float] | None = None

    def set_per_prototype_thresholds(self, thresholds: dict[str, float]) -> None:
        """Register per-prototype acceptance radii (overrides global threshold).

        Args:
            thresholds: Mapping from label to L2 distance cutoff.
        """
        if self.prototypes is None:
            raise RuntimeError("No prototypes set. Call set_prototypes() first.")
        missing = [label for label in self.labels if label not in thresholds]
        if missing:
            raise ValueError(f"Missing thresholds for labels: {missing}")
        self.thresholds_per_prototype = {label: float(thresholds[label]) for label in self.labels}

    def _threshold_for(self, label: str) -> float:
        if self.thresholds_per_prototype is not None:
            return self.thresholds_per_prototype[label]
        if self.threshold is None:
            raise RuntimeError("No threshold set. Call calibrate() first.")
        return float(self.threshold)

    def set_prototypes(
        self, prototypes: torch.Tensor, labels: list[str]
    ) -> None:
        """Register keyword prototypes.

        Args:
            prototypes: (N, embedding_dim) prototype embeddings.
            labels: List of N keyword names corresponding to each prototype.
        """
        if prototypes.shape[0] != len(labels):
            raise ValueError(
                f"prototypes ({prototypes.shape[0]}) and labels ({len(labels)}) "
                f"count mismatch"
            )
        self.prototypes = prototypes.detach().clone()
        self.labels = list(labels)

    def predict(
        self, query_embedding: torch.Tensor
    ) -> tuple[str, float]:
        """Predict keyword for a single query.

        Args:
            query_embedding: (embedding_dim,) or (1, embedding_dim) tensor.

        Returns:
            (predicted_keyword, l2_distance).
            Returns ('unknown', min_distance) if min_distance > threshold.
        """
        if self.prototypes is None:
            raise RuntimeError("No prototypes set. Call set_prototypes() first.")
        if self.threshold is None and self.thresholds_per_prototype is None:
            raise RuntimeError("No threshold set. Call calibrate() first.")

        if query_embedding.dim() == 2:
            query_embedding = query_embedding.squeeze(0)

        distances = torch.cdist(
            query_embedding.unsqueeze(0), self.prototypes
        ).squeeze(0)  # (N,)

        min_dist, min_idx = distances.min(dim=0)
        predicted_label = self.labels[min_idx.item()]
        threshold = self._threshold_for(predicted_label)

        if min_dist.item() > threshold:
            return ("unknown", min_dist.item())
        return (predicted_label, min_dist.item())

    def predict_batch(
        self, query_embeddings: torch.Tensor
    ) -> list[tuple[str, float]]:
        """Predict keywords for a batch of queries.

        Args:
            query_embeddings: (B, embedding_dim) tensor.

        Returns:
            List of (predicted_keyword, l2_distance) tuples.
        """
        results = []
        for i in range(query_embeddings.shape[0]):
            results.append(self.predict(query_embeddings[i]))
        return results

    def calibrate(
        self,
        val_embeddings: torch.Tensor,
        val_labels: list[str],
        target_far: float = 0.05,
    ) -> float:
        """Find threshold achieving target False Acceptance Rate.

        Uses validation embeddings to find the L2 distance threshold where
        FAR (rate of unknown samples accepted as known) equals target_far.

        Args:
            val_embeddings: (M, embedding_dim) validation embeddings.
            val_labels: List of M labels ('unknown' for negative samples).
            target_far: Target FAR (default 5%).

        Returns:
            Calibrated threshold value.
        """
        if self.prototypes is None:
            raise RuntimeError("No prototypes set. Call set_prototypes() first.")

        distances = torch.cdist(val_embeddings, self.prototypes)  # (M, N)
        min_dists = distances.min(dim=1).values  # (M,)

        unknown_mask = np.array([l == "unknown" for l in val_labels])
        unknown_dists = min_dists[unknown_mask].numpy()

        if len(unknown_dists) == 0:
            self.threshold = float(min_dists.max().item() * 1.1)
            return self.threshold

        unknown_dists_sorted = np.sort(unknown_dists)
        idx = int(len(unknown_dists_sorted) * target_far)
        idx = max(0, min(idx, len(unknown_dists_sorted) - 1))
        self.threshold = float(unknown_dists_sorted[idx])

        return self.threshold

    def get_distances(
        self, query_embedding: torch.Tensor
    ) -> dict[str, float]:
        """Get distances from query to all prototypes.

        Args:
            query_embedding: (embedding_dim,) tensor.

        Returns:
            Dict mapping keyword label to L2 distance.
        """
        if self.prototypes is None:
            raise RuntimeError("No prototypes set. Call set_prototypes() first.")

        if query_embedding.dim() == 2:
            query_embedding = query_embedding.squeeze(0)

        distances = torch.cdist(
            query_embedding.unsqueeze(0), self.prototypes
        ).squeeze(0)

        return {
            label: distances[i].item() for i, label in enumerate(self.labels)
        }
