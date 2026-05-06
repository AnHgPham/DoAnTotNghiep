"""Energy-based out-of-distribution detector for prototypical KWS.

Adapts Liu et al. (NeurIPS 2020) "Energy-based Out-of-distribution Detection"
to a prototypical network with L2-normalized embeddings:

  * "Logits" are taken as the negative L2 distances ``-d_i`` to each class
    prototype (closer = larger logit, matching softmax-cross-entropy semantics).
  * In-distribution score is ``E(x) = T * logsumexp(-d_i / T)`` over all
    classes; larger means the query lies in a high-density region of the
    enrolled keyword space.
  * Classification keeps the OpenNCM rule (argmin distance), so accuracy
    is preserved; energy only drives the open-set rejection score.

Compared to OpenMAX, energy needs no Weibull fitting and stays robust in
few-shot regimes — it has a single hyperparameter (temperature ``T``) and
uses every prototype's distance, not only the support tail.
"""

import numpy as np
import torch


class EnergyOODClassifier:
    """Prototypical Energy-based OOD classifier.

    Args:
        threshold: Energy cutoff for open-set rejection. Queries with
            energy below this threshold are classified as ``'unknown'``.
            If ``None``, must be set via :meth:`calibrate` first.
        temperature: Softmin temperature ``T``. Lower ``T`` makes the
            energy approximate ``-min(d)`` (raw L2 baseline); larger ``T``
            blends contributions from all prototypes.
    """

    def __init__(self, threshold: float | None = None, temperature: float = 1.0):
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.threshold = threshold
        self.temperature = temperature
        self.prototypes: torch.Tensor | None = None
        self.labels: list[str] = []

    def set_prototypes(
        self, prototypes: torch.Tensor, labels: list[str]
    ) -> None:
        """Register keyword prototypes."""
        if prototypes.shape[0] != len(labels):
            raise ValueError(
                f"prototypes ({prototypes.shape[0]}) and labels ({len(labels)}) "
                f"count mismatch"
            )
        self.prototypes = prototypes.detach().clone()
        self.labels = list(labels)

    def get_distances(
        self, query_embedding: torch.Tensor
    ) -> dict[str, float]:
        """Return L2 distances from query to all prototypes (drop-in API)."""
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

    def get_energy(self, query_embedding: torch.Tensor) -> float:
        """Energy score ``T * logsumexp(-d_i / T)``. Higher = more in-distribution."""
        if self.prototypes is None:
            raise RuntimeError("No prototypes set. Call set_prototypes() first.")
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.squeeze(0)
        d = torch.cdist(
            query_embedding.unsqueeze(0), self.prototypes
        ).squeeze(0)
        logits = -d / self.temperature
        energy = self.temperature * torch.logsumexp(logits, dim=0)
        return float(energy.item())

    def predict(
        self, query_embedding: torch.Tensor
    ) -> tuple[str, float]:
        """Predict keyword.

        Classification: argmin L2 distance (matches OpenNCM).
        Rejection: energy below ``self.threshold`` returns ``'unknown'``.

        Returns:
            ``(label, energy)``.
        """
        if self.threshold is None:
            raise RuntimeError("No threshold set. Call calibrate() first.")
        dists = self.get_distances(query_embedding)
        pred_label = min(dists, key=dists.get)
        energy = self.get_energy(query_embedding)
        if energy < self.threshold:
            return ("unknown", energy)
        return (pred_label, energy)

    def predict_batch(
        self, query_embeddings: torch.Tensor
    ) -> list[tuple[str, float]]:
        """Batch prediction."""
        return [self.predict(query_embeddings[i]) for i in range(query_embeddings.shape[0])]

    def calibrate(
        self,
        val_embeddings: torch.Tensor,
        val_labels: list[str],
        target_far: float = 0.05,
    ) -> float:
        """Find energy threshold achieving target FAR on unknowns.

        Args:
            val_embeddings: ``(M, embedding_dim)`` validation embeddings.
            val_labels: M labels; ``'unknown'`` entries are negative samples.
            target_far: Desired FAR (default 5%).

        Returns:
            Calibrated energy threshold.
        """
        unknown_mask = np.array([l == "unknown" for l in val_labels])
        if not unknown_mask.any():
            self.threshold = float("-inf")
            return self.threshold

        energies = np.asarray(
            [self.get_energy(val_embeddings[i]) for i in range(val_embeddings.shape[0])],
            dtype=float,
        )
        unknown_e_sorted = np.sort(energies[unknown_mask])[::-1]  # descending
        idx = int(len(unknown_e_sorted) * target_far)
        idx = max(0, min(idx, len(unknown_e_sorted) - 1))
        # Tiny offset so an unknown sitting exactly at the quantile is rejected.
        self.threshold = float(unknown_e_sorted[idx]) + 1e-6
        return self.threshold
