"""OpenMAX classifier with Weibull tail fitting on prototype distances.

Adapts Bendale & Boult (CVPR 2016) "Towards Open Set Deep Networks" to a
prototypical-network setting:

  * Each class c has a prototype p_c (mean of support embeddings).
  * Fit a Weibull_min distribution to the *tail* (largest distances) of
    support-to-prototype distances per class.
  * At inference, score_c(x) = 1 - W_cdf(d(x, p_c); params_c) lies in [0, 1]
    and decays as the query falls outside the calibrated boundary.
  * predict() returns argmax score; rejects to 'unknown' when max < threshold.

The classifier mirrors :class:`OpenNCMClassifier` for the methods used by
``src.evaluation.protocols.EvaluationProtocol`` (``set_prototypes``,
``get_distances``, ``threshold``) so it can be slotted in without changing
the protocol's prototype-management code. OpenMAX-specific scoring is
exposed via :meth:`get_scores`.
"""

import numpy as np
import torch
from scipy.stats import weibull_min


class OpenMAXClassifier:
    """Prototypical OpenMAX classifier.

    Args:
        threshold: Score cutoff in [0, 1] for open-set rejection. Queries
            whose maximum Weibull-revised score is below this threshold are
            classified as ``'unknown'``. If ``None``, must be set via
            :meth:`calibrate` before prediction.
        tail_size: Number of largest support distances per class used to fit
            the Weibull tail. Effective value is ``min(tail_size, k_shot)``.
    """

    def __init__(
        self,
        threshold: float | None = None,
        tail_size: int = 20,
        mode: str = "per_class",
        hybrid_alpha: float = 0.0,
    ):
        if tail_size < 2:
            raise ValueError("tail_size must be >= 2 for Weibull fitting")
        if mode not in ("per_class", "global"):
            raise ValueError("mode must be 'per_class' or 'global'")
        if not (0.0 <= hybrid_alpha <= 1.0):
            raise ValueError("hybrid_alpha must be in [0, 1]")
        self.threshold = threshold
        self.tail_size = tail_size
        self.mode = mode
        self.hybrid_alpha = hybrid_alpha
        self.prototypes: torch.Tensor | None = None
        self.labels: list[str] = []
        self._weibull_params: dict[str, tuple[float, float, float]] = {}

    def set_prototypes(
        self, prototypes: torch.Tensor, labels: list[str]
    ) -> None:
        """Register keyword prototypes.

        Args:
            prototypes: ``(N, embedding_dim)`` prototype embeddings.
            labels: List of N keyword names.
        """
        if prototypes.shape[0] != len(labels):
            raise ValueError(
                f"prototypes ({prototypes.shape[0]}) and labels ({len(labels)}) "
                f"count mismatch"
            )
        self.prototypes = prototypes.detach().clone()
        self.labels = list(labels)
        self._weibull_params = {}

    def fit_weibull(self, support_distances: dict[str, list[float]]) -> None:
        """Fit Weibull tail per class on support-to-prototype distances.

        Args:
            support_distances: Mapping ``label -> [d_1, ..., d_k]`` where
                ``d_i`` is the L2 distance from support sample i to that
                class's prototype.
        """
        if self.prototypes is None:
            raise RuntimeError("No prototypes set. Call set_prototypes() first.")
        missing = [label for label in self.labels if label not in support_distances]
        if missing:
            raise ValueError(f"Missing support distances for labels: {missing}")

        params: dict[str, tuple[float, float, float]] = {}
        if self.mode == "global":
            pooled = np.concatenate([
                np.asarray(support_distances[label], dtype=float)
                for label in self.labels
            ])
            tail_n = min(self.tail_size, pooled.size)
            tail = np.sort(pooled)[-tail_n:]
            if tail.size < 2 or np.allclose(tail, tail[0]):
                shared = (1.0, float(tail[0]) if tail.size else 0.0, 1e-3)
            else:
                shape, loc, scale = weibull_min.fit(tail, floc=0.0)
                shared = (float(shape), float(loc), float(max(scale, 1e-6)))
            for label in self.labels:
                params[label] = shared
            self._weibull_params = params
            return

        for label in self.labels:
            dists = np.asarray(support_distances[label], dtype=float)
            if dists.size < 2:
                # Degenerate: not enough samples to fit. Fall back to a
                # delta around the single observation; downstream score
                # then behaves like a hard threshold at that distance.
                d0 = float(dists[0]) if dists.size == 1 else 0.0
                params[label] = (1.0, d0, 1e-3)
                continue
            tail_n = min(self.tail_size, dists.size)
            tail = np.sort(dists)[-tail_n:]
            if np.allclose(tail, tail[0]):
                params[label] = (1.0, float(tail[0]), 1e-3)
                continue
            shape, loc, scale = weibull_min.fit(tail, floc=0.0)
            params[label] = (float(shape), float(loc), float(max(scale, 1e-6)))
        self._weibull_params = params

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

    def get_scores(self, query_embedding: torch.Tensor) -> dict[str, float]:
        """Return Weibull-revised membership scores in [0, 1] per class.

        ``score_c = 1 - W_cdf(d(x, p_c); params_c)``. Larger means better fit
        to class c's calibrated boundary.
        """
        if not self._weibull_params:
            raise RuntimeError("Weibull not fit. Call fit_weibull() first.")
        dists = self.get_distances(query_embedding)
        scores: dict[str, float] = {}
        for label, d in dists.items():
            shape, loc, scale = self._weibull_params[label]
            cdf = float(weibull_min.cdf(d, shape, loc=loc, scale=scale))
            base = 1.0 - cdf
            if self.hybrid_alpha > 0.0:
                # Blend with squashed raw-distance signal so well-separated
                # raw distances still contribute when the Weibull saturates.
                dist_signal = float(np.exp(-d))
                base = (1.0 - self.hybrid_alpha) * base + self.hybrid_alpha * dist_signal
            scores[label] = base
        return scores

    def predict(
        self, query_embedding: torch.Tensor
    ) -> tuple[str, float]:
        """Predict keyword via argmax Weibull score.

        Returns:
            ``(label, score)``. Returns ``('unknown', max_score)`` when
            ``max_score`` is below ``self.threshold``.
        """
        if self.threshold is None:
            raise RuntimeError("No threshold set. Call calibrate() first.")
        scores = self.get_scores(query_embedding)
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        if best_score < self.threshold:
            return ("unknown", best_score)
        return (best_label, best_score)

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
        """Find score threshold achieving target FAR on unknowns.

        FAR = fraction of ``'unknown'`` validation samples whose max score
        is *above* the threshold (i.e. wrongly accepted as known).

        Args:
            val_embeddings: ``(M, embedding_dim)`` validation embeddings.
            val_labels: M labels; entries equal to ``'unknown'`` are
                negative samples.
            target_far: Desired FAR (default 5%).

        Returns:
            Calibrated threshold in [0, 1].
        """
        if not self._weibull_params:
            raise RuntimeError("Weibull not fit. Call fit_weibull() first.")

        unknown_mask = np.array([l == "unknown" for l in val_labels])
        if not unknown_mask.any():
            self.threshold = 0.0
            return self.threshold

        max_scores = []
        for i in range(val_embeddings.shape[0]):
            scores = self.get_scores(val_embeddings[i])
            max_scores.append(max(scores.values()))
        max_scores_arr = np.asarray(max_scores, dtype=float)

        unknown_scores = np.sort(max_scores_arr[unknown_mask])[::-1]  # descending
        idx = int(len(unknown_scores) * target_far)
        idx = max(0, min(idx, len(unknown_scores) - 1))
        # Add a tiny offset so an unknown sitting exactly at the quantile
        # (and saturated-zero outliers) get rejected instead of accepted.
        self.threshold = float(unknown_scores[idx]) + 1e-6
        return self.threshold
