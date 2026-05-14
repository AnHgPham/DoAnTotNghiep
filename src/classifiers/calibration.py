"""Prototype and threshold calibration utilities for few-shot open-set KWS."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PrototypeBundle:
    labels: list[str]
    prototypes: torch.Tensor
    thresholds: dict[str, float]
    strategy: str


def _as_normalized(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


def mean_prototype(embeddings: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    embeddings = _as_normalized(embeddings)
    if weights is None:
        proto = embeddings.mean(dim=0)
    else:
        w = weights.float().clamp_min(0)
        w = w / w.sum().clamp_min(1e-8)
        proto = (embeddings * w.unsqueeze(-1)).sum(dim=0)
    return F.normalize(proto, p=2, dim=0)


def medoid_prototype(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = _as_normalized(embeddings)
    dists = torch.cdist(embeddings, embeddings, p=2)
    idx = dists.mean(dim=1).argmin()
    return embeddings[idx]


def multi_prototypes(embeddings: torch.Tensor, n_prototypes: int = 2, n_iter: int = 8) -> torch.Tensor:
    """Small deterministic k-means for support embeddings."""
    embeddings = _as_normalized(embeddings)
    n = embeddings.shape[0]
    k = max(1, min(int(n_prototypes), n))
    if k == 1:
        return mean_prototype(embeddings).unsqueeze(0)

    # Deterministic farthest-point initialization.
    centroids = [embeddings[0]]
    while len(centroids) < k:
        cur = torch.stack(centroids)
        d = torch.cdist(embeddings, cur, p=2).min(dim=1).values
        centroids.append(embeddings[d.argmax()])
    centers = torch.stack(centroids)

    for _ in range(n_iter):
        assign = torch.cdist(embeddings, centers, p=2).argmin(dim=1)
        new_centers = []
        for i in range(k):
            part = embeddings[assign == i]
            new_centers.append(mean_prototype(part) if part.numel() else centers[i])
        centers = torch.stack(new_centers)
    return _as_normalized(centers)


def support_uncertainty_threshold(
    embeddings: torch.Tensor,
    prototype: torch.Tensor,
    alpha: float = 2.0,
    floor: float = 0.10,
    ceil: float = 1.50,
) -> float:
    embeddings = _as_normalized(embeddings)
    prototype = F.normalize(prototype.float(), p=2, dim=0)
    dists = torch.cdist(embeddings, prototype.view(1, -1), p=2).squeeze(1)
    threshold = float(dists.mean().item() + alpha * dists.std(unbiased=False).item())
    return max(float(floor), min(float(ceil), threshold))


def impostor_threshold(
    positive_embeddings: torch.Tensor,
    prototype: torch.Tensor,
    impostor_embeddings: torch.Tensor,
    target_far: float = 0.05,
    support_alpha: float = 2.0,
) -> float:
    """Threshold constrained by impostor FAR and support compactness."""
    support_thr = support_uncertainty_threshold(
        positive_embeddings, prototype, alpha=support_alpha,
    )
    if impostor_embeddings.numel() == 0:
        return support_thr
    impostors = _as_normalized(impostor_embeddings)
    prototype = F.normalize(prototype.float(), p=2, dim=0)
    dists = torch.cdist(impostors, prototype.view(1, -1), p=2).squeeze(1)
    sorted_d = torch.sort(dists).values
    allowed = int(torch.floor(torch.tensor(float(target_far) * sorted_d.numel())).item())
    if allowed <= 0:
        impostor_thr = float(sorted_d[0].item() - 1e-6)
    else:
        impostor_thr = float(sorted_d[min(allowed - 1, sorted_d.numel() - 1)].item())
    return min(support_thr, impostor_thr)


def build_prototype_bundle(
    embeddings_by_label: dict[str, torch.Tensor],
    strategy: str = "mean",
    quality_weights: dict[str, torch.Tensor] | None = None,
    impostor_embeddings: torch.Tensor | None = None,
    target_far: float = 0.05,
) -> PrototypeBundle:
    """Build calibrated prototypes for enrollment/evaluation sweeps."""
    labels = sorted(embeddings_by_label)
    prototypes = []
    thresholds = {}
    for label in labels:
        embeddings = embeddings_by_label[label]
        if strategy == "mean":
            proto = mean_prototype(embeddings)
        elif strategy == "quality_weighted":
            weights = quality_weights.get(label) if quality_weights else None
            proto = mean_prototype(embeddings, weights=weights)
        elif strategy == "medoid":
            proto = medoid_prototype(embeddings)
        else:
            raise ValueError("strategy must be mean | quality_weighted | medoid")
        prototypes.append(proto)
        if impostor_embeddings is None:
            thresholds[label] = support_uncertainty_threshold(embeddings, proto)
        else:
            thresholds[label] = impostor_threshold(
                embeddings, proto, impostor_embeddings, target_far=target_far,
            )
    return PrototypeBundle(
        labels=labels,
        prototypes=torch.stack(prototypes),
        thresholds=thresholds,
        strategy=strategy,
    )
