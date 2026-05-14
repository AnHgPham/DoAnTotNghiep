import torch
import torch.nn.functional as F

from src.classifiers.calibration import (
    build_prototype_bundle,
    impostor_threshold,
    medoid_prototype,
    multi_prototypes,
)


def test_medoid_and_multi_prototype_shapes():
    embs = F.normalize(torch.randn(5, 8), p=2, dim=-1)
    assert medoid_prototype(embs).shape == (8,)
    assert multi_prototypes(embs, n_prototypes=2).shape == (2, 8)


def test_impostor_threshold_is_conservative():
    positives = F.normalize(torch.randn(5, 8) * 0.01 + torch.tensor([1.0] + [0.0] * 7), p=2, dim=-1)
    proto = F.normalize(positives.mean(dim=0), p=2, dim=0)
    impostors = F.normalize(torch.randn(20, 8) * 0.01 + torch.tensor([0.0, 1.0] + [0.0] * 6), p=2, dim=-1)
    thr = impostor_threshold(positives, proto, impostors, target_far=0.05)
    impostor_d = torch.cdist(impostors, proto.view(1, -1)).squeeze(1)
    assert float((impostor_d <= thr).float().mean()) <= 0.05


def test_build_prototype_bundle():
    support = {
        "yes": F.normalize(torch.randn(4, 8), p=2, dim=-1),
        "no": F.normalize(torch.randn(4, 8), p=2, dim=-1),
    }
    bundle = build_prototype_bundle(support, strategy="mean")
    assert bundle.prototypes.shape == (2, 8)
    assert set(bundle.thresholds) == {"yes", "no"}
