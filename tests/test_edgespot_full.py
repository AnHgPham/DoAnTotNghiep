import torch

from src.models.bcresnet_fs import BCResNetFS
from src.models.edgespot_full import EdgeSpotFull


def test_edgespot_full_tau_shapes():
    x = torch.rand(2, 1, 40, 101)
    for tau in (1, 2, 3, 4):
        model = EdgeSpotFull(tau=tau)
        y = model(x)
        assert y.shape == (2, 64)


def test_bcresnet_fs_shape():
    model = BCResNetFS(tau=1)
    x = torch.rand(2, 1, 40, 101)
    y = model(x)
    assert y.shape == (2, 64)
