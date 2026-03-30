"""Tests for DSCNN encoder."""

import torch
import torch.nn.functional as F

from src.models.dscnn import DSCNN


def test_dscnn_l_output_shape():
    model = DSCNN(model_size="L", feature_mode="CONV")
    x = torch.randn(4, 1, 47, 10)
    out = model(x)
    assert out.shape == (4, 276)


def test_dscnn_s_output_shape():
    model = DSCNN(model_size="S", feature_mode="CONV")
    x = torch.randn(4, 1, 47, 10)
    out = model(x)
    assert out.shape == (4, 64)


def test_dscnn_l_l2_norm_external():
    model = DSCNN(model_size="L", feature_mode="NORM")
    x = torch.randn(4, 1, 47, 10)
    out = model(x)
    out_normed = F.normalize(out, p=2, dim=-1)
    norms = out_normed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_dscnn_l_param_count():
    model = DSCNN(model_size="L")
    total = sum(p.numel() for p in model.parameters())
    print(f"DSCNN-L param count: {total:,}")
    assert total > 100_000  # sanity: should be substantial


def test_dscnn_s_param_count():
    model = DSCNN(model_size="S")
    total = sum(p.numel() for p in model.parameters())
    print(f"DSCNN-S param count: {total:,}")
    assert total < 100_000  # S should be lightweight


def test_dscnn_forward_backward():
    model = DSCNN(model_size="L")
    x = torch.randn(2, 1, 47, 10, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_dscnn_feature_modes():
    for mode in ["CONV", "RELU", "NORM"]:
        model = DSCNN(model_size="L", feature_mode=mode)
        x = torch.randn(2, 1, 47, 10)
        out = model(x)
        assert out.shape == (2, 276)


def test_dscnn_invalid_size():
    try:
        DSCNN(model_size="XL")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_dscnn_embedding_dim():
    model_l = DSCNN(model_size="L")
    model_s = DSCNN(model_size="S")
    assert model_l.embedding_dim == 276
    assert model_s.embedding_dim == 64


def test_dscnn_single_sample():
    model = DSCNN(model_size="L")
    x = torch.randn(1, 1, 47, 10)
    out = model(x)
    assert out.shape == (1, 276)
