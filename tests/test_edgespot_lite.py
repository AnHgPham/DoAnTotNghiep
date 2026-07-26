"""Tests for EdgeSpot-lite mel frontend and encoder."""

import torch
import torch.nn.functional as F

from src.features.mel import MelSpectrogramExtractor
from src.models.edgespot_lite import EdgeSpotLite


def test_mel_shape():
    ext = MelSpectrogramExtractor()
    wav = torch.randn(1, 16000)
    mel = ext.extract(wav)
    assert mel.shape == (1, 40, 101)


def test_mel_batch_shape():
    ext = MelSpectrogramExtractor()
    wavs = torch.randn(3, 1, 16000)
    mel = ext.extract_batch(wavs)
    assert mel.shape == (3, 1, 40, 101)
    expected = torch.stack([ext.extract(wav) for wav in wavs])
    assert torch.allclose(mel, expected, atol=1e-6, rtol=1e-6)


def test_edgespot_lite_output_shape():
    model = EdgeSpotLite(width_mult=1)
    x = torch.rand(2, 1, 40, 101)
    out = model(x)
    assert out.shape == (2, 64)


def test_edgespot_lite_l2_norm_external():
    model = EdgeSpotLite(width_mult=1)
    x = torch.rand(2, 1, 40, 101)
    out = model(x)
    out_normed = F.normalize(out, p=2, dim=-1)
    assert torch.allclose(out_normed.norm(dim=-1), torch.ones(2), atol=1e-5)
