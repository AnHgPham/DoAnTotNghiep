"""Tests for NoiseAugmenter."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.features.augmentation import NoiseAugmenter


def _create_mock_noise_dir() -> Path:
    """Create a temp directory with a synthetic noise WAV file."""
    tmpdir = Path(tempfile.mkdtemp())
    noise = np.random.randn(32000).astype(np.float32) * 0.1
    sf.write(str(tmpdir / "noise_01.wav"), noise, 16000)
    return tmpdir


def test_augment_prob_zero():
    noise_dir = _create_mock_noise_dir()
    aug = NoiseAugmenter(noise_dir=noise_dir, prob=0.0, snr_db=5.0)
    wav = torch.randn(1, 16000)
    result = aug.augment(wav)
    assert torch.equal(result, wav)


def test_augment_prob_one():
    noise_dir = _create_mock_noise_dir()
    aug = NoiseAugmenter(noise_dir=noise_dir, prob=1.0, snr_db=5.0)
    wav = torch.randn(1, 16000)
    result = aug.augment(wav)
    assert not torch.equal(result, wav)


def test_augment_shape():
    noise_dir = _create_mock_noise_dir()
    aug = NoiseAugmenter(noise_dir=noise_dir, prob=1.0, snr_db=5.0)
    wav = torch.randn(1, 16000)
    result = aug.augment(wav)
    assert result.shape == wav.shape


def test_augment_snr_approximate():
    noise_dir = _create_mock_noise_dir()
    aug = NoiseAugmenter(noise_dir=noise_dir, prob=1.0, snr_db=5.0)
    clean = torch.randn(1, 16000)
    noisy = aug.augment(clean)
    noise_component = noisy - clean
    rms_clean = torch.sqrt(torch.mean(clean**2))
    rms_noise = torch.sqrt(torch.mean(noise_component**2))
    actual_snr = 20 * torch.log10(rms_clean / (rms_noise + 1e-8))
    assert abs(actual_snr.item() - 5.0) < 1.0  # within 1dB tolerance


def test_augment_no_noise_dir():
    aug = NoiseAugmenter(noise_dir=Path("/nonexistent"), prob=1.0, snr_db=5.0)
    wav = torch.randn(1, 16000)
    result = aug.augment(wav)
    assert torch.equal(result, wav)


def test_mix_snr_static():
    clean = torch.ones(1, 1000) * 0.5
    noise = torch.ones(1, 1000) * 0.5
    mixed = NoiseAugmenter._mix_snr(clean, noise, snr_db=0.0)
    assert mixed.shape == clean.shape
    expected = clean + noise  # at 0dB with equal RMS, scale ~= 1
    assert torch.allclose(mixed, expected, atol=0.01)
