"""Tests for the denoiser module (EXT-1)."""

import pytest
import torch

nr = pytest.importorskip("noisereduce", reason="noisereduce not installed")


class TestSpectralGateDenoiser:
    """Tests for SpectralGateDenoiser."""

    def test_import(self):
        from src.enhancements.denoiser import SpectralGateDenoiser
        denoiser = SpectralGateDenoiser()
        assert denoiser is not None

    def test_denoise_shape_2d(self):
        from src.enhancements.denoiser import SpectralGateDenoiser
        denoiser = SpectralGateDenoiser()
        waveform = torch.randn(1, 16000)
        result = denoiser.denoise(waveform)
        assert result.shape == (1, 16000)

    def test_denoise_shape_1d(self):
        from src.enhancements.denoiser import SpectralGateDenoiser
        denoiser = SpectralGateDenoiser()
        waveform = torch.randn(16000)
        result = denoiser.denoise(waveform)
        assert result.dim() == 1
        assert result.shape[0] == 16000

    def test_denoise_reduces_noise(self):
        from src.enhancements.denoiser import SpectralGateDenoiser
        denoiser = SpectralGateDenoiser(prop_decrease=1.0)

        clean = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)
        noise = torch.randn(1, 16000) * 0.5
        noisy = clean + noise

        denoised = denoiser.denoise(noisy)
        noise_power_before = (noisy - clean).pow(2).mean()
        noise_power_after = (denoised - clean).pow(2).mean()
        assert noise_power_after < noise_power_before


class TestDenoiserWrapper:
    """Tests for the unified Denoiser wrapper."""

    def test_disabled_passthrough(self):
        from src.enhancements.denoiser import Denoiser
        denoiser = Denoiser(enabled=False)
        waveform = torch.randn(1, 16000)
        result = denoiser.denoise(waveform)
        assert torch.equal(result, waveform)

    def test_toggle_enabled(self):
        from src.enhancements.denoiser import Denoiser
        denoiser = Denoiser(backend="spectral_gate", enabled=True)
        waveform = torch.randn(1, 16000)

        denoiser.set_enabled(False)
        result = denoiser.denoise(waveform)
        assert torch.equal(result, waveform)

    def test_invalid_backend(self):
        from src.enhancements.denoiser import Denoiser
        with pytest.raises(ValueError, match="Unknown denoiser backend"):
            Denoiser(backend="nonexistent", enabled=True)
