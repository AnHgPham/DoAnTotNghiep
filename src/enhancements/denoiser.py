"""Inference-time speech denoising (EXT-1).

Provides two backends:
  1. noisereduce  – spectral-gating denoiser (lightweight, CPU-friendly)
  2. SpeechBrain  – neural speech enhancement (higher quality, optional GPU)

The wrapper auto-selects based on availability and configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class SpectralGateDenoiser:
    """Spectral-gating denoiser using the ``noisereduce`` library.

    Args:
        stationary: If True, assume stationary noise (faster).
        prop_decrease: Fraction of noise to remove (0.0–1.0).
    """

    def __init__(self, stationary: bool = True, prop_decrease: float = 0.9):
        try:
            import noisereduce as nr  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "noisereduce is required for SpectralGateDenoiser. "
                "Install with: pip install noisereduce"
            ) from exc
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    @staticmethod
    def _smooth_fallback(waveform: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
        """Conservative low-pass fallback used when spectral gating over-suppresses."""
        if kernel_size <= 1:
            return waveform
        squeeze = waveform.dim() == 1
        x = waveform.unsqueeze(0) if squeeze else waveform
        if x.dim() != 2 or x.shape[0] != 1:
            return waveform
        left = kernel_size // 2
        right = kernel_size - 1 - left
        kernel = torch.ones(
            1,
            1,
            kernel_size,
            dtype=x.dtype,
            device=x.device,
        ) / kernel_size
        smoothed = F.conv1d(F.pad(x.unsqueeze(0), (left, right), mode="reflect"), kernel)
        smoothed = smoothed.squeeze(0)
        return smoothed.squeeze(0) if squeeze else smoothed

    def denoise(self, waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
        """Denoise a waveform tensor.

        Args:
            waveform: (1, T) or (T,) audio tensor.
            sr: Sample rate.

        Returns:
            Denoised waveform with same shape.
        """
        import noisereduce as nr
        import numpy as np

        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True

        audio_np = waveform.squeeze(0).cpu().numpy()
        denoised_np = nr.reduce_noise(
            y=audio_np,
            sr=sr,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease,
        )
        result = torch.from_numpy(denoised_np).unsqueeze(0).to(waveform.device)
        result = result.to(dtype=waveform.dtype)

        # noisereduce can over-suppress very short KWS clips when no explicit
        # noise-only segment is available. Guard against near-silence output by
        # falling back to a conservative smoother that still reduces white noise.
        in_rms = torch.sqrt(torch.mean(waveform.float().pow(2)) + 1e-8)
        out_rms = torch.sqrt(torch.mean(result.float().pow(2)) + 1e-8)
        if out_rms < 0.25 * in_rms:
            result = self._smooth_fallback(waveform)

        if squeeze:
            result = result.squeeze(0)
        return result


class SpeechBrainDenoiser:
    """Neural speech enhancement using SpeechBrain's pre-trained model.

    Downloads the model on first use (~100 MB). Requires GPU for
    real-time performance but works on CPU for short clips.

    Args:
        model_source: HuggingFace model ID or local path.
        save_dir: Directory to cache downloaded model.
        device: Torch device for inference.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/metricgan-plus-voicebank",
        save_dir: str = "pretrained_models/metricgan-plus",
        device: torch.device | None = None,
    ):
        try:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
        except ImportError as exc:
            raise ImportError(
                "speechbrain is required for SpeechBrainDenoiser. "
                "Install with: pip install speechbrain"
            ) from exc

        self.device = device or torch.device("cpu")
        self.model = SpectralMaskEnhancement.from_hparams(
            source=model_source,
            savedir=save_dir,
            run_opts={"device": str(self.device)},
        )
        logger.info("SpeechBrain denoiser loaded from %s", model_source)

    def denoise(self, waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
        """Denoise a waveform tensor.

        Args:
            waveform: (1, T) or (T,) audio tensor.
            sr: Sample rate.

        Returns:
            Denoised waveform with same shape.
        """
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        enhanced = self.model.enhance_batch(
            waveform.to(self.device), lengths=torch.tensor([1.0])
        )
        enhanced = enhanced.cpu()

        if sr != 16000:
            enhanced = torchaudio.transforms.Resample(16000, sr)(enhanced)

        if squeeze:
            enhanced = enhanced.squeeze(0)
        return enhanced


class Denoiser:
    """Unified denoiser interface with backend selection.

    Args:
        backend: ``"spectral_gate"`` or ``"speechbrain"``.
        enabled: If False, ``denoise()`` returns input unchanged.
        device: Torch device (only used by SpeechBrain backend).
    """

    def __init__(
        self,
        backend: str = "spectral_gate",
        enabled: bool = True,
        device: torch.device | None = None,
    ):
        self.enabled = enabled
        self.backend_name = backend
        self._impl = None

        if enabled:
            if backend == "spectral_gate":
                self._impl = SpectralGateDenoiser()
            elif backend == "speechbrain":
                self._impl = SpeechBrainDenoiser(device=device)
            else:
                raise ValueError(f"Unknown denoiser backend: {backend}")
            logger.info("Denoiser initialized: backend=%s", backend)

    def denoise(self, waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
        """Denoise waveform if enabled, otherwise pass through.

        Args:
            waveform: (1, T) or (T,) audio tensor.
            sr: Sample rate.

        Returns:
            (Possibly denoised) waveform with same shape.
        """
        if not self.enabled or self._impl is None:
            return waveform
        return self._impl.denoise(waveform, sr)

    def set_enabled(self, enabled: bool) -> None:
        """Toggle denoising on/off at runtime."""
        self.enabled = enabled
