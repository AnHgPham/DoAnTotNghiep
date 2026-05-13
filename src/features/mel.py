"""Mel-spectrogram frontend for EdgeSpot-style models.

For a 1-second waveform at 16 kHz, the default centered STFT and 10 ms hop
produce a 40 x 101 frequency-time map, matching the EdgeSpot paper setup.
"""

from __future__ import annotations

import torch
import torchaudio


class MelSpectrogramExtractor:
    """Extract 40-band mel energy features from raw audio waveforms.

    Args:
        n_mels: Number of mel bands.
        sample_rate: Audio sample rate in Hz.
        n_fft: STFT FFT size.
        win_length_ms: Window length in milliseconds.
        hop_length_ms: Hop length in milliseconds. Default 10 ms gives 101
            frames for 1-second audio when ``center=True``.
        center: Whether to use centered STFT framing.
        log: If True, return log-mel energies. EdgeSpot-lite defaults to raw
            non-negative energies because PCEN handles compression.
    """

    def __init__(
        self,
        n_mels: int = 40,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length_ms: int = 25,
        hop_length_ms: int = 10,
        center: bool = True,
        log: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.target_length = sample_rate
        self.log = log

        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=center,
            pad_mode="constant",
            mel_scale="slaney",
            norm="slaney",
        )

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        length = waveform.shape[-1]
        if length < self.target_length:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.target_length - length),
            )
        elif length > self.target_length:
            waveform = waveform[..., : self.target_length]
        return waveform

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract a single mel feature map as ``(1, 40, 101)``."""
        waveform = self._pad_or_trim(waveform)
        mel = self.mel_transform(waveform).clamp_min(1e-8)
        if self.log:
            mel = mel.log()
        return mel

    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract a batch as ``(B, 1, 40, 101)``."""
        return torch.stack([self.extract(waveforms[i]) for i in range(waveforms.shape[0])])
