"""Mel-spectrogram frontend for EdgeSpot-style models.

For a 1-second waveform at 16 kHz, the default centered STFT and 10 ms hop
produce a 40 x 101 frequency-time map, matching the EdgeSpot paper setup.
"""

from __future__ import annotations

import math

import torch


def _hz_to_mel(freqs: torch.Tensor) -> torch.Tensor:
    """Slaney-style Hz to mel conversion."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    mels = (freqs - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = freqs >= min_log_hz
    mels[log_region] = min_log_mel + torch.log(freqs[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    """Slaney-style mel to Hz conversion."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freqs[log_region] = min_log_hz * torch.exp(logstep * (mels[log_region] - min_log_mel))
    return freqs


def make_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Create a Slaney-normalized mel filterbank with shape ``(n_mels, bins)``."""
    if f_max is None:
        f_max = sample_rate / 2
    min_mel = _hz_to_mel(torch.tensor([f_min], dtype=torch.float32))[0]
    max_mel = _hz_to_mel(torch.tensor([f_max], dtype=torch.float32))[0]
    mels = torch.linspace(min_mel, max_mel, n_mels + 2)
    mel_freqs = _mel_to_hz(mels)
    fft_freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    fdiff = mel_freqs[1:] - mel_freqs[:-1]
    ramps = mel_freqs.unsqueeze(1) - fft_freqs.unsqueeze(0)
    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    weights = torch.maximum(torch.zeros_like(lower), torch.minimum(lower, upper))

    # Slaney area normalization.
    enorm = 2.0 / (mel_freqs[2:n_mels + 2] - mel_freqs[:n_mels])
    weights *= enorm.unsqueeze(1)
    return weights.float()


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
        self.n_fft = n_fft
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.center = center

        self.mel_filterbank = make_mel_filterbank(sample_rate, n_fft, n_mels)

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
        window = torch.hann_window(self.win_length, device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        fb = self.mel_filterbank.to(device=waveform.device, dtype=waveform.dtype)
        mel = torch.matmul(fb, power).clamp_min(1e-8)
        if self.log:
            mel = mel.log()
        return mel

    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract a batch as ``(B, 1, 40, 101)``."""
        return torch.stack([self.extract(waveforms[i]) for i in range(waveforms.shape[0])])
