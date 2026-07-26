"""MFCC feature extraction following Rusci et al. preprocessing pipeline.

Flow: waveform -> pad/trim -> MFCC(40) -> narrow(10) -> transpose -> (1, 47, 10)
"""

from __future__ import annotations

import math

import torch

from src.features.mel import make_mel_filterbank


class MFCCExtractor:
    """Extract MFCC features from raw audio waveforms.

    Computes 40 MFCC coefficients, retains only the first `num_features` (10),
    and transposes to (channel, T, features) format matching DSCNN input.

    Args:
        n_mfcc: Number of MFCC coefficients to compute.
        num_features: Number of coefficients to keep (first N via narrow).
        sample_rate: Audio sample rate in Hz.
        win_length_ms: STFT window length in milliseconds.
        hop_length_ms: STFT hop length in milliseconds.
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        num_features: int = 10,
        sample_rate: int = 16000,
        win_length_ms: int = 40,
        hop_length_ms: int = 20,
    ):
        self.n_mfcc = n_mfcc
        self.num_features = num_features
        self.sample_rate = sample_rate
        self.target_length = sample_rate  # 1 second
        self.n_fft = 1024
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.hop_length = int(sample_rate * hop_length_ms / 1000)

        self.mel_filterbank = make_mel_filterbank(sample_rate, self.n_fft, n_mfcc)
        self.dct_mat = self._make_dct(n_mfcc, n_mfcc)
        self._tensor_cache: dict[
            tuple[torch.device, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    @staticmethod
    def _make_dct(n_mfcc: int, n_mels: int) -> torch.Tensor:
        n = torch.arange(n_mels, dtype=torch.float32)
        k = torch.arange(n_mfcc, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(math.pi / n_mels * (n + 0.5) * k)
        basis[0] *= math.sqrt(1.0 / n_mels)
        if n_mfcc > 1:
            basis[1:] *= math.sqrt(2.0 / n_mels)
        return basis

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad with zeros (right) or truncate (right) to target_length.

        Args:
            waveform: (1, T) or (T,) tensor.

        Returns:
            (1, target_length) tensor.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        length = waveform.shape[-1]
        if length < self.target_length:
            pad_amount = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif length > self.target_length:
            waveform = waveform[..., : self.target_length]

        return waveform

    def _runtime_tensors(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return device/dtype-specific tensors without rebuilding them per clip."""
        key = (waveform.device, waveform.dtype)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = (
                torch.hann_window(
                    self.win_length,
                    device=waveform.device,
                    dtype=waveform.dtype,
                ),
                self.mel_filterbank.to(device=waveform.device, dtype=waveform.dtype),
                self.dct_mat.to(device=waveform.device, dtype=waveform.dtype),
            )
            self._tensor_cache[key] = cached
        return cached

    def _extract_prepared(self, waveform: torch.Tensor) -> torch.Tensor:
        window, fb, dct = self._runtime_tensors(waveform)
        leading_shape = waveform.shape[:-1]
        flat_waveform = waveform.reshape(-1, waveform.shape[-1])
        spec = torch.stft(
            flat_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.matmul(fb, power).clamp_min(1e-10)
        log_mel = 10.0 * torch.log10(mel)
        mfcc = torch.matmul(dct, log_mel)
        mfcc = mfcc.narrow(dim=-2, start=0, length=self.num_features)
        mfcc = mfcc.transpose(-2, -1)
        return mfcc.reshape(*leading_shape, mfcc.shape[-2], mfcc.shape[-1])

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract one waveform as ``(1, T_frames, num_features)``."""
        return self._extract_prepared(self._pad_or_trim(waveform))

    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from a batch of waveforms.

        Args:
            waveforms: (B, 1, T) raw audio.

        Returns:
            (B, 1, T_frames, num_features) tensor.
            For 1-sec audio: (B, 1, 47, 10).
        """
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0).unsqueeze(0)
        elif waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        elif waveforms.dim() != 3:
            raise ValueError(
                "waveforms must have shape (T,), (B, T), or (B, 1, T)"
            )
        return self._extract_prepared(self._pad_or_trim(waveforms))
