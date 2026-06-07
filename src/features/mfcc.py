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

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from a single waveform.

        Args:
            waveform: (1, T) raw audio at sample_rate Hz.

        Returns:
            (1, T_frames, num_features) tensor.
            For 1-sec audio: (1, 47, 10).
        """
        waveform = self._pad_or_trim(waveform)
        window = torch.hann_window(self.win_length, device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            waveform,
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
        fb = self.mel_filterbank.to(device=waveform.device, dtype=waveform.dtype)
        mel = torch.matmul(fb, power).clamp_min(1e-10)
        log_mel = 10.0 * torch.log10(mel)
        dct = self.dct_mat.to(device=waveform.device, dtype=waveform.dtype)
        mfcc = torch.matmul(dct, log_mel)  # (1, n_mfcc, T_frames)
        mfcc = mfcc.narrow(dim=-2, start=0, length=self.num_features)  # (1, 10, T_frames)
        mfcc = mfcc.mT  # (1, T_frames, 10)
        return mfcc

    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from a batch of waveforms.

        Args:
            waveforms: (B, 1, T) raw audio.

        Returns:
            (B, 1, T_frames, num_features) tensor.
            For 1-sec audio: (B, 1, 47, 10).
        """
        batch_size = waveforms.shape[0]
        results = []
        for i in range(batch_size):
            mfcc = self.extract(waveforms[i])  # (1, T_frames, num_features)
            results.append(mfcc)
        return torch.stack(results, dim=0)  # (B, 1, T_frames, num_features)
