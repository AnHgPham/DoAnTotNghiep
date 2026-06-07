"""Small audio I/O helpers that avoid importing torchaudio.

The USTH GPU servers currently expose an old CUDA 10.2 driver. Some
``torchaudio`` wheels crash at import time on that stack, so the training and
evaluation path uses ``soundfile`` + ``scipy`` for WAV loading/resampling.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad or truncate ``waveform`` to ``target_length`` samples."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    length = waveform.shape[-1]
    if length < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - length))
    elif length > target_length:
        waveform = waveform[..., :target_length]
    return waveform


def _resample_array(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    gcd = math.gcd(int(source_sr), int(target_sr))
    up = int(target_sr) // gcd
    down = int(source_sr) // gcd
    return resample_poly(audio, up, down, axis=-1).astype(np.float32, copy=False)


def load_waveform(
    path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
    target_length: int | None = None,
) -> torch.Tensor:
    """Load audio as a float32 tensor in ``(channels, samples)`` format.

    Args:
        path: WAV/FLAC/OGG path readable by libsndfile.
        sample_rate: Target sample rate.
        mono: Average channels to mono when true.
        target_length: Optional sample count to pad/trim to.
    """
    data, source_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.T
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    audio = _resample_array(audio, source_sr, sample_rate)
    waveform = torch.from_numpy(np.ascontiguousarray(audio))
    if target_length is not None:
        waveform = pad_or_trim(waveform, target_length)
    return waveform
