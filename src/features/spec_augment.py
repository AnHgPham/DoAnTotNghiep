"""SpecAugment: frequency and time masking for MFCC/spectrogram features.

Applies random frequency and time masks during training to improve
robustness, following Park et al. (2019).
"""

import random

import torch


class SpecAugment:
    """Apply SpecAugment masking to MFCC features.

    Args:
        freq_mask_width: Maximum width of frequency mask (F).
        time_mask_width: Maximum width of time mask (T).
        n_freq_masks: Number of frequency masks to apply.
        n_time_masks: Number of time masks to apply.
    """

    def __init__(
        self,
        freq_mask_width: int = 6,
        time_mask_width: int = 8,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        time_axis: int = 1,
        freq_axis: int = 2,
    ):
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.time_axis = time_axis
        self.freq_axis = freq_axis

    def __call__(self, mfcc: torch.Tensor) -> torch.Tensor:
        """Apply frequency and time masks.

        Args:
            mfcc: (1, T, F), (1, F, T), or 2D feature tensor.

        Returns:
            Masked MFCC tensor (same shape).
        """
        mfcc = mfcc.clone()
        squeeze = False
        if mfcc.dim() == 2:
            mfcc = mfcc.unsqueeze(0)
            squeeze = True

        time_axis = self.time_axis
        freq_axis = self.freq_axis
        if time_axis < 0:
            time_axis += mfcc.dim()
        if freq_axis < 0:
            freq_axis += mfcc.dim()

        T = mfcc.shape[time_axis]
        F = mfcc.shape[freq_axis]

        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_width, F - 1))
            f0 = random.randint(0, F - f)
            slc = [slice(None)] * mfcc.dim()
            slc[freq_axis] = slice(f0, f0 + f)
            mfcc[tuple(slc)] = 0.0

        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_width, T - 1))
            t0 = random.randint(0, T - t)
            slc = [slice(None)] * mfcc.dim()
            slc[time_axis] = slice(t0, t0 + t)
            mfcc[tuple(slc)] = 0.0

        return mfcc.squeeze(0) if squeeze else mfcc
