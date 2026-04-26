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
    ):
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, mfcc: torch.Tensor) -> torch.Tensor:
        """Apply frequency and time masks.

        Args:
            mfcc: (1, T, F) or (T, F) MFCC tensor.

        Returns:
            Masked MFCC tensor (same shape).
        """
        mfcc = mfcc.clone()
        squeeze = False
        if mfcc.dim() == 2:
            mfcc = mfcc.unsqueeze(0)
            squeeze = True

        _, T, F = mfcc.shape

        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_width, F - 1))
            f0 = random.randint(0, F - f)
            mfcc[:, :, f0:f0 + f] = 0.0

        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_width, T - 1))
            t0 = random.randint(0, T - t)
            mfcc[:, t0:t0 + t, :] = 0.0

        return mfcc.squeeze(0) if squeeze else mfcc
