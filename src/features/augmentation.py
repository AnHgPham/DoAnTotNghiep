"""Noise augmentation using DEMAND dataset at fixed SNR.

Mixes clean audio with random background noise at 5dB SNR with RMS normalization.
"""

import logging
import random
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


class NoiseAugmenter:
    """Add background noise from DEMAND dataset to audio waveforms.

    Args:
        noise_dir: Path to directory containing DEMAND noise WAV files.
        prob: Probability of applying augmentation per sample.
        snr_db: Fixed Signal-to-Noise Ratio in dB.
    """

    def __init__(self, noise_dir: Path, prob: float = 0.95, snr_db: float = 5.0):
        self.prob = prob
        self.snr_db = snr_db
        self.noise_dir = Path(noise_dir)
        self.noise_files = self._discover_noise_files()

        if len(self.noise_files) == 0:
            logger.warning(
                "No noise files found in %s. Augmentation will be a no-op.",
                noise_dir,
            )

    def _discover_noise_files(self) -> list[Path]:
        """Find all WAV files in the noise directory."""
        if not self.noise_dir.exists():
            return []
        return sorted(self.noise_dir.glob("**/*.wav"))

    def _load_random_noise(self, target_length: int) -> torch.Tensor:
        """Load a random noise clip, looping or cropping to match target_length.

        Args:
            target_length: Desired number of samples.

        Returns:
            (1, target_length) noise tensor.
        """
        noise_path = random.choice(self.noise_files)
        noise, sr = torchaudio.load(noise_path)

        if sr != 16000:
            noise = torchaudio.transforms.Resample(sr, 16000)(noise)

        noise = noise.mean(dim=0, keepdim=True)  # mono

        if noise.shape[-1] < target_length:
            repeats = (target_length // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)

        noise = noise[..., :target_length]
        return noise

    @staticmethod
    def _mix_snr(
        clean: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Mix clean and noise at specified SNR using RMS normalization.

        Args:
            clean: (1, T) clean waveform.
            noise: (1, T) noise waveform (same length as clean).
            snr_db: Target SNR in dB.

        Returns:
            (1, T) mixed waveform.
        """
        eps = 1e-8
        rms_clean = torch.sqrt(torch.mean(clean**2) + eps)
        rms_noise = torch.sqrt(torch.mean(noise**2) + eps)
        scale = rms_clean / (rms_noise * 10 ** (snr_db / 20))
        return clean + scale * noise

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply noise augmentation with configured probability.

        Args:
            waveform: (1, T) clean waveform.

        Returns:
            (1, T) waveform, possibly with added noise.
        """
        if len(self.noise_files) == 0:
            return waveform

        if random.random() > self.prob:
            return waveform

        noise = self._load_random_noise(waveform.shape[-1])
        return self._mix_snr(waveform, noise, self.snr_db)
