"""Audio augmentation: DEMAND noise mixing + speed/gain/time-shift perturbation.

Mixes clean audio with background noise at configurable SNR, plus lightweight
waveform perturbations that improve embedding robustness without extra parameters.
"""

import logging
import random
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


class WaveformAugmenter:
    """Lightweight waveform perturbations (no external data needed).

    Args:
        speed_range: (min, max) speed factor, e.g. (0.9, 1.1).
        gain_range_db: (min, max) gain in dB, e.g. (-6, 6).
        time_shift_ms: Maximum time shift in milliseconds.
        sample_rate: Audio sample rate.
    """

    def __init__(
        self,
        speed_range: tuple[float, float] = (0.9, 1.1),
        gain_range_db: tuple[float, float] = (-6.0, 6.0),
        time_shift_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.speed_range = speed_range
        self.gain_range_db = gain_range_db
        self.max_shift = int(sample_rate * time_shift_ms / 1000)
        self.sample_rate = sample_rate

    def speed_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly change playback speed (pitch+tempo)."""
        factor = random.uniform(*self.speed_range)
        if abs(factor - 1.0) < 0.01:
            return waveform
        effects = [["speed", str(factor)], ["rate", str(self.sample_rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True,
        )
        return augmented

    def gain_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly adjust volume."""
        gain_db = random.uniform(*self.gain_range_db)
        return waveform * (10.0 ** (gain_db / 20.0))

    def time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly shift audio left/right, filling with zeros."""
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return waveform
        if shift > 0:
            return torch.cat([torch.zeros(1, shift), waveform[:, :-shift]], dim=-1)
        return torch.cat([waveform[:, -shift:], torch.zeros(1, -shift)], dim=-1)

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply gain and time-shift perturbations randomly.

        Note: speed_perturb is disabled by default because sox_effects is
        extremely slow (~50-100ms/call), adding ~2h per epoch at 240k samples.
        """
        if random.random() < 0.5:
            waveform = self.gain_perturb(waveform)
        if random.random() < 0.5:
            waveform = self.time_shift(waveform)
        return waveform


class NoiseAugmenter:
    """Add background noise from DEMAND dataset to audio waveforms.

    Args:
        noise_dir: Path to directory containing DEMAND noise WAV files.
        prob: Probability of applying augmentation per sample.
        snr_db: Fixed Signal-to-Noise Ratio in dB.
    """

    def __init__(
        self,
        noise_dir: Path,
        prob: float = 0.95,
        snr_db: float | tuple[float, float] = 5.0,
    ):
        self.prob = prob
        self.snr_db = snr_db  # float for fixed, tuple for random range
        self.noise_dir = Path(noise_dir)
        self.noise_files = self._discover_noise_files()

        if len(self.noise_files) == 0:
            logger.warning(
                "No noise files found in %s. Augmentation will be a no-op.",
                noise_dir,
            )

        self._noise_cache = self._preload_noise()

    def _discover_noise_files(self) -> list[Path]:
        """Find all WAV files in the noise directory."""
        if not self.noise_dir.exists():
            return []
        return sorted(self.noise_dir.glob("**/*.wav"))

    def _preload_noise(self) -> list[torch.Tensor]:
        """Load all noise files into memory once (mono, 16kHz)."""
        cached = []
        for path in self.noise_files:
            try:
                noise, sr = torchaudio.load(str(path))
                if sr != 16000:
                    noise = torchaudio.transforms.Resample(sr, 16000)(noise)
                noise = noise.mean(dim=0, keepdim=True)
                cached.append(noise)
            except Exception as e:
                logger.warning("Failed to preload %s: %s", path, e)
        logger.info("Preloaded %d noise clips into memory", len(cached))
        return cached

    def _load_random_noise(self, target_length: int) -> torch.Tensor:
        """Get a random noise clip, looping or cropping to match target_length.

        Args:
            target_length: Desired number of samples.

        Returns:
            (1, target_length) noise tensor.
        """
        noise = random.choice(self._noise_cache).clone()

        if noise.shape[-1] < target_length:
            repeats = (target_length // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)

        start = random.randint(0, max(0, noise.shape[-1] - target_length))
        noise = noise[..., start:start + target_length]
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
        if len(self._noise_cache) == 0:
            return waveform

        if random.random() > self.prob:
            return waveform

        noise = self._load_random_noise(waveform.shape[-1])
        if isinstance(self.snr_db, tuple):
            snr = random.uniform(self.snr_db[0], self.snr_db[1])
        else:
            snr = self.snr_db
        return self._mix_snr(waveform, noise, snr)
