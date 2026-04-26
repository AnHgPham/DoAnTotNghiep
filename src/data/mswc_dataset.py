"""MSWC / GSC dataset for episodic training.

Loads WAV files from a directory of word folders, extracts MFCC features,
and provides an episodic DataLoader compatible with both Triplet and ArcFace training.
"""

import logging
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from src.features.mfcc import MFCCExtractor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000


class MSWCDataset(Dataset):
    """Audio keyword dataset that returns pre-extracted MFCC features.

    Supports both MSWC (clips/<word>/*.wav) and GSC (<word>/*.wav) layouts.

    Args:
        root_dir: Root directory containing word folders.
        words: List of word names to include.
        max_per_word: Maximum samples per word (cap for balance).
        noise_augmenter: Optional NoiseAugmenter for training.
        wave_augmenter: Optional WaveformAugmenter for training.
        spec_augmenter: Optional SpecAugment applied to MFCC features (training only).
    """

    def __init__(
        self,
        root_dir: str | Path,
        words: list[str],
        max_per_word: int = 200,
        noise_augmenter=None,
        wave_augmenter=None,
        spec_augmenter=None,
    ):
        self.root_dir = Path(root_dir)
        self.extractor = MFCCExtractor()
        self.noise_augmenter = noise_augmenter
        self.wave_augmenter = wave_augmenter
        self.spec_augmenter = spec_augmenter

        self.samples: list[tuple[Path, int]] = []
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}

        for i, word in enumerate(sorted(words)):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word

            # Support both layouts
            word_dir = self.root_dir / word
            if not word_dir.exists():
                word_dir = self.root_dir / "clips" / word
            if not word_dir.exists():
                logger.warning("Word directory not found: %s", word)
                continue

            wav_files = sorted(word_dir.glob("*.wav"))
            if len(wav_files) > max_per_word:
                rng = random.Random(42)
                wav_files = rng.sample(wav_files, max_per_word)

            for f in wav_files:
                self.samples.append((f, i))

        logger.info(
            "Dataset: %d samples, %d words from %s",
            len(self.samples), len(self.word_to_idx), self.root_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        waveform, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad/trim to 1 second
        if waveform.shape[-1] < TARGET_LENGTH:
            waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[-1]))
        elif waveform.shape[-1] > TARGET_LENGTH:
            waveform = waveform[..., :TARGET_LENGTH]

        # Augmentation (training only)
        if self.wave_augmenter is not None:
            waveform = self.wave_augmenter.augment(waveform)
        if self.noise_augmenter is not None:
            waveform = self.noise_augmenter.augment(waveform)

        mfcc = self.extractor.extract(waveform)  # (1, 47, 10)

        if self.spec_augmenter is not None:
            mfcc = self.spec_augmenter(mfcc)

        return mfcc, label


def build_episodic_loader(
    dataset: MSWCDataset,
    n_classes: int = 30,
    n_samples: int = 20,
    n_episodes: int = 400,
    num_workers: int = 0,
) -> DataLoader:
    """Build episodic DataLoader for metric learning.

    Args:
        dataset: MSWCDataset instance.
        n_classes: Classes per episode.
        n_samples: Samples per class per episode.
        n_episodes: Episodes per epoch.
        num_workers: DataLoader workers.
    """
    from src.models.prototypical import EpisodicBatchSampler

    labels = [s[1] for s in dataset.samples]
    sampler = EpisodicBatchSampler(
        labels=labels,
        n_classes=n_classes,
        n_samples=n_samples,
        n_episodes=n_episodes,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
