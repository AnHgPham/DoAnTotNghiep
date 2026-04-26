"""GSC v2 few-shot data provider for evaluation.

Loads real audio from data/gsc_v2/, splits support/query using the official
validation_list.txt and testing_list.txt, and returns batched MFCC tensors.
"""

import logging
import random
from pathlib import Path

import torch
import torchaudio

from src.features.mfcc import MFCCExtractor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000


def _load_wav(path: Path) -> torch.Tensor:
    """Load and preprocess a single WAV file to (1, 16000)."""
    waveform, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    length = waveform.shape[-1]
    if length < TARGET_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - length))
    elif length > TARGET_LENGTH:
        waveform = waveform[..., :TARGET_LENGTH]
    return waveform


class GSCFewShotProvider:
    """Provides support/query MFCC samples from GSC v2 for evaluation.

    Uses validation_list.txt files as the support (enrollment) pool and
    testing_list.txt as the query pool to prevent data leakage.

    Args:
        gsc_dir: Path to data/gsc_v2/ directory.
    """

    def __init__(self, gsc_dir: str | Path):
        self.gsc_dir = Path(gsc_dir)
        self.extractor = MFCCExtractor()

        val_list = self.gsc_dir / "validation_list.txt"
        test_list = self.gsc_dir / "testing_list.txt"

        if not val_list.exists():
            raise FileNotFoundError(f"Missing {val_list}")
        if not test_list.exists():
            raise FileNotFoundError(f"Missing {test_list}")

        self._val_files: dict[str, list[Path]] = {}
        self._test_files: dict[str, list[Path]] = {}

        for line in val_list.read_text().strip().splitlines():
            word = line.split("/")[0]
            path = self.gsc_dir / line.strip()
            if path.exists():
                self._val_files.setdefault(word, []).append(path)

        for line in test_list.read_text().strip().splitlines():
            word = line.split("/")[0]
            path = self.gsc_dir / line.strip()
            if path.exists():
                self._test_files.setdefault(word, []).append(path)

        logger.info(
            "GSCFewShotProvider: %d words in val, %d words in test",
            len(self._val_files), len(self._test_files),
        )

    def validate_words(self, words: list[str], min_support: int = 5) -> None:
        """Check that all requested words have enough samples."""
        for word in words:
            val_count = len(self._val_files.get(word, []))
            test_count = len(self._test_files.get(word, []))
            if val_count < min_support:
                raise ValueError(
                    f"Word '{word}' has only {val_count} validation samples "
                    f"(need {min_support})"
                )
            if test_count == 0:
                raise ValueError(f"Word '{word}' has 0 test samples")

    def _load_mfcc_batch(self, paths: list[Path]) -> tuple[torch.Tensor, list[str]]:
        """Load WAVs and extract MFCC for a list of file paths.

        Returns:
            (mfcc_batch, file_names) where mfcc_batch is (N, 1, 47, 10).
        """
        mfccs = []
        names = []
        for p in paths:
            wav = _load_wav(p)
            mfcc = self.extractor.extract(wav)   # (1, 47, 10)
            mfccs.append(mfcc.unsqueeze(0))       # (1, 1, 47, 10)
            names.append(p.name)
        return torch.cat(mfccs, dim=0), names     # (N, 1, 47, 10)

    def get_support_samples(
        self, word: str, n_samples: int, seed: int = 42
    ) -> tuple[torch.Tensor, list[str]]:
        """Get n_samples from validation set for enrollment.

        Args:
            word: Keyword string.
            n_samples: Number of support samples (k-shot).
            seed: Random seed for reproducible selection.

        Returns:
            (mfcc_batch, file_names) where mfcc_batch is (n_samples, 1, 47, 10).
        """
        pool = self._val_files.get(word, [])
        rng = random.Random(seed)
        selected = rng.sample(pool, min(n_samples, len(pool)))
        return self._load_mfcc_batch(selected)

    def get_query_samples(
        self, word: str, max_samples: int = 50
    ) -> tuple[torch.Tensor, list[str]]:
        """Get query samples from testing set.

        Args:
            word: Keyword string.
            max_samples: Cap per word to keep evaluation tractable.

        Returns:
            (mfcc_batch, file_names) where mfcc_batch is (N, 1, 47, 10).
        """
        pool = self._test_files.get(word, [])
        if len(pool) > max_samples:
            rng = random.Random(42)
            pool = rng.sample(pool, max_samples)
        return self._load_mfcc_batch(pool)
