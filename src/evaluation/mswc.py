"""MSWC few-shot data provider for evaluation.

Loads audio from data/mswc_en/clips/<word>/, splits support/query
using a 1:9 ratio, and returns batched MFCC tensors.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch

from src.audio_io import load_waveform
from src.features.mel import MelSpectrogramExtractor
from src.features.mfcc import MFCCExtractor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000


def _load_wav(path: Path) -> torch.Tensor:
    """Load and preprocess a single audio file to (1, 16000)."""
    return load_waveform(path, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH)


class MSWCFewShotProvider:
    """Provides support/query MFCC samples from MSWC for evaluation.

    Uses a 1:9 split ratio (10% support, 90% query) per word to
    maximize query data for reliable few-shot evaluation.

    Args:
        mswc_dir: Path to data/mswc_en/ directory.
        support_ratio: Fraction of samples used for support (default 0.1).
    """

    def __init__(
        self,
        mswc_dir: str | Path,
        support_ratio: float = 0.1,
        feature_type: str = "mfcc",
    ):
        self.mswc_dir = Path(mswc_dir)
        if feature_type == "mfcc":
            self.extractor = MFCCExtractor()
        elif feature_type == "mel":
            self.extractor = MelSpectrogramExtractor()
        else:
            raise ValueError("feature_type must be 'mfcc' or 'mel'")
        self.feature_type = feature_type
        self.support_ratio = support_ratio

        clips_dir = self.mswc_dir / "clips"
        if not clips_dir.exists():
            clips_dir = self.mswc_dir

        self._support_files: dict[str, list[Path]] = {}
        self._query_files: dict[str, list[Path]] = {}

        for word_dir in sorted(clips_dir.iterdir()):
            if not word_dir.is_dir():
                continue
            wav_files = sorted(word_dir.glob("*.wav"))
            if len(wav_files) < 10:
                continue

            rng = random.Random(42)
            rng.shuffle(wav_files)
            split_idx = max(5, int(len(wav_files) * support_ratio))
            self._support_files[word_dir.name] = wav_files[:split_idx]
            self._query_files[word_dir.name] = wav_files[split_idx:]

        logger.info(
            "MSWCFewShotProvider: %d words from %s",
            len(self._support_files), self.mswc_dir,
        )

    def validate_words(self, words: list[str], min_support: int = 5) -> None:
        """Check that all requested words have enough samples."""
        for word in words:
            sup_count = len(self._support_files.get(word, []))
            query_count = len(self._query_files.get(word, []))
            if sup_count < min_support:
                raise ValueError(
                    f"Word '{word}' has only {sup_count} support samples "
                    f"(need {min_support})"
                )
            if query_count == 0:
                raise ValueError(f"Word '{word}' has 0 query samples")

    def _load_mfcc_batch(self, paths: list[Path]) -> tuple[torch.Tensor, list[str]]:
        """Load audio files and extract MFCC features."""
        mfccs = []
        names = []
        for p in paths:
            try:
                wav = _load_wav(p)
                mfcc = self.extractor.extract(wav)
                mfccs.append(mfcc.unsqueeze(0))
                names.append(p.name)
            except Exception as e:
                logger.warning("Skipping %s: %s", p, e)
                continue

        if not mfccs:
            raise RuntimeError(f"No valid audio files found in batch")
        return torch.cat(mfccs, dim=0), names

    def get_support_samples(
        self, word: str, n_samples: int, seed: int = 42
    ) -> tuple[torch.Tensor, list[str]]:
        """Get n_samples from support pool for enrollment."""
        pool = self._support_files.get(word, [])
        rng = random.Random(seed)
        selected = rng.sample(pool, min(n_samples, len(pool)))
        return self._load_mfcc_batch(selected)

    def get_query_samples(
        self, word: str, max_samples: int = 50
    ) -> tuple[torch.Tensor, list[str]]:
        """Get query samples for evaluation."""
        pool = self._query_files.get(word, [])
        if len(pool) > max_samples:
            rng = random.Random(42)
            pool = rng.sample(pool, max_samples)
        return self._load_mfcc_batch(pool)
