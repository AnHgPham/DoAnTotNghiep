"""GSC v2 few-shot data provider for evaluation.

Loads real audio from data/gsc_v2/, splits support/query using the official
validation_list.txt and testing_list.txt, and returns batched feature tensors.
The EdgeSpot reproduction protocol also needs a true silence class, which is
generated from deterministic 1-second crops of ``_background_noise_`` files.
"""

import logging
import random
from pathlib import Path

import torch
import torchaudio

from src.features.mel import MelSpectrogramExtractor
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

    SILENCE_WORD = "_silence_"

    def __init__(
        self,
        gsc_dir: str | Path,
        feature_type: str = "mfcc",
        query_split: str = "test",
        support_split: str = "val",
    ):
        self.gsc_dir = Path(gsc_dir)
        if feature_type == "mfcc":
            self.extractor = MFCCExtractor()
        elif feature_type == "mel":
            self.extractor = MelSpectrogramExtractor()
        else:
            raise ValueError("feature_type must be 'mfcc' or 'mel'")
        self.feature_type = feature_type
        if query_split not in ("dev", "test"):
            raise ValueError("query_split must be 'dev' or 'test'")
        if support_split != "val":
            raise ValueError("Only support_split='val' is currently supported")
        self.query_split = query_split
        self.support_split = support_split

        val_list = self.gsc_dir / "validation_list.txt"
        test_list = self.gsc_dir / "testing_list.txt"

        if not val_list.exists():
            raise FileNotFoundError(f"Missing {val_list}")
        if not test_list.exists():
            raise FileNotFoundError(f"Missing {test_list}")

        self._val_files: dict[str, list[Path]] = {}
        self._test_files: dict[str, list[Path]] = {}
        self._train_files: dict[str, list[Path]] = {}
        self._background_noise_files = sorted((self.gsc_dir / "_background_noise_").glob("*.wav"))

        val_entries = {line.strip() for line in val_list.read_text().strip().splitlines() if line.strip()}
        test_entries = {line.strip() for line in test_list.read_text().strip().splitlines() if line.strip()}

        for line in sorted(val_entries):
            word = line.split("/")[0]
            path = self.gsc_dir / line
            if path.exists():
                self._val_files.setdefault(word, []).append(path)

        for line in sorted(test_entries):
            word = line.split("/")[0]
            path = self.gsc_dir / line
            if path.exists():
                self._test_files.setdefault(word, []).append(path)

        held_out = val_entries | test_entries
        for word_dir in sorted(self.gsc_dir.iterdir()):
            if not word_dir.is_dir() or word_dir.name.startswith(("_", ".")):
                continue
            for path in sorted(word_dir.glob("*.wav")):
                rel = path.relative_to(self.gsc_dir).as_posix()
                if rel in held_out:
                    continue
                self._train_files.setdefault(word_dir.name, []).append(path)

        logger.info(
            "GSCFewShotProvider: %d words in val, %d words in test, "
            "%d words in train, query_split=%s",
            len(self._val_files), len(self._test_files),
            len(self._train_files), self.query_split,
        )

    def validate_words(self, words: list[str], min_support: int = 5) -> None:
        """Check that all requested words have enough samples."""
        for word in words:
            if word == self.SILENCE_WORD:
                if not self._background_noise_files:
                    raise ValueError(
                        "EdgeSpot silence class needs files under "
                        "data/gsc_v2/_background_noise_"
                    )
                continue
            val_count = len(self._val_files.get(word, []))
            query_count = len(self._query_files().get(word, []))
            if val_count < min_support:
                raise ValueError(
                    f"Word '{word}' has only {val_count} validation samples "
                    f"(need {min_support})"
                )
            if query_count == 0:
                raise ValueError(f"Word '{word}' has 0 {self.query_split} query samples")

    def _query_files(self) -> dict[str, list[Path]]:
        return self._train_files if self.query_split == "dev" else self._test_files

    def _load_feature_batch_from_waves(
        self, waves: list[torch.Tensor], names: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        features = []
        for wav in waves:
            features.append(self.extractor.extract(wav).unsqueeze(0))
        return torch.cat(features, dim=0), names

    def _make_silence_waves(
        self, n_samples: int, seed: int, namespace: str
    ) -> tuple[list[torch.Tensor], list[str]]:
        if not self._background_noise_files:
            raise ValueError("No _background_noise_ WAV files found for silence class")

        rng = random.Random(f"{namespace}:{seed}:{n_samples}")
        waves: list[torch.Tensor] = []
        names: list[str] = []
        for i in range(n_samples):
            path = rng.choice(self._background_noise_files)
            bg, sr = torchaudio.load(str(path))
            if sr != SAMPLE_RATE:
                bg = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(bg)
            if bg.shape[0] > 1:
                bg = bg.mean(dim=0, keepdim=True)
            if bg.shape[-1] <= TARGET_LENGTH:
                wav = torch.nn.functional.pad(bg, (0, max(0, TARGET_LENGTH - bg.shape[-1])))
                wav = wav[..., :TARGET_LENGTH]
                start = 0
            else:
                start = rng.randint(0, bg.shape[-1] - TARGET_LENGTH)
                wav = bg[..., start:start + TARGET_LENGTH]
            waves.append(wav)
            names.append(f"{self.SILENCE_WORD}/{path.stem}_{start}_{i}.wav")
        return waves, names

    def _load_mfcc_batch(self, paths: list[Path]) -> tuple[torch.Tensor, list[str]]:
        """Load WAVs and extract MFCC for a list of file paths.

        Returns:
            (mfcc_batch, file_names) where mfcc_batch is (N, 1, 47, 10).
        """
        mfccs = []
        names = []
        for p in paths:
            wav = _load_wav(p)
            mfcc = self.extractor.extract(wav)
            mfccs.append(mfcc.unsqueeze(0))
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
        if word == self.SILENCE_WORD:
            waves, names = self._make_silence_waves(n_samples, seed, "support")
            return self._load_feature_batch_from_waves(waves, names)

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
        if word == self.SILENCE_WORD:
            waves, names = self._make_silence_waves(max_samples, 10_000 + max_samples, "query")
            return self._load_feature_batch_from_waves(waves, names)

        pool = self._query_files().get(word, [])
        if len(pool) > max_samples:
            rng = random.Random(42)
            pool = rng.sample(pool, max_samples)
        return self._load_mfcc_batch(pool)
