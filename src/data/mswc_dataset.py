"""MSWC / GSC dataset for episodic training.

Loads WAV files from a directory of word folders, extracts features, and provides
an episodic DataLoader compatible with Triplet/ArcFace/SCAF training.
"""

import logging
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from src.features.mel import MelSpectrogramExtractor
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
        max_per_word: Maximum samples per word (cap for balance). ``<= 0`` uses
            all ``*.wav`` files found for each word.
        noise_augmenter: Optional NoiseAugmenter for training.
        wave_augmenter: Optional WaveformAugmenter for training.
        spec_augmenter: Optional SpecAugment applied to MFCC features (training only).
        feature_type: ``"mfcc"`` for DSCNN input or ``"mel"`` for EdgeSpot-lite input.
    """

    def __init__(
        self,
        root_dir: str | Path,
        words: list[str],
        max_per_word: int = 200,
        noise_augmenter=None,
        wave_augmenter=None,
        spec_augmenter=None,
        feature_type: str = "mfcc",
    ):
        self.root_dir = Path(root_dir)
        if feature_type == "mfcc":
            self.extractor = MFCCExtractor()
        elif feature_type == "mel":
            self.extractor = MelSpectrogramExtractor()
        else:
            raise ValueError("feature_type must be 'mfcc' or 'mel'")
        self.feature_type = feature_type
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
            if max_per_word > 0 and len(wav_files) > max_per_word:
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

        features = self.extractor.extract(waveform)

        if self.spec_augmenter is not None:
            features = self.spec_augmenter(features)

        return features, label


def build_episodic_loader(
    dataset: MSWCDataset,
    n_classes: int = 30,
    n_samples: int = 20,
    n_episodes: int = 400,
    num_workers: int = 0,
    hard_pairs_path: str | None = None,
    hard_pair_prob: float = 0.0,
) -> DataLoader:
    """Build episodic DataLoader for metric learning.

    Args:
        dataset: MSWCDataset instance.
        n_classes: Classes per episode.
        n_samples: Samples per class per episode.
        n_episodes: Episodes per epoch.
        num_workers: DataLoader workers.
        hard_pairs_path: Optional path to ``results/hard_pairs.json`` produced
            by ``scripts/analyze_confusion.py``. Pairs whose words are not in
            ``dataset.word_to_idx`` are dropped silently.
        hard_pair_prob: Probability per episode of seeding the episode with one
            hard pair (forces both confused words into the support batch).
            Recommended 0.3-0.5. Ignored if hard_pairs_path is None.
    """
    from src.models.prototypical import EpisodicBatchSampler

    labels = [s[1] for s in dataset.samples]

    hard_pairs_int: list[tuple[int, int]] = []
    pair_weights: list[float] | None = None
    if hard_pairs_path:
        import json
        from pathlib import Path

        p = Path(hard_pairs_path)
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            word_to_idx = dataset.word_to_idx
            seen = set()
            for entry in payload.get("hard_pairs_directional", []):
                a, b = entry.get("true"), entry.get("pred")
                if a not in word_to_idx or b not in word_to_idx:
                    continue
                ia, ib = word_to_idx[a], word_to_idx[b]
                if ia == ib:
                    continue
                key = frozenset({ia, ib})
                if key in seen:
                    continue
                seen.add(key)
                hard_pairs_int.append((ia, ib))
            if hard_pairs_int:
                weights_map = payload.get("hard_pairs_undirected_weights", {})
                pair_weights = []
                for ia, ib in hard_pairs_int:
                    a_w = dataset.idx_to_word[ia] if hasattr(dataset, "idx_to_word") else None
                    b_w = dataset.idx_to_word[ib] if hasattr(dataset, "idx_to_word") else None
                    if a_w and b_w:
                        a, b = sorted([a_w, b_w])
                        pair_weights.append(float(weights_map.get(f"{a}|{b}", 1.0)))
                    else:
                        pair_weights.append(1.0)

    sampler = EpisodicBatchSampler(
        labels=labels,
        n_classes=n_classes,
        n_samples=n_samples,
        n_episodes=n_episodes,
        hard_pairs=hard_pairs_int or None,
        hard_pair_prob=hard_pair_prob if hard_pairs_int else 0.0,
        pair_weights=pair_weights,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
