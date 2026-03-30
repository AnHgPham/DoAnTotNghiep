"""Evaluation protocols for few-shot open-set keyword spotting.

Three protocols:
- GSC Fixed: 10 positive + 20 negative from GSC (deterministic)
- GSC Randomized: random selection from 35 GSC words each run
- MSWC Randomized: 5 positive + 50 negative from 263 MSWC eval words
"""

import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import (
    compute_auc,
    compute_det_curve,
    compute_frr_at_far,
    compute_mean_det,
)

logger = logging.getLogger(__name__)

GSC_POSITIVE_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
GSC_EXCLUDED_WORDS = ["backward", "forward", "visual", "follow", "learn"]
GSC_ALL_35_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual",
]


class EvaluationProtocol:
    """Few-shot open-set evaluation protocol.

    Runs multiple evaluation episodes, each time partitioning words into
    positive (enrolled) and negative (non-enrolled) sets, computing prototypes
    from support samples, and evaluating on query samples.

    Args:
        dataset: 'gsc' or 'mswc'.
        mode: 'fixed' or 'random'.
        n_runs: Number of evaluation runs (5 mandatory, 10 extended).
        n_way: Number of positive classes per episode.
        k_shot: Number of support samples per positive class.
        seed: Base random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: str = "gsc",
        mode: str = "fixed",
        n_runs: int = 5,
        n_way: int = 5,
        k_shot: int = 5,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.mode = mode
        self.n_runs = n_runs
        self.n_way = n_way
        self.k_shot = k_shot
        self.seed = seed

        if dataset == "gsc" and mode == "fixed":
            self.n_positive = 10
            self.n_negative = 20
        elif dataset == "gsc" and mode == "random":
            self.n_positive = 10
            self.n_negative = 20
        elif dataset == "mswc":
            self.n_positive = 5
            self.n_negative = 50
        else:
            raise ValueError(f"Unknown dataset/mode: {dataset}/{mode}")

    def get_partitions(
        self, run_idx: int
    ) -> tuple[list[str], list[str]]:
        """Get word partitions for a specific run.

        Args:
            run_idx: Run index (0-based).

        Returns:
            (positive_words, negative_words) lists.
        """
        if self.dataset == "gsc" and self.mode == "fixed":
            return self._gsc_fixed_partition()
        elif self.dataset == "gsc" and self.mode == "random":
            return self._gsc_random_partition(run_idx)
        elif self.dataset == "mswc":
            return self._mswc_random_partition(run_idx)
        else:
            raise ValueError(f"Unknown dataset/mode: {self.dataset}/{self.mode}")

    def _gsc_fixed_partition(self) -> tuple[list[str], list[str]]:
        """GSC Fixed protocol: deterministic word partition."""
        positive = list(GSC_POSITIVE_WORDS)
        available = [w for w in GSC_ALL_35_WORDS
                     if w not in positive and w not in GSC_EXCLUDED_WORDS]
        negative = available[:self.n_negative]
        return positive, negative

    def _gsc_random_partition(self, run_idx: int) -> tuple[list[str], list[str]]:
        """GSC Randomized: random selection from all 35 words (excl. excluded)."""
        rng = random.Random(self.seed + run_idx)
        available = [w for w in GSC_ALL_35_WORDS if w not in GSC_EXCLUDED_WORDS]
        selected = rng.sample(available, self.n_positive + self.n_negative)
        positive = selected[:self.n_positive]
        negative = selected[self.n_positive:]
        return positive, negative

    def _mswc_random_partition(self, run_idx: int) -> tuple[list[str], list[str]]:
        """MSWC Randomized: random from 263 eval words."""
        rng = random.Random(self.seed + run_idx)
        raise NotImplementedError(
            "MSWC partition requires the eval word list. "
            "Load 263 eval words from data/mswc_en/splits/eval_words.json first."
        )

    def evaluate(
        self,
        encoder: torch.nn.Module,
        classifier: Any,
        get_samples_fn: Any,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """Run full evaluation over n_runs.

        Args:
            encoder: DSCNN model (in eval mode).
            classifier: OpenNCMClassifier instance.
            get_samples_fn: Callable(word, n_samples) -> (mfcc_tensor, labels).
                Returns MFCC features and labels for a word.
            device: Computation device.

        Returns:
            Dict with averaged metrics: 'auc', 'frr_at_5far', 'det_curve', 'per_run'.
        """
        all_auc = []
        all_frr = []
        all_det_curves = []
        per_run_results = []

        encoder.eval()

        for run_idx in range(self.n_runs):
            run_seed = self.seed + run_idx
            positive_words, negative_words = self.get_partitions(run_idx)

            logger.info(
                "Run %d/%d: %d positive, %d negative words",
                run_idx + 1, self.n_runs,
                len(positive_words), len(negative_words),
            )

            with torch.no_grad():
                # Enroll: compute prototypes from k_shot support samples
                prototypes = []
                proto_labels = []
                for word in positive_words:
                    support_mfcc, _ = get_samples_fn(word, self.k_shot)
                    support_mfcc = support_mfcc.to(device)
                    emb = encoder(support_mfcc)
                    emb = F.normalize(emb, p=2, dim=-1)
                    prototype = emb.mean(dim=0)
                    prototypes.append(prototype)
                    proto_labels.append(word)

                proto_tensor = torch.stack(prototypes)
                classifier.set_prototypes(proto_tensor, proto_labels)

                # Query: evaluate on remaining samples
                y_true_list = []
                scores_list = []

                for word in positive_words:
                    query_mfcc, _ = get_samples_fn(word, -1)  # all remaining
                    query_mfcc = query_mfcc.to(device)
                    emb = encoder(query_mfcc)
                    emb = F.normalize(emb, p=2, dim=-1)
                    for i in range(emb.shape[0]):
                        dists = classifier.get_distances(emb[i])
                        min_dist = min(dists.values())
                        y_true_list.append(1)
                        scores_list.append(-min_dist)  # negate: lower dist = higher score

                for word in negative_words:
                    query_mfcc, _ = get_samples_fn(word, -1)
                    query_mfcc = query_mfcc.to(device)
                    emb = encoder(query_mfcc)
                    emb = F.normalize(emb, p=2, dim=-1)
                    for i in range(emb.shape[0]):
                        dists = classifier.get_distances(emb[i])
                        min_dist = min(dists.values())
                        y_true_list.append(0)
                        scores_list.append(-min_dist)

            y_true = np.array(y_true_list)
            scores = np.array(scores_list)

            auc = compute_auc(y_true, scores)
            frr = compute_frr_at_far(y_true, scores, target_far=0.05)
            far_curve, frr_curve = compute_det_curve(y_true, scores)

            all_auc.append(auc)
            all_frr.append(frr)
            all_det_curves.append((far_curve, frr_curve))
            per_run_results.append({"auc": auc, "frr_at_5far": frr, "seed": run_seed})

            logger.info("  AUC=%.4f, FRR@5%%FAR=%.4f", auc, frr)

        return {
            "auc": float(np.mean(all_auc)),
            "auc_std": float(np.std(all_auc)),
            "frr_at_5far": float(np.mean(all_frr)),
            "frr_at_5far_std": float(np.std(all_frr)),
            "n_runs": self.n_runs,
            "dataset": self.dataset,
            "mode": self.mode,
            "per_run": per_run_results,
        }
