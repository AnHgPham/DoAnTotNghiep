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
    compute_eer,
    compute_frr_at_far,
    compute_keyword_accuracy,
    compute_open_set_acc_at_far,
    compute_precision_recall_f1,
)

logger = logging.getLogger(__name__)

GSC_POSITIVE_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
GSC_EXCLUDED_WORDS = ["backward", "forward", "visual", "follow", "learn"]

# EdgeSpot protocol: 11 target keywords (10 commands + silence proxy "marvin")
EDGESPOT_TARGET_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "marvin",
]
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
        scoring_method: str = "l2",
        normalize_embeddings: bool = True,
    ):
        self.dataset = dataset
        self.mode = mode
        self.n_runs = n_runs
        self.n_way = n_way
        self.k_shot = k_shot
        self.seed = seed
        self.normalize_embeddings = normalize_embeddings
        if scoring_method not in ("l2", "probability", "scaled_l2", "openmax", "energy"):
            raise ValueError(
                f"scoring_method must be 'l2' | 'probability' | 'scaled_l2' | "
                f"'openmax' | 'energy', got {scoring_method!r}"
            )
        self.scoring_method = scoring_method

        if dataset == "gsc" and mode == "fixed":
            self.n_positive = 10
            self.n_negative = 20
        elif dataset == "gsc" and mode == "random":
            self.n_positive = 10
            self.n_negative = 20
        elif dataset == "gsc" and mode == "edgespot":
            self.n_positive = 11
            self.n_negative = 24  # 35 - 11 = 24
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
        elif self.dataset == "gsc" and self.mode == "edgespot":
            return self._gsc_edgespot_partition(run_idx)
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

    def _gsc_edgespot_partition(self, run_idx: int) -> tuple[list[str], list[str]]:
        """EdgeSpot protocol: 11 target keywords vs 24 remaining, 100 random trials."""
        rng = random.Random(self.seed + run_idx)
        positive = list(EDGESPOT_TARGET_WORDS)
        rng.shuffle(positive)
        negative = [w for w in GSC_ALL_35_WORDS if w not in positive]
        rng.shuffle(negative)
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

        if not hasattr(self, "_mswc_eval_words"):
            self._mswc_eval_words = self._load_mswc_eval_words()

        total_needed = self.n_positive + self.n_negative
        selected = rng.sample(self._mswc_eval_words, min(total_needed, len(self._mswc_eval_words)))
        positive = selected[: self.n_positive]
        negative = selected[self.n_positive :]
        return positive, negative

    @staticmethod
    def _load_mswc_eval_words() -> list[str]:
        """Load MSWC eval word list from splits directory."""
        import json

        candidates = [
            Path("data/mswc_en/splits/eval_words.json"),
            Path("data/mswc_en/eval_words.json"),
        ]
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    words = json.load(f)
                logger.info("Loaded %d MSWC eval words from %s", len(words), path)
                return words

        mswc_dir = Path("data/mswc_en/clips")
        if mswc_dir.exists():
            words = sorted(
                d.name for d in mswc_dir.iterdir()
                if d.is_dir() and any(d.glob("*.wav"))
            )
            if words:
                logger.info("Discovered %d MSWC words from directory listing", len(words))
                return words

        raise FileNotFoundError(
            "Cannot find MSWC eval words. Expected one of: "
            "data/mswc_en/splits/eval_words.json, "
            "data/mswc_en/eval_words.json, or "
            "data/mswc_en/clips/ with word directories."
        )

    def evaluate(
        self,
        encoder: torch.nn.Module,
        classifier: Any,
        sample_provider: Any,
        device: torch.device = torch.device("cpu"),
        target_far: float = 0.05,
    ) -> dict:
        """Run full evaluation over n_runs.

        Args:
            encoder: DSCNN model (in eval mode).
            classifier: OpenNCMClassifier instance.
            sample_provider: Object exposing get_support_samples(word, n_samples,
                seed) and get_query_samples(word) methods.
            device: Computation device.
            target_far: FAR operating point for thresholded metrics.

        Returns:
            Dict with averaged metrics and per-run breakdown.
        """
        all_auc = []
        all_eer = []
        all_frr = []
        all_open_set_acc = []
        all_keyword_acc = []
        all_precision = []
        all_recall = []
        all_f1 = []
        per_run_results = []

        encoder.eval()

        for run_idx in range(self.n_runs):
            run_seed = self.seed + run_idx
            positive_words, negative_words = self.get_partitions(run_idx)
            sample_provider.validate_words(
                positive_words + negative_words,
                min_support=self.k_shot,
            )

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
                    support_mfcc, _ = sample_provider.get_support_samples(
                        word,
                        self.k_shot,
                        seed=run_seed,
                    )
                    support_mfcc = support_mfcc.to(device)
                    emb = encoder(support_mfcc)
                    if self.normalize_embeddings:
                        emb = F.normalize(emb, p=2, dim=-1)
                    prototype = emb.mean(dim=0)
                    prototypes.append(prototype)
                    proto_labels.append(word)

                proto_tensor = torch.stack(prototypes)
                classifier.set_prototypes(proto_tensor, proto_labels)

                # Calibrate per-prototype thresholds and intra-class std from support distances.
                per_proto_thresholds: dict[str, float] = {}
                proto_stds: dict[str, float] = {}
                proto_means: dict[str, float] = {}
                global_dists: list[float] = []
                support_distances: dict[str, list[float]] = {}
                alpha = 2.0
                for word_idx, word in enumerate(positive_words):
                    sup_mfcc, _ = sample_provider.get_support_samples(
                        word, self.k_shot, seed=run_seed,
                    )
                    sup_mfcc = sup_mfcc.to(device)
                    sup_emb = encoder(sup_mfcc)
                    if self.normalize_embeddings:
                        sup_emb = F.normalize(sup_emb, p=2, dim=-1)
                    proto_dists = []
                    for i in range(sup_emb.shape[0]):
                        d = torch.dist(sup_emb[i], prototypes[word_idx], p=2).item()
                        proto_dists.append(d)
                        global_dists.append(d)
                    support_distances[word] = proto_dists
                    mean_d = float(np.mean(proto_dists))
                    std_d = float(np.std(proto_dists)) if len(proto_dists) > 1 else 0.0
                    proto_means[word] = mean_d
                    proto_stds[word] = max(std_d, 1e-3)  # avoid divide-by-zero
                    per_proto_thresholds[word] = mean_d + alpha * std_d

                if self.scoring_method == "openmax":
                    classifier.fit_weibull(support_distances)
                    sample_w = ", ".join(
                        f"{label}(shape={p[0]:.2f},scale={p[2]:.3f})"
                        for label, p in list(classifier._weibull_params.items())[:3]
                    )
                    logger.info("  Weibull params (first 3): %s, ...", sample_w)
                elif self.scoring_method == "energy":
                    logger.info(
                        "  Energy classifier (T=%.2f), no per-proto calibration",
                        getattr(classifier, "temperature", float("nan")),
                    )
                elif per_proto_thresholds:
                    classifier.set_per_prototype_thresholds(per_proto_thresholds)
                    classifier.threshold = float(
                        np.mean(global_dists) + alpha * np.std(global_dists)
                    )
                    sample_thr = ", ".join(
                        f"{label}={value:.3f}" for label, value in list(per_proto_thresholds.items())[:3]
                    )
                    logger.info(
                        "  Per-prototype thresholds (first 3): %s, ... (global=%.4f)",
                        sample_thr, classifier.threshold,
                    )

                # Query: evaluate on remaining samples
                y_true_list = []
                y_true_labels = []
                y_pred_labels = []
                scores_list = []

                def _score_query(query_word: str, is_known: int, true_label: str) -> None:
                    query_mfcc, _ = sample_provider.get_query_samples(query_word)
                    query_mfcc = query_mfcc.to(device)
                    emb = encoder(query_mfcc)
                    if self.normalize_embeddings:
                        emb = F.normalize(emb, p=2, dim=-1)
                    for i in range(emb.shape[0]):
                        dists = classifier.get_distances(emb[i])
                        if self.scoring_method == "openmax":
                            scores = classifier.get_scores(emb[i])
                            # Decouple classification (argmin distance) from
                            # rejection score (Weibull-revised confidence).
                            # Argmax-score classification confused classes
                            # with wider Weibull scale; argmin-dist matches
                            # the OpenMAX paper's modified-softmax behaviour.
                            pred_label = min(dists, key=dists.get)
                            score = scores[pred_label]
                            min_dist = dists[pred_label]
                        elif self.scoring_method == "energy":
                            pred_label = min(dists, key=dists.get)
                            score = classifier.get_energy(emb[i])
                            min_dist = dists[pred_label]
                        elif self.scoring_method == "scaled_l2":
                            scaled = {
                                label: (d - proto_means.get(label, 0.0)) / proto_stds.get(label, 1.0)
                                for label, d in dists.items()
                            }
                            min_label = min(scaled, key=scaled.get)
                            score = -scaled[min_label]
                            min_dist = dists[min_label]
                            pred_label = min_label
                        elif self.scoring_method == "probability":
                            min_dist = min(dists.values())
                            pred_label = min(dists, key=dists.get)
                            dist_array = np.array(list(dists.values()), dtype=float)
                            logits = -dist_array
                            logits -= logits.max()
                            probs = np.exp(logits)
                            probs /= probs.sum()
                            score = float(probs.max())
                        else:
                            min_dist = min(dists.values())
                            pred_label = min(dists, key=dists.get)
                            score = -min_dist
                        y_true_list.append(is_known)
                        y_true_labels.append(true_label)
                        y_pred_labels.append(pred_label)
                        scores_list.append(score)

                for word in positive_words:
                    _score_query(word, 1, word)
                for word in negative_words:
                    _score_query(word, 0, "unknown")

            y_true = np.array(y_true_list)
            scores = np.array(scores_list)

            auc = compute_auc(y_true, scores)
            eer, eer_thr = compute_eer(y_true, scores)
            frr = compute_frr_at_far(y_true, scores, target_far=target_far)
            open_set_acc = compute_open_set_acc_at_far(
                y_true,
                y_true_labels,
                y_pred_labels,
                scores,
                target_far=target_far,
            )
            keyword_acc = compute_keyword_accuracy(
                [label for label, is_known in zip(y_true_labels, y_true, strict=True) if is_known == 1],
                [label for label, is_known in zip(y_pred_labels, y_true, strict=True) if is_known == 1],
            )
            far_curve, frr_curve = compute_det_curve(y_true, scores)

            y_pred_binary = (scores >= eer_thr).astype(int)
            prf = compute_precision_recall_f1(y_true, y_pred_binary)

            all_auc.append(auc)
            all_eer.append(eer)
            all_frr.append(frr)
            all_open_set_acc.append(open_set_acc)
            all_keyword_acc.append(keyword_acc)
            all_precision.append(prf["precision"])
            all_recall.append(prf["recall"])
            all_f1.append(prf["f1"])
            per_run_results.append(
                {
                    "auc": auc,
                    "eer": eer,
                    "frr_at_far": frr,
                    "open_set_acc_at_far": open_set_acc,
                    "keyword_acc": keyword_acc,
                    "precision": prf["precision"],
                    "recall": prf["recall"],
                    "f1": prf["f1"],
                    "seed": run_seed,
                    "det_curve": {
                        "far": far_curve.tolist(),
                        "frr": frr_curve.tolist(),
                    },
                }
            )

            logger.info(
                "  AUC=%.4f, EER=%.4f, FRR@%.1f%%FAR=%.4f, "
                "open-set-acc=%.4f, keyword-acc=%.4f, F1=%.4f",
                auc, eer, target_far * 100, frr,
                open_set_acc, keyword_acc, prf["f1"],
            )

        results = {
            "auc": float(np.mean(all_auc)),
            "auc_std": float(np.std(all_auc)),
            "eer": float(np.mean(all_eer)),
            "eer_std": float(np.std(all_eer)),
            "frr_at_far": float(np.mean(all_frr)),
            "frr_at_far_std": float(np.std(all_frr)),
            "open_set_acc_at_far": float(np.mean(all_open_set_acc)),
            "open_set_acc_at_far_std": float(np.std(all_open_set_acc)),
            "keyword_acc": float(np.mean(all_keyword_acc)),
            "keyword_acc_std": float(np.std(all_keyword_acc)),
            "precision": float(np.mean(all_precision)),
            "precision_std": float(np.std(all_precision)),
            "recall": float(np.mean(all_recall)),
            "recall_std": float(np.std(all_recall)),
            "f1": float(np.mean(all_f1)),
            "f1_std": float(np.std(all_f1)),
            "target_far": target_far,
            "n_runs": self.n_runs,
            "dataset": self.dataset,
            "mode": self.mode,
            "per_run": per_run_results,
        }

        if np.isclose(target_far, 0.05):
            results["frr_at_5far"] = results["frr_at_far"]
            results["frr_at_5far_std"] = results["frr_at_far_std"]
            results["open_set_acc_at_5far"] = results["open_set_acc_at_far"]
            results["open_set_acc_at_5far_std"] = results["open_set_acc_at_far_std"]

        return results
