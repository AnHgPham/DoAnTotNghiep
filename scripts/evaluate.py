"""Evaluation script for few-shot open-set KWS.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --protocol gsc_random
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.energy_ood import EnergyOODClassifier
from src.classifiers.open_ncm import OpenNCMClassifier
from src.classifiers.openmax import OpenMAXClassifier
from src.evaluation.gsc import GSCFewShotProvider
from src.evaluation.metrics import plot_det_curves
from src.evaluation.protocols import EvaluationProtocol
from src.models.dscnn import DSCNN

logger = logging.getLogger(__name__)

PROTOCOL_MAP = {
    "gsc_fixed": ("gsc", "fixed"),
    "gsc_random": ("gsc", "random"),
    "gsc_edgespot": ("gsc", "edgespot"),
    "mswc_random": ("mswc", "random"),
}


def _mean_det_curve(results: dict, n_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Average saved per-run DET curves on a common FAR grid."""
    common_far = np.linspace(0, 1, n_points)
    frr_curves = []

    for run in results.get("per_run", []):
        curve = run.get("det_curve", {})
        far = np.asarray(curve.get("far", []), dtype=float)
        frr = np.asarray(curve.get("frr", []), dtype=float)
        if far.size == 0 or frr.size == 0:
            continue
        frr_curves.append(np.interp(common_far, far, frr))

    if not frr_curves:
        return common_far, np.ones_like(common_far)

    return common_far, np.mean(frr_curves, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KWS model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--protocol", type=str, default="gsc_fixed",
        choices=list(PROTOCOL_MAP.keys()),
    )
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None,
                        help="Override config evaluation.k_shot")
    parser.add_argument("--n-way", type=int, default=None,
                        help="Override config evaluation.n_way")
    parser.add_argument("--scoring", type=str, default="l2",
                        choices=["l2", "probability", "scaled_l2", "openmax", "energy"],
                        help="l2 = Direct L2 distance (proposed); "
                             "probability = softmax over -distances (Rusci baseline); "
                             "scaled_l2 = per-prototype variance-normalized L2; "
                             "openmax = Weibull tail-fit OpenMAX score; "
                             "energy = T*logsumexp(-d/T) energy-based OOD score")
    parser.add_argument("--classifier", type=str, default="openncm",
                        choices=["openncm", "openmax", "energy"],
                        help="openncm = OpenNCMClassifier (L2-distance baseline); "
                             "openmax = OpenMAXClassifier (Weibull tail-fit); "
                             "energy = EnergyOODClassifier (energy-based OOD). "
                             "Selecting openmax/energy forces matching --scoring.")
    parser.add_argument("--tail-size", type=int, default=20,
                        help="OpenMAX Weibull tail size (effective: min(tail_size, k_shot))")
    parser.add_argument("--openmax-mode", type=str, default="per_class",
                        choices=["per_class", "global"],
                        help="OpenMAX Weibull mode. 'global' pools all support "
                             "distances into a single Weibull (more stable for low-shot).")
    parser.add_argument("--openmax-hybrid", type=float, default=0.0,
                        help="OpenMAX hybrid score weight in [0,1]: "
                             "score = (1-alpha)*(1-cdf) + alpha*exp(-d). 0 = pure Weibull.")
    parser.add_argument("--energy-temperature", type=float, default=1.0,
                        help="Energy softmin temperature T (>0). "
                             "Lower T -> energy approaches -min(d) (= L2 baseline).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip L2 normalization of embeddings. Tests whether "
                             "raw embedding magnitude carries OOD signal.")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--plot-det",
        action="store_true",
        help="Save the mean DET curve PNG after evaluation.",
    )
    parser.add_argument(
        "--det-output",
        type=str,
        default=None,
        help="Optional output path for DET curve PNG.",
    )
    args = parser.parse_args()

    if args.classifier == "openmax":
        args.scoring = "openmax"
    elif args.classifier == "energy":
        args.scoring = "energy"
    elif args.scoring == "openmax" and args.classifier != "openmax":
        parser.error("--scoring openmax requires --classifier openmax")
    elif args.scoring == "energy" and args.classifier != "energy":
        parser.error("--scoring energy requires --classifier energy")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    model_size = cfg["model"]["architecture"][-1]
    encoder = DSCNN(model_size=model_size).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    logger.info("Loaded checkpoint: %s (epoch %d)", args.checkpoint, checkpoint["epoch"])

    # Setup protocol
    dataset, mode = PROTOCOL_MAP[args.protocol]
    n_runs = args.n_runs or cfg["evaluation"]["n_runs"]

    k_shot = args.k_shot or cfg["evaluation"]["k_shot"]
    n_way = args.n_way or cfg["evaluation"]["n_way"]
    protocol = EvaluationProtocol(
        dataset=dataset,
        mode=mode,
        n_runs=n_runs,
        n_way=n_way,
        k_shot=k_shot,
        seed=cfg["seed"],
        scoring_method=args.scoring,
        normalize_embeddings=not args.no_normalize,
    )

    if args.classifier == "openmax":
        classifier = OpenMAXClassifier(
            tail_size=args.tail_size,
            mode=args.openmax_mode,
            hybrid_alpha=args.openmax_hybrid,
        )
    elif args.classifier == "energy":
        classifier = EnergyOODClassifier(temperature=args.energy_temperature)
    else:
        classifier = OpenNCMClassifier()

    logger.info("=" * 60)
    logger.info(
        "Protocol: %s (dataset=%s, mode=%s, n_runs=%d, classifier=%s, scoring=%s)",
        args.protocol, dataset, mode, n_runs, args.classifier, args.scoring,
    )
    logger.info("=" * 60)

    if dataset == "gsc":
        sample_provider = GSCFewShotProvider(cfg["data"]["gsc_dir"])
    elif dataset == "mswc":
        from src.evaluation.mswc import MSWCFewShotProvider
        sample_provider = MSWCFewShotProvider(cfg["data"]["mswc_dir"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    results = protocol.evaluate(
        encoder,
        classifier,
        sample_provider,
        device=device,
        target_far=cfg["evaluation"]["target_far"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_k{k_shot}" if k_shot != cfg["evaluation"]["k_shot"] else ""
    if args.classifier != "openncm":
        suffix += f"_{args.classifier}"
    elif args.scoring != "l2":
        suffix += f"_{args.scoring}"
    if args.no_normalize:
        suffix += "_nonorm"
    output_path = output_dir / f"{args.protocol}{suffix}_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_path)

    if args.plot_det or args.det_output:
        det_path = Path(args.det_output) if args.det_output else output_dir / f"{args.protocol}_det_curve.png"
        far, frr = _mean_det_curve(results)
        plot_det_curves(
            {args.protocol: (far, frr)},
            save_path=det_path,
            title=f"{args.protocol} DET Curve",
        )
        logger.info("DET curve saved to %s", det_path)

    logger.info("=" * 50)
    logger.info("AUC:          %.4f +/- %.4f", results["auc"], results["auc_std"])
    logger.info("EER:          %.4f +/- %.4f", results["eer"], results["eer_std"])
    logger.info(
        "FRR@%.1f%%FAR:  %.4f +/- %.4f",
        results["target_far"] * 100,
        results["frr_at_far"],
        results["frr_at_far_std"],
    )
    logger.info(
        "Open-set ACC: %.4f +/- %.4f",
        results["open_set_acc_at_far"],
        results["open_set_acc_at_far_std"],
    )
    logger.info(
        "Keyword ACC:  %.4f +/- %.4f",
        results["keyword_acc"],
        results["keyword_acc_std"],
    )
    logger.info(
        "Precision:    %.4f +/- %.4f",
        results["precision"],
        results["precision_std"],
    )
    logger.info(
        "Recall:       %.4f +/- %.4f",
        results["recall"],
        results["recall_std"],
    )
    logger.info(
        "F1:           %.4f +/- %.4f",
        results["f1"],
        results["f1_std"],
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
