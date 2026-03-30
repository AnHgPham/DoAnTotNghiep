"""Evaluation script for few-shot open-set KWS.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --protocol gsc_random
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

from src.classifiers.open_ncm import OpenNCMClassifier
from src.evaluation.metrics import plot_det_curves
from src.evaluation.protocols import EvaluationProtocol
from src.models.dscnn import DSCNN

logger = logging.getLogger(__name__)

PROTOCOL_MAP = {
    "gsc_fixed": ("gsc", "fixed"),
    "gsc_random": ("gsc", "random"),
    "mswc_random": ("mswc", "random"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KWS model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--protocol", type=str, default="gsc_fixed",
        choices=list(PROTOCOL_MAP.keys()),
    )
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

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

    protocol = EvaluationProtocol(
        dataset=dataset,
        mode=mode,
        n_runs=n_runs,
        n_way=cfg["evaluation"]["n_way"],
        k_shot=cfg["evaluation"]["k_shot"],
        seed=cfg["seed"],
    )

    classifier = OpenNCMClassifier()

    logger.info("=" * 60)
    logger.info("Protocol: %s (dataset=%s, mode=%s, n_runs=%d)", args.protocol, dataset, mode, n_runs)
    logger.info("=" * 60)
    logger.info(
        "NOTE: Evaluation requires dataset loaded. "
        "Implement get_samples_fn for your dataset loader."
    )

    # TODO: Wire up actual dataset loading
    # results = protocol.evaluate(encoder, classifier, get_samples_fn, device)
    #
    # Save results
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    # with open(output_dir / f"{args.protocol}_results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    # logger.info("Results saved to %s", output_dir / f"{args.protocol}_results.json")


if __name__ == "__main__":
    main()
