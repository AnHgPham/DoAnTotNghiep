"""Run lightweight prototype calibration sweeps on saved embedding tensors.

Input is a ``.pt`` file with:
    {
      "support": {"word": Tensor[k, d], ...},
      "impostors": Tensor[n, d] optional
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.classifiers.calibration import build_prototype_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype calibration sweep")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("results/calibration_sweep.json"))
    args = parser.parse_args()

    payload = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    support = payload["support"]
    impostors = payload.get("impostors")
    results = {}
    for strategy in ("mean", "medoid"):
        bundle = build_prototype_bundle(
            support,
            strategy=strategy,
            impostor_embeddings=impostors,
            target_far=args.target_far,
        )
        results[strategy] = {
            "labels": bundle.labels,
            "thresholds": bundle.thresholds,
            "prototype_shape": list(bundle.prototypes.shape),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
