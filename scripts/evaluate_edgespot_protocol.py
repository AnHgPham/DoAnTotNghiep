"""Canonical EdgeSpot-style GSC evaluation wrapper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical gsc_edgespot_exact evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-family", default="auto",
                        choices=["auto", "dscnn", "edgespot_lite", "edgespot_full", "bcresnet_fs"])
    parser.add_argument("--edge-tau", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--gsc-query-split", choices=["dev", "test"], default="test")
    parser.add_argument("--output-dir", default="results/edgespot_exact")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--model-family", args.model_family,
        "--protocol", "gsc_edgespot_exact",
        "--k-shot", str(args.k_shot),
        "--n-runs", str(args.n_runs),
        "--gsc-query-split", args.gsc_query_split,
        "--output-dir", args.output_dir,
        "--plot-det",
    ]
    if args.edge_tau is not None:
        cmd.extend(["--edge-tau", str(args.edge_tau)])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
