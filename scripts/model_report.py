"""Report parameter count and profiler FLOPs for KWS encoders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bcresnet_fs import BCResNetFS
from src.models.dscnn import DSCNN
from src.models.edgespot_full import EdgeSpotFull
from src.models.edgespot_lite import EdgeSpotLite


def build_model(family: str, tau: int):
    if family == "dscnn":
        return DSCNN(model_size="L"), torch.randn(1, 1, 47, 10)
    if family == "edgespot_lite":
        return EdgeSpotLite(width_mult=tau), torch.randn(1, 1, 40, 101)
    if family == "edgespot_full":
        return EdgeSpotFull(tau=tau), torch.randn(1, 1, 40, 101)
    if family == "bcresnet_fs":
        return BCResNetFS(tau=tau), torch.randn(1, 1, 40, 101)
    raise ValueError(f"Unknown family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KWS model report")
    parser.add_argument("--family", choices=["dscnn", "edgespot_lite", "edgespot_full", "bcresnet_fs"],
                        default="edgespot_full")
    parser.add_argument("--tau", type=int, default=4)
    args = parser.parse_args()

    model, x = build_model(args.family, args.tau)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        y = model(x)
    print(f"family: {args.family}")
    print(f"tau: {args.tau}")
    print(f"params: {params}")
    print(f"input_shape: {tuple(x.shape)}")
    print(f"output_shape: {tuple(y.shape)}")

    try:
        with torch.profiler.profile(with_flops=True, record_shapes=False) as prof:
            with torch.no_grad():
                model(x)
        flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
        print(f"profiler_flops: {flops}")
        print(f"profiler_macs_approx: {flops / 2:.0f}")
    except Exception as exc:
        print(f"profiler_flops: unavailable ({exc})")


if __name__ == "__main__":
    main()
