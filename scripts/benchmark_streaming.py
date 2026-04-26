"""Benchmark EXT-2 streaming pipeline latency on synthetic audio.

Measures:
- VAD latency per 32 ms chunk
- MFCC extraction latency per 1 s window
- DSCNN inference latency per 1 s window
- End-to-end Real-Time Factor (RTF) on a long synthetic audio

Usage:
    python scripts/benchmark_streaming.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.streaming.vad_engine import StreamingKWS

SR = 16000
N_WARMUP = 5
N_RUNS = 50


def time_ms(fn, n_runs: int) -> tuple[float, float]:
    """Run callable n_runs times, return (mean ms, std ms)."""
    samples = []
    for _ in range(N_WARMUP):
        fn()
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return mean(samples), stdev(samples) if len(samples) > 1 else 0.0


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DSCNN(model_size=cfg["model"]["architecture"][-1]).to(device)
    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    extractor = MFCCExtractor()
    print(f"Device: {device}, checkpoint epoch={ckpt.get('epoch', '?')}")

    chunk = torch.randn(512)
    one_sec = torch.randn(1, SR)

    rows: list[dict] = []

    def bench(name: str, fn, n_runs: int = N_RUNS) -> None:
        m, s = time_ms(fn, n_runs)
        rows.append({"step": name, "mean_ms": m, "std_ms": s})
        print(f"{name:<40} {m:>8.2f} +/- {s:5.2f} ms")

    print("\nPer-step latency:")
    print("-" * 70)

    bench("MFCC extraction (1 s window)", lambda: extractor.extract(one_sec))

    def encode():
        with torch.no_grad():
            mfcc = extractor.extract(one_sec).unsqueeze(0).to(device)
            emb = encoder(mfcc)
            F.normalize(emb, p=2, dim=-1)

    bench("DSCNN encode (1 s window)", encode)

    try:
        from src.streaming.vad_engine import SileroVAD
        vad = SileroVAD(threshold=0.5, device=device)
        bench("Silero VAD (32 ms chunk)", lambda: vad.is_speech(chunk))
        vad_available = True
    except Exception as exc:
        print(f"Silero VAD skipped: {exc}")
        vad = None
        vad_available = False

    print("-" * 70)

    print("\nEnd-to-end Real-Time Factor (RTF) on 30 s synthetic audio:")
    print("-" * 70)

    long_audio = torch.randn(1, SR * 30)
    proto = F.normalize(torch.randn(276), p=2, dim=0)
    prototypes = {"yes": proto}

    engine = StreamingKWS(
        encoder=encoder,
        mfcc_extractor=extractor,
        vad=None,
        window_size=SR,
        stride=SR // 2,
        device=device,
    )
    for _ in range(N_WARMUP):
        engine.process_file(long_audio[:, :SR * 5], prototypes, threshold=0.8)
    start = time.perf_counter()
    engine.process_file(long_audio, prototypes, threshold=0.8)
    elapsed = time.perf_counter() - start
    rtf = elapsed / 30.0
    print(f"{'no VAD':<40} {elapsed * 1000:>8.1f} ms total, RTF={rtf:.3f}")
    rows.append({"step": "end_to_end_30s (no VAD)", "mean_ms": elapsed * 1000, "rtf": rtf})

    if vad_available:
        chunk_count = (SR * 30) // 512
        chunks = [torch.randn(512) for _ in range(chunk_count)]
        for _ in range(N_WARMUP):
            for c in chunks[:5]:
                vad.is_speech(c)
        start = time.perf_counter()
        for c in chunks:
            vad.is_speech(c)
        elapsed_vad = time.perf_counter() - start
        rtf_vad = elapsed_vad / 30.0
        print(f"{'Silero VAD only on 30 s (chunked)':<40} {elapsed_vad * 1000:>8.1f} ms total, RTF={rtf_vad:.3f}")
        rows.append({"step": "vad_only_30s", "mean_ms": elapsed_vad * 1000, "rtf": rtf_vad})

    print("-" * 70)

    output_path = Path("results/streaming_latency.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
