"""Benchmark robust streaming few-shot KWS on concatenated GSC audio.

This script is meant for post-training validation. It does not train or modify
checkpoints. It enrolls 3-5 samples per target keyword, builds a long stream
with target and unknown words, runs RobustStreamingKWS, and reports deployment
metrics: miss rate, false alarms/hour, duplicate detections, and latency.

Usage:
    python scripts/benchmark_robust_streaming.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.streaming.enrollment import EmbeddingBackend, build_enrollment_profile
from src.streaming.robust_engine import RobustStreamingKWS, StreamingDecisionConfig

SR = 16000
GSC_ALL_35 = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual",
]


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[-1] < SR:
        wav = F.pad(wav, (0, SR - wav.shape[-1]))
    return wav[..., :SR]


def word_files(gsc_dir: Path, word: str) -> list[Path]:
    return sorted((gsc_dir / word).glob("*.wav"))


def build_stream(
    gsc_dir: Path,
    target_words: list[str],
    unknown_words: list[str],
    k_shot: int,
    queries_per_word: int,
    silence_ms: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]], list[dict]]:
    rng = random.Random(seed)
    support: dict[str, list[torch.Tensor]] = {}
    query_items: list[tuple[str, torch.Tensor, bool]] = []

    for word in target_words:
        files = word_files(gsc_dir, word)
        if len(files) < k_shot + queries_per_word:
            raise RuntimeError(f"Not enough files for {word}: {len(files)}")
        rng.shuffle(files)
        support[word] = [load_wav(p) for p in files[:k_shot]]
        for p in files[k_shot:k_shot + queries_per_word]:
            query_items.append((word, load_wav(p), True))

    for word in unknown_words:
        files = word_files(gsc_dir, word)
        if not files:
            continue
        rng.shuffle(files)
        for p in files[:queries_per_word]:
            query_items.append((word, load_wav(p), False))

    rng.shuffle(query_items)
    silence = torch.zeros(1, int(SR * silence_ms / 1000))
    chunks = []
    timeline = []
    cursor = 0
    for label, wav, is_target in query_items:
        start = cursor
        end = cursor + wav.shape[-1]
        timeline.append({
            "label": label,
            "is_target": is_target,
            "start_sec": start / SR,
            "end_sec": end / SR,
        })
        chunks.append(wav)
        cursor = end
        chunks.append(silence)
        cursor += silence.shape[-1]

    return torch.cat(chunks, dim=-1), support, timeline


def match_events(events: list[dict], timeline: list[dict], tolerance_sec: float = 0.35) -> dict:
    hits = 0
    misses = 0
    false_alarms = 0
    duplicates = 0
    latencies = []
    matched_event_idx = set()

    target_items = [item for item in timeline if item["is_target"]]
    for item in target_items:
        matches = []
        for idx, event in enumerate(events):
            center = 0.5 * (event["start_sec"] + event["end_sec"])
            inside = item["start_sec"] - tolerance_sec <= center <= item["end_sec"] + tolerance_sec
            if inside and event["keyword"] == item["label"]:
                matches.append((idx, event))
        if matches:
            hits += 1
            first_idx, first_event = matches[0]
            matched_event_idx.add(first_idx)
            duplicates += max(0, len(matches) - 1)
            latencies.append(first_event["start_sec"] - item["start_sec"])
            for extra_idx, _ in matches[1:]:
                matched_event_idx.add(extra_idx)
        else:
            misses += 1

    for idx, event in enumerate(events):
        if idx in matched_event_idx:
            continue
        center = 0.5 * (event["start_sec"] + event["end_sec"])
        on_any_target = any(
            item["start_sec"] - tolerance_sec <= center <= item["end_sec"] + tolerance_sec
            and item["is_target"]
            for item in timeline
        )
        if not on_any_target:
            false_alarms += 1

    duration_sec = timeline[-1]["end_sec"] if timeline else 0.0
    return {
        "targets": len(target_items),
        "hits": hits,
        "misses": misses,
        "miss_rate": misses / max(len(target_items), 1),
        "false_alarms": false_alarms,
        "false_alarms_per_hour": false_alarms / max(duration_sec / 3600.0, 1e-6),
        "duplicates": duplicates,
        "mean_latency_sec": float(sum(latencies) / len(latencies)) if latencies else None,
        "duration_sec": duration_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark robust streaming KWS")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gsc-dir", type=Path, default=Path("data/gsc_v2"))
    parser.add_argument("--words", default="yes,no,stop,go,up,down,left,right,on,off")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--queries-per-word", type=int, default=5)
    parser.add_argument("--silence-ms", type=int, default=700)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/robust_streaming_benchmark.json"))
    args = parser.parse_args()

    target_words = [w.strip().lower() for w in args.words.split(",") if w.strip()]
    unknown_words = [w for w in GSC_ALL_35 if w not in set(target_words)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DSCNN(model_size="L").to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()

    stream, support, timeline = build_stream(
        args.gsc_dir,
        target_words=target_words,
        unknown_words=unknown_words,
        k_shot=args.k_shot,
        queries_per_word=args.queries_per_word,
        silence_ms=args.silence_ms,
        seed=args.seed,
    )

    backend = EmbeddingBackend(encoder, MFCCExtractor(), device=device)
    profile = build_enrollment_profile(support, backend, views_per_sample=5)
    cfg = StreamingDecisionConfig(
        threshold_scale=args.threshold_scale,
        min_margin=args.min_margin,
        min_votes=args.min_votes,
    )
    engine = RobustStreamingKWS(backend, profile, config=cfg)
    events = engine.process_file(stream)
    metrics = match_events(events, timeline)

    payload = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "target_words": target_words,
        "k_shot": args.k_shot,
        "queries_per_word": args.queries_per_word,
        "policy": {
            "threshold_scale": args.threshold_scale,
            "min_margin": args.min_margin,
            "min_votes": args.min_votes,
        },
        "metrics": metrics,
        "events": events,
        "timeline": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"metrics": metrics, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
