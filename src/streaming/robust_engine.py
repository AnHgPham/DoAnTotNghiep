"""Robust streaming KWS engine with segmentation, multi-window scoring, and cooldown.

This module is inference-only. It does not touch training checkpoints or data.
It is designed for the deployment target where users enroll 3-5 examples per
keyword and then speak into a continuous microphone stream.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import torch

from src.streaming.enrollment import (
    EmbeddingBackend,
    EnrollmentProfile,
    MatchResult,
    SAMPLE_RATE,
)


@dataclass
class StreamingDecisionConfig:
    """Decision policy for robust streaming keyword detection."""

    sample_rate: int = SAMPLE_RATE
    candidate_window_ms: tuple[int, ...] = (600, 800, 1000, 1200)
    candidate_offsets_ms: tuple[int, ...] = (-120, 0, 120)
    energy_frame_ms: int = 30
    energy_hop_ms: int = 10
    energy_ratio: float = 0.12
    segment_pad_ms: int = 180
    min_segment_ms: int = 120
    min_margin: float = 0.05
    min_votes: int = 2
    cooldown_ms: int = 900
    threshold_scale: float = 1.0
    chunk_process_stride_ms: int = 250
    stream_buffer_ms: int = 3500


@dataclass
class StreamingEvent:
    """One accepted streaming detection."""

    keyword: str
    start_sec: float
    end_sec: float
    confidence: float
    distance: float
    threshold: float
    margin: float
    second_label: str | None
    speech_start_sec: float
    speech_end_sec: float
    distances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        ordered = sorted(self.distances.items(), key=lambda item: item[1])
        return {
            "state": "detected",
            "keyword": self.keyword,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "start_ms": round(self.start_sec * 1000),
            "end_ms": round(self.end_sec * 1000),
            "confidence": self.confidence,
            "distance": self.distance,
            "threshold": self.threshold,
            "margin": self.margin,
            "second_label": self.second_label,
            "speech_start_sec": self.speech_start_sec,
            "speech_end_sec": self.speech_end_sec,
            "all_distances": {
                label: float(distance) for label, distance in ordered
            },
            "top_3": [
                {"word": label, "dist": float(distance)}
                for label, distance in ordered[:3]
            ],
            "timestamp": time.time(),
        }


def _mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.float().cpu()


def energy_segments(
    waveform: torch.Tensor,
    config: StreamingDecisionConfig | None = None,
) -> list[tuple[int, int]]:
    """Find speech-like regions using frame energy."""
    cfg = config or StreamingDecisionConfig()
    wav = _mono(waveform).squeeze(0)
    total = wav.numel()
    if total == 0:
        return []

    frame = max(1, int(cfg.sample_rate * cfg.energy_frame_ms / 1000))
    hop = max(1, int(cfg.sample_rate * cfg.energy_hop_ms / 1000))
    if total < frame:
        return [(0, total)]

    frames = wav.unfold(0, frame, hop)
    energies = torch.sqrt(frames.square().mean(dim=-1) + 1e-8)
    starts = torch.arange(energies.numel(), dtype=torch.long) * hop
    max_energy = float(energies.max().item()) if energies.numel() else 0.0
    if max_energy <= 1e-6:
        return []

    threshold = max(0.0005, max_energy * cfg.energy_ratio)
    active: list[tuple[int, int]] = []
    cur_start: int | None = None
    for start, energy in zip(starts.tolist(), energies.tolist()):
        if energy >= threshold and cur_start is None:
            cur_start = start
        elif energy < threshold and cur_start is not None:
            active.append((cur_start, start + frame))
            cur_start = None
    if cur_start is not None:
        active.append((cur_start, int(starts[-1].item()) + frame))

    gap = int(cfg.sample_rate * 0.25)
    merged: list[tuple[int, int]] = []
    for start, end in active:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    pad = int(cfg.sample_rate * cfg.segment_pad_ms / 1000)
    min_len = int(cfg.sample_rate * cfg.min_segment_ms / 1000)
    return [
        (max(0, start - pad), min(total, end + pad))
        for start, end in merged
        if min(total, end + pad) - max(0, start - pad) >= min_len
    ]


def _candidate_windows(
    segment: tuple[int, int],
    total_samples: int,
    cfg: StreamingDecisionConfig,
) -> list[tuple[int, int]]:
    seg_start, seg_end = segment
    center = (seg_start + seg_end) // 2
    windows: list[tuple[int, int]] = []
    seen = set()
    for win_ms in cfg.candidate_window_ms:
        length = int(cfg.sample_rate * win_ms / 1000)
        for off_ms in cfg.candidate_offsets_ms:
            offset = int(cfg.sample_rate * off_ms / 1000)
            start = center - length // 2 + offset
            end = start + length
            if start < 0:
                end -= start
                start = 0
            if end > total_samples:
                start = max(0, start - (end - total_samples))
                end = total_samples
            if end <= start:
                continue
            key = (start, end)
            if key not in seen:
                windows.append(key)
                seen.add(key)
    return windows


class RobustStreamingKWS:
    """Streaming detector with speech segmentation and stable event decisions."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        profile: EnrollmentProfile,
        config: StreamingDecisionConfig | None = None,
        vad=None,
    ):
        self.backend = backend
        self.profile = profile
        self.config = config or StreamingDecisionConfig(sample_rate=backend.sample_rate)
        self.vad = vad
        self._max_buffer = int(self.config.sample_rate * self.config.stream_buffer_ms / 1000)
        self._buffer: torch.Tensor = torch.zeros(0, dtype=torch.float32)
        self._abs_samples_seen = 0
        self._samples_since_process = 0
        self._last_event_end_abs = -10**12

    def _segments(self, waveform: torch.Tensor) -> list[tuple[int, int]]:
        if self.vad is not None and hasattr(self.vad, "get_speech_timestamps"):
            try:
                timestamps = self.vad.get_speech_timestamps(_mono(waveform).squeeze(0))
                segments = [(int(t["start"]), int(t["end"])) for t in timestamps]
                if segments:
                    return segments
            except Exception:
                pass
        return energy_segments(waveform, self.config)

    def _embed_windows(self, windows: list[torch.Tensor]) -> torch.Tensor:
        if not windows:
            return torch.empty((0, 0), dtype=torch.float32)
        embed_batch = getattr(self.backend, "embed_batch", None)
        if callable(embed_batch):
            embeddings = embed_batch(windows)
        else:
            embed_many = getattr(self.backend, "embed_many", None)
            if not callable(embed_many):
                raise TypeError(
                    "Embedding backend must provide embed_batch() or embed_many()"
                )
            embeddings = embed_many(windows)
        if len(windows) != len(embeddings):
            raise RuntimeError(
                "Embedding backend returned a different number of embeddings "
                f"({len(embeddings)}) than candidate windows ({len(windows)})"
            )
        return embeddings

    def _event_from_scored(
        self,
        segment: tuple[int, int],
        scored: list[tuple[tuple[int, int], MatchResult]],
    ) -> tuple[StreamingEvent | None, list[MatchResult]]:
        if not scored:
            return None, []

        accepted = [(window, result) for window, result in scored if result.detected]
        if not accepted:
            return None, [result for _, result in scored]

        votes = Counter(result.label for _, result in accepted)
        best_label, vote_count = votes.most_common(1)[0]
        if vote_count < min(self.config.min_votes, len(accepted)):
            return None, [result for _, result in scored]

        best_window, best_result = max(
            [
                (window, result)
                for window, result in accepted
                if result.label == best_label
            ],
            key=lambda item: item[1].confidence,
        )
        start, end = best_window
        seg_start, seg_end = segment
        sr = self.config.sample_rate
        event = StreamingEvent(
            keyword=best_result.label,
            start_sec=start / sr,
            end_sec=end / sr,
            confidence=best_result.confidence,
            distance=best_result.distance,
            threshold=best_result.threshold,
            margin=best_result.margin,
            second_label=best_result.second_label,
            speech_start_sec=seg_start / sr,
            speech_end_sec=seg_end / sr,
            distances=best_result.distances,
        )
        return event, [result for _, result in scored]

    def _score_segment(
        self,
        waveform: torch.Tensor,
        segment: tuple[int, int],
    ) -> tuple[StreamingEvent | None, list[MatchResult]]:
        wav = _mono(waveform)
        total = wav.shape[-1]
        candidates = _candidate_windows(segment, total, self.config)
        scored: list[tuple[tuple[int, int], MatchResult]] = []

        if candidates:
            # Single batched forward over all candidate windows. This is
            # numerically identical to scoring each window with backend.embed
            # (encoder runs in eval mode -> batch-invariant outputs) but
            # amortizes feature-extraction and forward-pass overhead.
            windows = [wav[..., start:end] for start, end in candidates]
            embeddings = self._embed_windows(windows)
            for (start, end), emb in zip(candidates, embeddings):
                result = self.profile.score(
                    emb,
                    min_margin=self.config.min_margin,
                    threshold_scale=self.config.threshold_scale,
                )
                scored.append(((start, end), result))

        return self._event_from_scored(segment, scored)

    def process_file(self, waveform: torch.Tensor) -> list[dict]:
        """Detect keywords in a complete audio tensor."""
        wav = _mono(waveform)
        events: list[StreamingEvent] = []
        cooldown = int(self.config.sample_rate * self.config.cooldown_ms / 1000)
        last_event_end = -10**12
        total = wav.shape[-1]
        groups = [
            (segment, _candidate_windows(segment, total, self.config))
            for segment in self._segments(wav)
        ]
        all_windows = [
            wav[..., start:end]
            for _, candidates in groups
            for start, end in candidates
        ]
        all_embeddings = self._embed_windows(all_windows)
        offset = 0

        for segment, candidates in groups:
            segment_embeddings = all_embeddings[offset:offset + len(candidates)]
            offset += len(candidates)
            scored = []
            for window, embedding in zip(candidates, segment_embeddings):
                result = self.profile.score(
                    embedding,
                    min_margin=self.config.min_margin,
                    threshold_scale=self.config.threshold_scale,
                )
                scored.append((window, result))
            event, _ = self._event_from_scored(segment, scored)
            if event is None:
                continue
            event_start_sample = int(event.start_sec * self.config.sample_rate)
            event_end_sample = int(event.end_sec * self.config.sample_rate)
            if event_start_sample - last_event_end < cooldown:
                continue
            events.append(event)
            last_event_end = event_end_sample

        return [event.to_dict() for event in events]

    def append_samples(self, chunk: torch.Tensor) -> None:
        """Append PCM audio to the rolling buffer (cheap; runs no inference).

        Separating ingestion from inference lets the streaming server keep the
        network receive path non-blocking and run detection on its own cadence
        (see :meth:`process_buffer`), so latency stays bounded under load.
        """
        chunk_1d = _mono(chunk).squeeze(0).to(torch.float32)
        self._buffer = torch.cat([self._buffer, chunk_1d])
        if self._buffer.shape[0] > self._max_buffer:
            self._buffer = self._buffer[-self._max_buffer:].contiguous()
        self._abs_samples_seen += chunk_1d.numel()
        self._samples_since_process += chunk_1d.numel()

    def ready_to_process(self) -> bool:
        """True once enough fresh audio has accumulated for a process cycle."""
        stride = int(self.config.sample_rate * self.config.chunk_process_stride_ms / 1000)
        return (
            self._buffer.shape[0] >= self.config.sample_rate
            and self._samples_since_process >= stride
        )

    def process_buffer(self) -> list[dict]:
        """Run detection on the current rolling buffer (CPU-heavy step).

        Gated by :meth:`ready_to_process`, so calling it on a fixed cadence is
        safe: sparse/early calls are cheap no-ops. Returns newly accepted events.
        """
        if not self.ready_to_process():
            return []
        self._samples_since_process = 0

        buffer_tensor = self._buffer.unsqueeze(0)
        buffer_start_abs = self._abs_samples_seen - buffer_tensor.shape[-1]
        cooldown = int(self.config.sample_rate * self.config.cooldown_ms / 1000)

        new_events = []
        for event in self.process_file(buffer_tensor):
            start_abs = buffer_start_abs + int(event["start_sec"] * self.config.sample_rate)
            end_abs = buffer_start_abs + int(event["end_sec"] * self.config.sample_rate)
            if start_abs - self._last_event_end_abs < cooldown:
                continue
            event["start_sec"] = start_abs / self.config.sample_rate
            event["end_sec"] = end_abs / self.config.sample_rate
            event["start_ms"] = round(event["start_sec"] * 1000)
            event["end_ms"] = round(event["end_sec"] * 1000)
            event["speech_start_sec"] = (
                buffer_start_abs / self.config.sample_rate + event["speech_start_sec"]
            )
            event["speech_end_sec"] = (
                buffer_start_abs / self.config.sample_rate + event["speech_end_sec"]
            )
            new_events.append(event)
            self._last_event_end_abs = end_abs
        return new_events

    def process_chunk(self, chunk: torch.Tensor) -> list[dict]:
        """Append a chunk of PCM audio and return newly accepted events.

        Convenience composition of :meth:`append_samples` + :meth:`process_buffer`;
        behavior is unchanged for file/offline callers.
        """
        self.append_samples(chunk)
        return self.process_buffer()

    def reset(self) -> None:
        self._buffer = torch.zeros(0, dtype=torch.float32)
        self._abs_samples_seen = 0
        self._samples_since_process = 0
        self._last_event_end_abs = -10**12
        if self.vad is not None and hasattr(self.vad, "reset_states"):
            self.vad.reset_states()
