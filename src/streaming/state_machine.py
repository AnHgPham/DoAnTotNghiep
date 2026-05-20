"""CPU-only streaming decision state machine.

This layer is intentionally independent from the embedding model. It accepts
candidate detections produced by any backend and turns them into stable UI/API
events with smoothing, top-2 margin checks, and cooldown.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


STATE_IDLE = "idle"
STATE_SCORING = "scoring"
STATE_DETECTED = "detected"
STATE_REJECTED = "rejected"
STATE_COOLDOWN = "cooldown"


@dataclass(frozen=True)
class DetectionCandidate:
    """One candidate decision before smoothing/cooldown."""

    detected: bool
    keyword: str = "unknown"
    confidence: float = 0.0
    distance: float = float("inf")
    threshold: float = 0.0
    margin: float = 0.0
    second_label: str | None = None
    start_ms: int = 0
    end_ms: int = 0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DetectionCandidate":
        start_ms = payload.get("start_ms")
        end_ms = payload.get("end_ms")
        if start_ms is None and payload.get("start_sec") is not None:
            start_ms = round(float(payload["start_sec"]) * 1000)
        if end_ms is None and payload.get("end_sec") is not None:
            end_ms = round(float(payload["end_sec"]) * 1000)
        return cls(
            detected=bool(payload.get("detected", False)),
            keyword=str(payload.get("keyword") or payload.get("best_label") or "unknown"),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            distance=float(payload.get("distance", float("inf"))),
            threshold=float(payload.get("threshold", 0.0) or 0.0),
            margin=float(payload.get("margin", 0.0) or 0.0),
            second_label=payload.get("second_label"),
            start_ms=int(start_ms or 0),
            end_ms=int(end_ms or start_ms or 0),
        )


@dataclass(frozen=True)
class StreamingStateConfig:
    smoothing_window: int = 3
    min_votes: int = 2
    min_margin: float = 0.05
    min_confidence: float = 0.0
    cooldown_ms: int = 900


class StreamingStateMachine:
    """Stabilize candidate detections into public streaming events."""

    def __init__(self, config: StreamingStateConfig | None = None):
        self.config = config or StreamingStateConfig()
        self._recent: deque[DetectionCandidate] = deque(maxlen=max(1, self.config.smoothing_window))
        self._last_detection_end_ms = -10**12
        self.state = STATE_IDLE

    def reset(self) -> None:
        self._recent.clear()
        self._last_detection_end_ms = -10**12
        self.state = STATE_IDLE

    def update(
        self,
        candidate: DetectionCandidate | dict[str, Any] | None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        if candidate is None:
            self.state = STATE_IDLE
            return self._snapshot(now_ms)
        if isinstance(candidate, dict):
            candidate = DetectionCandidate.from_mapping(candidate)

        if candidate.start_ms - self._last_detection_end_ms < self.config.cooldown_ms:
            self.state = STATE_COOLDOWN
            return self._snapshot(now_ms, candidate)

        candidate_ok = (
            candidate.detected
            and candidate.keyword != "unknown"
            and candidate.margin >= self.config.min_margin
            and candidate.confidence >= self.config.min_confidence
        )
        if not candidate_ok:
            self._recent.clear()
            self.state = STATE_REJECTED
            return self._snapshot(now_ms, candidate)

        self._recent.append(candidate)
        votes = Counter(item.keyword for item in self._recent)
        keyword, vote_count = votes.most_common(1)[0]
        required_votes = min(self.config.min_votes, self._recent.maxlen or self.config.min_votes)
        if vote_count < required_votes:
            self.state = STATE_SCORING
            return self._snapshot(now_ms, candidate)

        accepted = max(
            (item for item in self._recent if item.keyword == keyword),
            key=lambda item: item.confidence,
        )
        self._last_detection_end_ms = accepted.end_ms
        self._recent.clear()
        self.state = STATE_DETECTED
        return self._snapshot(now_ms, accepted, detected=True)

    def _snapshot(
        self,
        timestamp_ms: int,
        candidate: DetectionCandidate | None = None,
        detected: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "state": self.state,
            "timestamp": timestamp_ms / 1000.0,
            "detected": detected,
            "keyword": candidate.keyword if detected and candidate else "unknown",
            "confidence": candidate.confidence if candidate else 0.0,
            "distance": candidate.distance if candidate else float("inf"),
            "threshold": candidate.threshold if candidate else 0.0,
            "margin": candidate.margin if candidate else 0.0,
            "second_label": candidate.second_label if candidate else None,
            "start_ms": candidate.start_ms if candidate else None,
            "end_ms": candidate.end_ms if candidate else None,
        }
        return payload
