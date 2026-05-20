from src.streaming.state_machine import (
    DetectionCandidate,
    STATE_COOLDOWN,
    STATE_DETECTED,
    STATE_REJECTED,
    STATE_SCORING,
    StreamingStateConfig,
    StreamingStateMachine,
)


def test_state_machine_requires_smoothing_votes():
    machine = StreamingStateMachine(StreamingStateConfig(smoothing_window=3, min_votes=2))
    candidate = DetectionCandidate(
        detected=True,
        keyword="yes",
        confidence=0.8,
        distance=0.2,
        threshold=0.5,
        margin=0.3,
        start_ms=1000,
        end_ms=1500,
    )

    first = machine.update(candidate, now_ms=1000)
    second = machine.update(candidate, now_ms=1100)

    assert first["state"] == STATE_SCORING
    assert first["detected"] is False
    assert second["state"] == STATE_DETECTED
    assert second["detected"] is True
    assert second["keyword"] == "yes"


def test_state_machine_rejects_low_margin_and_enforces_cooldown():
    machine = StreamingStateMachine(
        StreamingStateConfig(smoothing_window=1, min_votes=1, min_margin=0.2, cooldown_ms=900)
    )
    low_margin = DetectionCandidate(
        detected=True,
        keyword="yes",
        confidence=0.8,
        distance=0.2,
        threshold=0.5,
        margin=0.05,
        start_ms=1000,
        end_ms=1500,
    )
    strong = DetectionCandidate(
        detected=True,
        keyword="yes",
        confidence=0.8,
        distance=0.2,
        threshold=0.5,
        margin=0.3,
        start_ms=2000,
        end_ms=2400,
    )
    duplicate = DetectionCandidate(
        detected=True,
        keyword="yes",
        confidence=0.9,
        distance=0.2,
        threshold=0.5,
        margin=0.4,
        start_ms=2600,
        end_ms=2900,
    )

    assert machine.update(low_margin, now_ms=1000)["state"] == STATE_REJECTED
    assert machine.update(strong, now_ms=2000)["state"] == STATE_DETECTED
    assert machine.update(duplicate, now_ms=2600)["state"] == STATE_COOLDOWN
