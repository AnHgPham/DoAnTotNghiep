"""Tests for robust enrollment and streaming decision logic."""

import math

import torch
import torch.nn.functional as F

from src.streaming.enrollment import (
    build_enrollment_profile,
    crop_to_active_region,
    pad_or_trim,
)
from src.streaming.robust_engine import (
    RobustStreamingKWS,
    StreamingDecisionConfig,
    energy_segments,
)


SR = 16000


class TinyBackend:
    """Deterministic embedding backend for unit tests."""

    sample_rate = SR

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = pad_or_trim(waveform, SR).squeeze(0)
        if wav.numel() == 0:
            return F.normalize(torch.ones(8), p=2, dim=0)
        first = wav[: SR // 2]
        second = wav[SR // 2 :]
        zcr = ((wav[:-1] * wav[1:]) < 0).float().mean()
        vec = torch.tensor(
            [
                torch.sqrt(torch.mean(wav.pow(2)) + 1e-8).item(),
                torch.sqrt(torch.mean(first.pow(2)) + 1e-8).item(),
                torch.sqrt(torch.mean(second.pow(2)) + 1e-8).item(),
                wav.abs().max().item(),
                zcr.item(),
                wav.mean().item(),
                wav.abs().mean().item(),
                1.0,
            ],
            dtype=torch.float32,
        )
        return F.normalize(vec, p=2, dim=0)

    def embed_many(self, waveforms):
        return torch.stack([self.embed(w) for w in waveforms])


def _tone(freq: float = 440.0, duration: float = 0.65, amp: float = 0.25) -> torch.Tensor:
    t = torch.arange(int(SR * duration), dtype=torch.float32) / SR
    return (amp * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)


def test_crop_to_active_region_finds_tone():
    wav = torch.cat([torch.zeros(1, 4000), _tone(), torch.zeros(1, 5000)], dim=-1)

    cropped, (start, end), quality = crop_to_active_region(wav)

    assert cropped.shape[-1] < wav.shape[-1]
    assert start < end
    assert quality.active_ms > 100


def test_build_enrollment_profile_scores_same_keyword():
    backend = TinyBackend()
    samples = {"tone": [_tone(440, 0.6), _tone(445, 0.62), _tone(435, 0.58)]}

    profile = build_enrollment_profile(samples, backend)
    result = profile.score(backend.embed(_tone(440, 0.6)), min_margin=0.0)

    assert "tone" in profile.keywords
    assert result.detected is True
    assert result.label == "tone"
    assert result.distance <= result.threshold


def test_build_enrollment_profile_batches_views_across_keywords():
    class CountingBackend(TinyBackend):
        def __init__(self):
            self.batch_sizes = []

        def embed_many(self, waveforms):
            self.batch_sizes.append(len(waveforms))
            return super().embed_many(waveforms)

    backend = CountingBackend()
    profile = build_enrollment_profile(
        {
            "low": [_tone(330, 0.6), _tone(335, 0.6)],
            "high": [_tone(660, 0.6), _tone(665, 0.6)],
        },
        backend,
        views_per_sample=3,
    )

    assert set(profile.keywords) == {"low", "high"}
    assert backend.batch_sizes == [12]


def test_energy_segments_and_robust_engine_detect_event():
    backend = TinyBackend()
    profile = build_enrollment_profile(
        {"tone": [_tone(440, 0.6), _tone(445, 0.62), _tone(435, 0.58)]},
        backend,
    )
    wav = torch.cat(
        [torch.zeros(1, SR), _tone(440, 0.65), torch.zeros(1, SR)],
        dim=-1,
    )

    cfg = StreamingDecisionConfig(min_votes=1, min_margin=0.0, cooldown_ms=500)
    assert energy_segments(wav, cfg)

    engine = RobustStreamingKWS(backend, profile, config=cfg)
    events = engine.process_file(wav)

    assert len(events) == 1
    assert events[0]["keyword"] == "tone"
    assert events[0]["confidence"] > 0
    assert events[0]["top_3"][0]["word"] == "tone"


def test_process_file_batches_candidate_windows_across_segments():
    class CountingBackend(TinyBackend):
        def __init__(self):
            self.embed_many_calls = 0

        def embed_many(self, waveforms):
            self.embed_many_calls += 1
            return super().embed_many(waveforms)

    backend = CountingBackend()
    profile = build_enrollment_profile(
        {"tone": [_tone(440, 0.6), _tone(445, 0.62), _tone(435, 0.58)]},
        backend,
    )
    backend.embed_many_calls = 0
    wav = torch.cat(
        [
            torch.zeros(1, SR // 2),
            _tone(440, 0.6),
            torch.zeros(1, SR),
            _tone(440, 0.6),
            torch.zeros(1, SR // 2),
        ],
        dim=-1,
    )
    engine = RobustStreamingKWS(
        backend,
        profile,
        config=StreamingDecisionConfig(cooldown_ms=100),
    )

    engine.process_file(wav)

    assert len(energy_segments(wav, engine.config)) == 2
    assert backend.embed_many_calls == 1
