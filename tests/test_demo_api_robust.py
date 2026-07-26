import asyncio
import json
import threading

import torch
import torch.nn.functional as F

from src.demo import api_server
from src.streaming.enrollment import EnrollmentProfile, KeywordProfile


def _profile(label: str, vector: torch.Tensor, threshold: float = 0.35) -> KeywordProfile:
    proto = F.normalize(vector.float(), p=2, dim=0)
    return KeywordProfile(
        label=label,
        prototype=proto,
        exemplars=proto.view(1, -1),
        threshold=threshold,
        support_distance_mean=0.0,
        support_distance_std=0.0,
        qualities=[],
    )


def test_score_embedding_uses_threshold_and_margin_for_robust_profile():
    old_profile = api_server.enrollment_profile
    old_prototypes = dict(api_server.prototypes)
    old_thresholds = dict(api_server.proto_thresholds)
    try:
        api_server.enrollment_profile = EnrollmentProfile({
            "yes": _profile("yes", torch.tensor([1.0, 0.0]), threshold=0.40),
            "no": _profile("no", torch.tensor([0.0, 1.0]), threshold=0.40),
        })
        api_server.prototypes.clear()
        api_server.proto_thresholds.clear()

        confident = api_server.score_embedding(
            F.normalize(torch.tensor([1.0, 0.02]), p=2, dim=0),
            threshold=0.10,
            use_per_class=True,
            min_margin=0.05,
        )
        assert confident["detected"] is True
        assert confident["keyword"] == "yes"
        assert confident["threshold"] == 0.40
        assert confident["margin"] > 0.05

        ambiguous = api_server.score_embedding(
            F.normalize(torch.tensor([1.0, 1.0]), p=2, dim=0),
            threshold=2.0,
            use_per_class=True,
            min_margin=0.05,
        )
        assert ambiguous["detected"] is False
        assert ambiguous["keyword"] == "unknown"
    finally:
        api_server.enrollment_profile = old_profile
        api_server.prototypes.clear()
        api_server.prototypes.update(old_prototypes)
        api_server.proto_thresholds.clear()
        api_server.proto_thresholds.update(old_thresholds)


def test_select_diverse_files_spreads_choices():
    files = [f"f{i}.wav" for i in range(10)]

    selected = api_server.select_diverse_files(files, 5)

    assert selected == ["f0.wav", "f2.wav", "f4.wav", "f6.wav", "f8.wav"]


def test_enroll_gsc_offloads_sync_work(monkeypatch):
    caller_thread = threading.get_ident()
    worker_threads = []

    def fake_enroll(word_list, k):
        worker_threads.append(threading.get_ident())
        return {"results": [], "enrolled": 0, "timing_ms": 1.0}

    monkeypatch.setattr(api_server, "_enroll_gsc_sync", fake_enroll)

    result = asyncio.run(api_server.enroll_gsc(words="yes,yes,no", k=3))

    assert result["timing_ms"] == 1.0
    assert worker_threads and worker_threads[0] != caller_thread


def test_enroll_gsc_rejects_invalid_sample_count_before_worker(monkeypatch):
    called = False

    def fake_enroll(word_list, k):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(api_server, "_enroll_gsc_sync", fake_enroll)

    response = asyncio.run(api_server.enroll_gsc(words="yes", k=0))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 422
    assert "between 1 and" in payload["error"]
    assert called is False
