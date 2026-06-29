import asyncio
import json
from pathlib import Path

import torch

from src.demo import api_server
from src.streaming.enrollment import EnrollmentProfile


def _json_response_body(response):
    return json.loads(response.body.decode("utf-8"))


def _touch_wav(root: Path, word: str, name: str = "sample.wav") -> None:
    folder = root / word
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"")


def _score(
    *,
    detected: bool,
    keyword: str,
    best_label: str,
    distance: float = 0.1,
    threshold: float = 0.3,
    margin: float = 0.2,
):
    return {
        "detected": detected,
        "keyword": keyword,
        "best_label": best_label,
        "distance": distance,
        "threshold": threshold,
        "margin": margin,
        "confidence": 0.9 if detected else 0.2,
        "second_label": "no",
        "all_distances": {best_label: distance, "no": distance + margin},
        "top_3": [
            {"word": best_label, "dist": distance},
            {"word": "no", "dist": distance + margin},
        ],
    }


def _install_open_set_fixture(monkeypatch, tmp_path, score_by_word):
    gsc = tmp_path / "gsc_v2"
    for word in score_by_word:
        _touch_wav(gsc, word)

    old_profile = api_server.enrollment_profile
    old_prototypes = dict(api_server.prototypes)
    old_thresholds = dict(api_server.proto_thresholds)

    api_server.enrollment_profile = EnrollmentProfile()
    api_server.prototypes.clear()
    api_server.prototypes["yes"] = torch.tensor([1.0])
    api_server.proto_thresholds.clear()
    api_server.proto_thresholds["yes"] = 0.3

    monkeypatch.setattr(api_server, "GSC_DIR", gsc)
    monkeypatch.setattr(api_server, "pad_trim", lambda wav: wav)

    def fake_load_wav_file(path):
        words = sorted(score_by_word)
        value = float(words.index(path.parent.name) + 1)
        return torch.tensor([[value]])

    def fake_embed(wav):
        return wav.flatten().float()

    def fake_score_embedding(
        embedding,
        threshold,
        use_per_class,
        min_margin=0.05,
        candidate_words=None,
    ):
        words = sorted(score_by_word)
        word = words[int(embedding.item()) - 1]
        return score_by_word[word](min_margin)

    monkeypatch.setattr(api_server, "load_wav_file", fake_load_wav_file)
    monkeypatch.setattr(api_server, "embed", fake_embed)
    monkeypatch.setattr(api_server, "score_embedding", fake_score_embedding)

    return old_profile, old_prototypes, old_thresholds


def _restore_demo_state(old_profile, old_prototypes, old_thresholds):
    api_server.enrollment_profile = old_profile
    api_server.prototypes.clear()
    api_server.prototypes.update(old_prototypes)
    api_server.proto_thresholds.clear()
    api_server.proto_thresholds.update(old_thresholds)


def test_open_set_requires_enrollment():
    old_profile = api_server.enrollment_profile
    old_prototypes = dict(api_server.prototypes)
    try:
        api_server.enrollment_profile = EnrollmentProfile()
        api_server.prototypes.clear()

        response = asyncio.run(api_server.open_set_test())

        assert response.status_code == 400
        assert _json_response_body(response)["error"] == "No keywords enrolled"
    finally:
        api_server.enrollment_profile = old_profile
        api_server.prototypes.clear()
        api_server.prototypes.update(old_prototypes)


def test_open_set_computes_known_and_unknown_metrics(monkeypatch, tmp_path):
    old_state = _install_open_set_fixture(
        monkeypatch,
        tmp_path,
        {
            "yes": lambda min_margin: _score(detected=True, keyword="yes", best_label="yes"),
            "cat": lambda min_margin: _score(detected=False, keyword="unknown", best_label="yes"),
        },
    )
    try:
        result = asyncio.run(api_server.open_set_test(
            unknown_words="cat",
            samples_per_word=1,
            threshold=0.3,
            use_per_class=True,
            use_close_word_guard=False,
            seed=7,
        ))

        assert result["settings"]["accept_margin"] == 0.0
        assert result["settings"]["close_word_guard"] is False
        assert result["summary"]["known_tested"] == 1
        assert result["summary"]["unknown_tested"] == 1
        assert result["summary"]["keyword_acc"] == 1.0
        assert result["summary"]["unknown_reject_acc"] == 1.0
        assert result["summary"]["false_accept_rate"] == 0.0
        assert result["false_accepts"] == []
        assert result["known_misses"] == []
    finally:
        _restore_demo_state(*old_state)


def test_open_set_reports_false_accept_and_missing_words(monkeypatch, tmp_path):
    old_state = _install_open_set_fixture(
        monkeypatch,
        tmp_path,
        {
            "yes": lambda min_margin: _score(detected=True, keyword="yes", best_label="yes"),
            "cat": lambda min_margin: _score(detected=True, keyword="yes", best_label="yes"),
        },
    )
    try:
        result = asyncio.run(api_server.open_set_test(
            unknown_words="cat,missing,yes",
            samples_per_word=2,
            threshold=0.3,
            use_per_class=True,
            use_close_word_guard=True,
            seed=7,
        ))

        assert result["summary"]["false_accept_rate"] == 1.0
        assert result["summary"]["false_accepts"] == 1
        assert len(result["false_accepts"]) == 1
        assert result["false_accepts"][0]["word"] == "cat"
        assert result["missing_unknown_words"] == ["missing"]
        assert result["skipped_unknown_words"] == ["yes"]
        assert result["short_unknown_words"][0]["word"] == "cat"
    finally:
        _restore_demo_state(*old_state)


def test_gsc_17_17_preset_shape():
    spec = api_server.OPEN_SET_PRESETS[api_server.GSC_OPEN_SET_PRESET_ID]

    assert len(spec["known_words"]) == 17
    assert len(spec["unknown_words"]) == 17
    assert spec["heldout_words"] == ["visual"]
    assert set(spec["known_words"]).isdisjoint(spec["unknown_words"])
    assert set(spec["known_words"]) | set(spec["unknown_words"]) | set(spec["heldout_words"]) == set(api_server.KNOWN_GSC_WORDS)


def test_score_embedding_filters_candidates():
    old_profile = api_server.enrollment_profile
    old_prototypes = dict(api_server.prototypes)
    try:
        api_server.enrollment_profile = EnrollmentProfile()
        api_server.prototypes.clear()
        api_server.prototypes["yes"] = torch.tensor([0.0])
        api_server.prototypes["cat"] = torch.tensor([10.0])

        result = api_server.score_embedding(
            torch.tensor([10.0]),
            threshold=0.3,
            use_per_class=False,
            min_margin=0.0,
            candidate_words=["yes"],
        )

        assert result["best_label"] == "yes"
        assert "cat" not in result["all_distances"]
        assert result["keyword"] == "unknown"
    finally:
        api_server.enrollment_profile = old_profile
        api_server.prototypes.clear()
        api_server.prototypes.update(old_prototypes)


def test_open_set_balanced_score(monkeypatch, tmp_path):
    old_state = _install_open_set_fixture(
        monkeypatch,
        tmp_path,
        {
            "yes": lambda min_margin: _score(detected=True, keyword="yes", best_label="yes"),
            "cat": lambda min_margin: _score(detected=False, keyword="unknown", best_label="yes"),
        },
    )
    try:
        result = asyncio.run(api_server.open_set_test(
            known_words="yes",
            unknown_words="cat",
            preset="manual",
            samples_per_word=1,
            threshold=0.3,
            use_per_class=True,
            use_close_word_guard=False,
            seed=7,
        ))

        assert result["known_words"] == ["yes"]
        assert result["unknown_words"] == ["cat"]
        assert result["candidate_words"] == ["yes"]
        assert result["summary"]["balanced_score"] == 1.0
    finally:
        _restore_demo_state(*old_state)


def test_open_set_calibration_selects_best_balanced(monkeypatch, tmp_path):
    def known_score(_min_margin):
        return _score(detected=True, keyword="yes", best_label="yes")

    def unknown_score_factory():
        def score(_min_margin):
            # The fake scorer below overrides this function, but keep fixture shape explicit.
            return _score(detected=False, keyword="unknown", best_label="yes")
        return score

    old_state = _install_open_set_fixture(
        monkeypatch,
        tmp_path,
        {
            "yes": known_score,
            "cat": unknown_score_factory(),
        },
    )

    def fake_score_embedding(embedding, threshold, use_per_class, min_margin=0.05, candidate_words=None):
        word = "cat" if int(embedding.item()) == 1 else "yes"
        if word == "yes":
            return _score(detected=threshold >= 0.3, keyword="yes" if threshold >= 0.3 else "unknown", best_label="yes")
        if threshold >= 0.5:
            return _score(detected=True, keyword="yes", best_label="yes")
        return _score(detected=False, keyword="unknown", best_label="yes")

    monkeypatch.setattr(api_server, "score_embedding", fake_score_embedding)
    try:
        result = asyncio.run(api_server.open_set_calibrate(
            known_words="yes",
            unknown_words="cat",
            preset="manual",
            samples_per_word=1,
            seed=7,
            threshold_min=0.1,
            threshold_max=0.5,
            threshold_step=0.2,
            accept_margin_values="0,0.05",
            use_per_class_options="false",
        ))

        assert result["best_balanced"]["threshold"] == 0.3
        assert result["best_balanced"]["balanced_score"] == 1.0
        assert result["best_balanced"]["keyword_acc"] == 1.0
        assert result["best_balanced"]["unknown_reject_acc"] == 1.0
    finally:
        _restore_demo_state(*old_state)
