"""Tests for split-aware MSWC Drive cache handling."""

from __future__ import annotations

import json
from pathlib import Path

import data.mswc_drive_cache as cache


def _write_split_cache(root: Path, words: list[str], with_wavs: bool = True) -> Path:
    drive_project = root / "drive"
    drive_cache = drive_project / cache.cache_dir_name("top500", 200)
    splits = drive_cache / "splits"
    clips = drive_cache / "clips"
    splits.mkdir(parents=True)
    clips.mkdir(parents=True)

    (splits / "train_words.json").write_text(json.dumps(words[:2]), encoding="utf-8")
    (splits / "val_words.json").write_text(json.dumps(words[2:]), encoding="utf-8")
    (splits / "eval_words.json").write_text("[]", encoding="utf-8")

    if with_wavs:
        for word in words:
            word_dir = clips / word
            word_dir.mkdir()
            (word_dir / f"{word}_0.wav").write_bytes(b"RIFF")
    return drive_project


def test_drive_cache_validates_train_val_coverage(tmp_path):
    drive_project = _write_split_cache(tmp_path, ["yes", "no", "up"])

    valid, status = cache.is_drive_cache_valid(
        drive_project,
        split_mode="top500",
        max_per_word=200,
        min_train_val_coverage=0.9,
    )

    assert valid is True
    assert status["required_present"] == 3
    assert status["required_total"] == 3


def test_drive_cache_rejects_missing_wavs(tmp_path):
    drive_project = _write_split_cache(tmp_path, ["yes", "no", "up"], with_wavs=False)

    valid, status = cache.is_drive_cache_valid(drive_project)

    assert valid is False
    assert status["n_wav"] == 0


def test_setup_loads_valid_cache(tmp_path, monkeypatch):
    drive_project = _write_split_cache(tmp_path, ["yes", "no", "up"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cache, "LOCAL_MSWC", Path("data/mswc_en"))
    monkeypatch.setattr(cache, "LOCAL_CLIPS", Path("data/mswc_en/clips"))

    from_cache = cache.setup_mswc_from_drive(drive_project, max_per_word=200)

    assert from_cache is True
    assert Path("data/mswc_en/splits/train_words.json").exists()
    assert Path("data/mswc_en/clips").exists()


def test_setup_miss_runs_download_and_save(tmp_path, monkeypatch):
    drive_project = tmp_path / "drive"
    calls: list[str] = []

    def fake_download_and_convert(**kwargs):
        calls.append(f"download:{kwargs['split_mode']}:{kwargs['max_per_word']}")
        clips = cache.LOCAL_CLIPS
        clips.mkdir(parents=True)
        word_dir = clips / "yes"
        word_dir.mkdir()
        (word_dir / "yes_0.wav").write_bytes(b"RIFF")
        splits = cache.LOCAL_MSWC / "splits"
        splits.mkdir(parents=True)
        (splits / "train_words.json").write_text('["yes"]', encoding="utf-8")
        (splits / "val_words.json").write_text('["no"]', encoding="utf-8")
        (splits / "eval_words.json").write_text("[]", encoding="utf-8")

    def fake_save_to_drive(*args, **kwargs):
        calls.append(f"save:{kwargs['split_mode']}:{kwargs['max_per_word']}")
        return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cache, "LOCAL_MSWC", Path("data/mswc_en"))
    monkeypatch.setattr(cache, "LOCAL_CLIPS", Path("data/mswc_en/clips"))
    monkeypatch.setattr(cache, "download_and_convert", fake_download_and_convert)
    monkeypatch.setattr(cache, "save_to_drive", fake_save_to_drive)

    from_cache = cache.setup_mswc_from_drive(drive_project)

    assert from_cache is False
    assert calls == ["download:top500:0", "save:top500:0"]


def test_setup_repairs_partial_cache_with_wavs_but_no_splits(tmp_path, monkeypatch):
    drive_project = tmp_path / "drive"
    drive_cache = drive_project / cache.cache_dir_name("top500", 200)
    clips = drive_cache / "clips"
    for i in range(35):
        word_dir = clips / f"word{i:02d}"
        word_dir.mkdir(parents=True)
        (word_dir / "0.wav").write_bytes(b"RIFF")

    def fail_download(**kwargs):
        raise AssertionError("download should not run when partial cache is repairable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cache, "LOCAL_MSWC", Path("data/mswc_en"))
    monkeypatch.setattr(cache, "LOCAL_CLIPS", Path("data/mswc_en/clips"))
    monkeypatch.setattr(cache, "download_and_convert", fail_download)

    from_cache = cache.setup_mswc_from_drive(drive_project, max_per_word=200)

    assert from_cache is True
    assert (drive_cache / "splits" / "train_words.json").exists()
    assert Path("data/mswc_en/splits/train_words.json").exists()


def test_setup_repairs_cache_with_splits_that_do_not_match_wavs(tmp_path, monkeypatch):
    drive_project = tmp_path / "drive"
    drive_cache = drive_project / cache.cache_dir_name("top500", 200)
    clips = drive_cache / "clips"
    splits = drive_cache / "splits"
    splits.mkdir(parents=True)
    (splits / "train_words.json").write_text(
        json.dumps([f"missing{i:02d}" for i in range(30)]),
        encoding="utf-8",
    )
    (splits / "val_words.json").write_text(
        json.dumps([f"missing_val{i:02d}" for i in range(5)]),
        encoding="utf-8",
    )
    (splits / "eval_words.json").write_text("[]", encoding="utf-8")
    for i in range(40):
        word_dir = clips / f"cached{i:02d}"
        word_dir.mkdir(parents=True)
        (word_dir / "0.wav").write_bytes(b"RIFF")

    def fail_download(**kwargs):
        raise AssertionError("download should not run when mismatched splits are repairable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cache, "LOCAL_MSWC", Path("data/mswc_en"))
    monkeypatch.setattr(cache, "LOCAL_CLIPS", Path("data/mswc_en/clips"))
    monkeypatch.setattr(cache, "download_and_convert", fail_download)

    from_cache = cache.setup_mswc_from_drive(drive_project, max_per_word=200)

    assert from_cache is True
    repaired = json.loads((drive_cache / "splits" / "train_words.json").read_text())
    assert repaired[0].startswith("cached")
