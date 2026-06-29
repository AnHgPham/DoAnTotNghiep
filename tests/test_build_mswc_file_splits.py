"""Tests for capped MSWC manifest generation."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from data.build_mswc_file_splits import build_file_splits


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_archive(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for word, count in {"yes": 3, "no": 1, "up": 2, "skip": 1}.items():
            for idx in range(count):
                payload = b"RIFF"
                info = tarfile.TarInfo(f"en/clips/{word}/{word}_{idx}.wav")
                info.size = len(payload)
                tar.addfile(info, fileobj=io.BytesIO(payload))


def test_build_file_splits_writes_named_manifest_without_touching_default(tmp_path):
    data_dir = tmp_path / "mswc_en"
    archive = data_dir / "en.tar.gz"
    _write_json(data_dir / "splits" / "train_words.json", ["yes", "no"])
    _write_json(data_dir / "splits" / "val_words.json", ["up"])
    data_dir.mkdir(parents=True, exist_ok=True)
    _make_archive(archive)

    summary = build_file_splits(
        data_dir=data_dir,
        archive_path=archive,
        max_per_word=2,
        output_suffix="max2",
    )

    train_files = json.loads((data_dir / "splits" / "train_files_max2.json").read_text())
    val_files = json.loads((data_dir / "splits" / "val_files_max2.json").read_text())

    assert not (data_dir / "splits" / "train_files.json").exists()
    assert len(train_files) == 3
    assert train_files == [
        "clips/yes/yes_0.wav",
        "clips/yes/yes_1.wav",
        "clips/no/no_0.wav",
    ]
    assert val_files == ["clips/up/up_0.wav", "clips/up/up_1.wav"]
    assert summary["train_files"] == 3
    assert summary["val_files"] == 2

    with pytest.raises(FileExistsError):
        build_file_splits(
            data_dir=data_dir,
            archive_path=archive,
            max_per_word=2,
            output_suffix="max2",
        )


def test_build_file_splits_can_read_extracted_clips(tmp_path):
    data_dir = tmp_path / "mswc_en"
    _write_json(data_dir / "splits" / "train_words.json", ["yes", "no"])
    _write_json(data_dir / "splits" / "val_words.json", ["up"])

    for word, count in {"yes": 3, "no": 1, "up": 2}.items():
        word_dir = data_dir / "clips" / word
        word_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            (word_dir / f"{word}_{idx}.wav").write_bytes(b"RIFF")

    summary = build_file_splits(
        data_dir=data_dir,
        archive_path=data_dir / "unused.tar.gz",
        max_per_word=2,
        output_suffix="max2",
        source="clips",
    )

    train_files = json.loads((data_dir / "splits" / "train_files_max2.json").read_text())
    val_files = json.loads((data_dir / "splits" / "val_files_max2.json").read_text())

    assert summary["source"] == "clips"
    assert len(train_files) == 3
    assert len([item for item in train_files if item.startswith("clips/yes/")]) == 2
    assert len([item for item in train_files if item.startswith("clips/no/")]) == 1
    assert all(item.endswith(".wav") for item in train_files)
    assert len(val_files) == 2
    assert all(item.startswith("clips/up/") and item.endswith(".wav") for item in val_files)


def test_build_file_splits_accepts_extracted_opus_clips(tmp_path):
    data_dir = tmp_path / "mswc_en"
    _write_json(data_dir / "splits" / "train_words.json", ["yes"])
    _write_json(data_dir / "splits" / "val_words.json", ["up"])

    for word in ("yes", "up"):
        word_dir = data_dir / "clips" / word
        word_dir.mkdir(parents=True, exist_ok=True)
        (word_dir / f"{word}_0.opus").write_bytes(b"OPUS")
        (word_dir / f"{word}_1.txt").write_text("ignore", encoding="utf-8")

    summary = build_file_splits(
        data_dir=data_dir,
        archive_path=data_dir / "unused.tar.gz",
        max_per_word=2,
        output_suffix="opus",
        source="clips",
    )

    train_files = json.loads((data_dir / "splits" / "train_files_opus.json").read_text())
    val_files = json.loads((data_dir / "splits" / "val_files_opus.json").read_text())

    assert summary["train_files"] == 1
    assert summary["val_files"] == 1
    assert train_files == ["clips/yes/yes_0.opus"]
    assert val_files == ["clips/up/up_0.opus"]
