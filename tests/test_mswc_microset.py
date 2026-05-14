"""Tests for the official MSWC Microset downloader helpers."""

from __future__ import annotations

import json
import io
import tarfile
from pathlib import Path

from data.download_mswc_microset import extract_language, write_word_splits


def _make_fake_microset_tar(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for word in ["yes", "no", "up"]:
            for idx in range(2):
                name = f"mswc_microset/en/clips/{word}/{word}_{idx}.opus"
                payload = b"OggS"
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, fileobj=io.BytesIO(payload))


def test_extract_language_and_write_splits(tmp_path):
    archive = tmp_path / "microset.tar.gz"
    out_dir = tmp_path / "mswc_microset_en"
    _make_fake_microset_tar(archive)

    extract_language(archive, "en", out_dir)
    write_word_splits(out_dir, val_fraction=0.34, seed=1)

    assert len(list((out_dir / "clips").rglob("*.opus"))) == 6
    train = json.loads((out_dir / "splits" / "train_words.json").read_text())
    val = json.loads((out_dir / "splits" / "val_words.json").read_text())
    assert sorted(train + val) == ["no", "up", "yes"]
    assert len(val) == 1
