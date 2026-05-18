"""Tests for the official MSWC Microset downloader helpers."""

from __future__ import annotations

import json
import io
import tarfile
from pathlib import Path

from data.download_mswc_microset import extract_language, write_word_splits
from src.data.mswc_dataset import MSWCDataset


def _make_fake_microset_tar(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        train_csv = "path\n" + "\n".join(
            f"clips/{word}/{word}_0.opus" for word in ["yes", "no", "up"]
        )
        for name, payload in {
            "mswc_microset/en/en_train.csv": train_csv.encode("utf-8"),
            "mswc_microset/en/en_dev.csv": b"path\nclips/yes/yes_1.opus\nclips/no/no_1.opus\n",
            "mswc_microset/en/en_test.csv": b"path\nclips/up/up_1.opus\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, fileobj=io.BytesIO(payload))

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
    write_word_splits(out_dir, val_fraction=0.34, seed=1, split_source="random")

    assert len(list((out_dir / "clips").rglob("*.opus"))) == 6
    train = json.loads((out_dir / "splits" / "train_words.json").read_text())
    val = json.loads((out_dir / "splits" / "val_words.json").read_text())
    assert sorted(train + val) == ["no", "up", "yes"]
    assert len(val) == 1


def test_official_csv_splits_use_sample_level_train_dev_test_manifests(tmp_path):
    archive = tmp_path / "microset.tar.gz"
    out_dir = tmp_path / "mswc_microset_en"
    _make_fake_microset_tar(archive)

    extract_language(archive, "en", out_dir)
    write_word_splits(out_dir, val_fraction=0.1, seed=1, language="en", split_source="official")

    train = json.loads((out_dir / "splits" / "train_words.json").read_text())
    val = json.loads((out_dir / "splits" / "val_words.json").read_text())
    eval_words = json.loads((out_dir / "splits" / "eval_words.json").read_text())
    train_files = json.loads((out_dir / "splits" / "train_files.json").read_text())
    val_files = json.loads((out_dir / "splits" / "val_files.json").read_text())
    eval_files = json.loads((out_dir / "splits" / "eval_files.json").read_text())

    assert train == ["no", "up", "yes"]
    assert val == ["no", "yes"]
    assert eval_words == ["up"]
    assert train_files == [
        "clips/no/no_0.wav",
        "clips/up/up_0.wav",
        "clips/yes/yes_0.wav",
    ]
    assert val_files == ["clips/no/no_1.wav", "clips/yes/yes_1.wav"]
    assert eval_files == ["clips/up/up_1.wav"]


def test_all_words_train_splits(tmp_path):
    archive = tmp_path / "microset.tar.gz"
    out_dir = tmp_path / "mswc_microset_en"
    _make_fake_microset_tar(archive)

    extract_language(archive, "en", out_dir)
    write_word_splits(out_dir, val_fraction=0.1, seed=1, all_words_train=True)

    train = json.loads((out_dir / "splits" / "train_words.json").read_text())
    val = json.loads((out_dir / "splits" / "val_words.json").read_text())

    assert train == ["no", "up", "yes"]
    assert val == []


def test_dataset_uses_explicit_csv_manifest_instead_of_scanning_folder(tmp_path):
    root = tmp_path / "mswc_microset_en"
    for rel in [
        "clips/yes/yes_0.wav",
        "clips/yes/yes_extra.wav",
        "clips/no/no_0.wav",
        "clips/up/up_0.wav",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    dataset = MSWCDataset(
        root_dir=root,
        words=["no", "up", "yes"],
        file_paths=["clips/yes/yes_0.wav", "clips/no/no_0.wav"],
    )

    assert len(dataset.samples) == 2
    assert sorted(path.name for path, _ in dataset.samples) == ["no_0.wav", "yes_0.wav"]
    assert set(dataset.word_to_idx) == {"no", "up", "yes"}
