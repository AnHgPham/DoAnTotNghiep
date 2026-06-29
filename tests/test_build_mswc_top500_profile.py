import json
from pathlib import Path

from data.build_mswc_top500_profile import build_top500_profile


def _write_clip(root: Path, word: str, name: str) -> None:
    path = root / "clips" / word / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")


def test_build_top500_profile_uses_ranked_train_val_split(tmp_path):
    data_dir = tmp_path / "mswc_en"
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    counts = {
        "was": 100,
        "and": 90,
        "you": 80,
        "that": 70,
        "are": 60,
        "tail": 1,
    }
    (metadata_dir / "en_word_counts.json").write_text(json.dumps(counts), encoding="utf-8")

    _write_clip(data_dir, "was", "a.wav")
    _write_clip(data_dir, "was", "b.opus")
    _write_clip(data_dir, "and", "a.wav")
    _write_clip(data_dir, "you", "a.wav")
    _write_clip(data_dir, "that", "a.wav")
    _write_clip(data_dir, "are", "a.wav")

    summary = build_top500_profile(
        source_data_dir=data_dir,
        output_data_dir=data_dir,
        max_per_word=1,
        n_train=3,
        n_val=2,
        overwrite=True,
    )

    splits_dir = data_dir / "splits"
    train_words = json.loads((splits_dir / "train_words.json").read_text(encoding="utf-8"))
    val_words = json.loads((splits_dir / "val_words.json").read_text(encoding="utf-8"))
    train_files = json.loads((splits_dir / "train_files.json").read_text(encoding="utf-8"))
    val_files = json.loads((splits_dir / "val_files.json").read_text(encoding="utf-8"))

    assert train_words == ["was", "and", "you"]
    assert val_words == ["that", "are"]
    assert len(train_files) == 3
    assert len(val_files) == 2
    assert summary["train_files"] == 3
    assert summary["val_files"] == 2
