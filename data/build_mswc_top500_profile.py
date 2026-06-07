"""Build a clean MSWC Top500-full profile from extracted English MSWC clips.

This preserves the existing full-English ``data/mswc_en/splits`` files by
writing a separate profile directory, usually ``data/mswc_top500_full``:

    data/mswc_top500_full/
      clips -> ../mswc_en/clips
      splits/train_words.json
      splits/val_words.json
      splits/eval_words.json
      splits/train_files.json
      splits/val_files.json

The word split matches the legacy Top500 logic in ``data/download_mswc.py``:
sort words by metadata count, take the top 500, use the first 450 for train
and the next 50 for validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_mswc_file_splits import build_file_splits
from data.download_mswc import create_splits

logger = logging.getLogger(__name__)


def _load_word_counts(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return {str(word): int(count) for word, count in payload.items()}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_clips_view(source_data_dir: Path, output_data_dir: Path, overwrite: bool) -> None:
    """Expose source clips under output profile without copying audio."""
    source_clips = source_data_dir / "clips"
    output_clips = output_data_dir / "clips"

    if not source_clips.is_dir():
        raise FileNotFoundError(f"Missing extracted clips directory: {source_clips}")

    if source_data_dir.resolve() == output_data_dir.resolve():
        return

    if output_clips.exists() or output_clips.is_symlink():
        if output_clips.is_symlink() and output_clips.resolve() == source_clips.resolve():
            return
        if not overwrite:
            raise FileExistsError(
                f"{output_clips} already exists and is not the expected symlink. "
                "Pass --overwrite to replace it."
            )
        if output_clips.is_dir() and not output_clips.is_symlink():
            raise IsADirectoryError(
                f"Refusing to remove real directory {output_clips}; "
                "choose a fresh output directory."
            )
        output_clips.unlink()

    try:
        output_clips.symlink_to(source_clips.resolve(), target_is_directory=True)
    except OSError as exc:
        if os.name == "nt":
            raise OSError(
                "Could not create a directory symlink on Windows. Use the same "
                "source/output dir for local tests, or run on Linux ict6."
            ) from exc
        raise


def build_top500_profile(
    source_data_dir: Path,
    output_data_dir: Path,
    max_per_word: int = 0,
    n_train: int = 450,
    n_val: int = 50,
    include_eval_words: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    source_data_dir = source_data_dir.resolve()
    output_data_dir.mkdir(parents=True, exist_ok=True)

    counts_path = source_data_dir / "metadata" / "en_word_counts.json"
    word_counts = _load_word_counts(counts_path)
    train_words, val_words, eval_words = create_splits(
        word_counts,
        n_train=n_train,
        n_val=n_val,
    )
    if not include_eval_words:
        eval_words = []

    splits_dir = output_data_dir / "splits"
    _write_json(splits_dir / "train_words.json", train_words)
    _write_json(splits_dir / "val_words.json", val_words)
    _write_json(splits_dir / "eval_words.json", eval_words)
    _ensure_clips_view(source_data_dir, output_data_dir, overwrite=overwrite)

    manifest_summary = build_file_splits(
        data_dir=output_data_dir,
        archive_path=source_data_dir / "en.tar.gz",
        max_per_word=max_per_word,
        overwrite=overwrite,
        output_suffix="",
        source="clips",
    )

    summary = {
        "source_data_dir": str(source_data_dir),
        "output_data_dir": str(output_data_dir),
        "split_mode": "top500",
        "n_train": len(train_words),
        "n_val": len(val_words),
        "n_eval": len(eval_words),
        "max_per_word": int(max_per_word),
        "train_files": manifest_summary["train_files"],
        "val_files": manifest_summary["val_files"],
        "missing_train_words": manifest_summary["missing_train_words"],
        "missing_val_words": manifest_summary["missing_val_words"],
        "top_train_words_first10": train_words[:10],
        "top_val_words_first10": val_words[:10],
    }
    _write_json(splits_dir / "top500_profile_summary.json", summary)
    logger.info("Top500 profile summary: %s", splits_dir / "top500_profile_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separate MSWC Top500-full profile")
    parser.add_argument("--source-data-dir", type=Path, default=Path("data/mswc_en"))
    parser.add_argument("--output-data-dir", type=Path, default=Path("data/mswc_top500_full"))
    parser.add_argument("--max-per-word", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=450)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--include-eval-words", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = build_top500_profile(
        source_data_dir=args.source_data_dir,
        output_data_dir=args.output_data_dir,
        max_per_word=args.max_per_word,
        n_train=args.n_train,
        n_val=args.n_val,
        include_eval_words=args.include_eval_words,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
