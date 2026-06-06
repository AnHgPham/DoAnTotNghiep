"""Build capped MSWC file manifests from MSWC English data.

The full-English MSWC layout can contain tens of thousands of word folders.
Scanning those folders repeatedly during training is too slow for an experiment
matrix. This script writes explicit file lists once:

    data/mswc_en/splits/train_files.json
    data/mswc_en/splits/val_files.json

The paths are relative to ``data/mswc_en`` so ``scripts/train.py`` can consume
them directly through ``MSWCDataset(file_paths=...)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)
AUDIO_SUFFIXES = {".wav", ".opus", ".flac", ".mp3"}


def _load_words(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return [str(item) for item in payload]


def _extract_word(member_name: str) -> str | None:
    parts = Path(member_name).parts
    for i, part in enumerate(parts):
        if part == "clips" and i + 2 <= len(parts) - 1:
            return parts[i + 1]
    if len(parts) >= 2:
        return parts[-2]
    return None


def _relative_clip_path(member_name: str) -> str:
    parts = Path(member_name).parts
    for i, part in enumerate(parts):
        if part == "clips" and i + 2 <= len(parts) - 1:
            return Path(*parts[i:]).as_posix()
    return (Path("clips") / Path(member_name).parent.name / Path(member_name).name).as_posix()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_clip_path(
    word: str,
    rel_path: str,
    train_set: set[str],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    train_files: list[str],
    val_files: list[str],
    max_per_word: int,
    unlimited: bool,
) -> None:
    if word in train_set:
        if unlimited or train_counts[word] < max_per_word:
            train_files.append(rel_path)
            train_counts[word] += 1
    elif word in val_counts:
        if unlimited or val_counts[word] < max_per_word:
            val_files.append(rel_path)
            val_counts[word] += 1


def _build_from_archive(
    archive_path: Path,
    train_set: set[str],
    val_set: set[str],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    train_files: list[str],
    val_files: list[str],
    max_per_word: int,
    unlimited: bool,
) -> None:
    target_words = train_set | val_set

    logger.info("Reading archive: %s", archive_path)
    with tarfile.open(str(archive_path), "r:gz") as tar:
        for member in tqdm(tar, desc="Building MSWC file manifests from archive"):
            if not member.isfile():
                continue
            word = _extract_word(member.name)
            if word not in target_words:
                continue
            _append_clip_path(
                word=word,
                rel_path=_relative_clip_path(member.name),
                train_set=train_set,
                train_counts=train_counts,
                val_counts=val_counts,
                train_files=train_files,
                val_files=val_files,
                max_per_word=max_per_word,
                unlimited=unlimited,
            )


def _build_from_clips(
    data_dir: Path,
    train_words: list[str],
    val_words: list[str],
    train_set: set[str],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    train_files: list[str],
    val_files: list[str],
    max_per_word: int,
    unlimited: bool,
) -> None:
    clips_dir = data_dir / "clips"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing extracted clips directory: {clips_dir}")

    logger.info("Reading extracted clips: %s", clips_dir)
    for word in tqdm(train_words + val_words, desc="Building MSWC file manifests from clips"):
        word_dir = clips_dir / word
        if not word_dir.is_dir():
            continue
        files: list[str] = []
        with os.scandir(word_dir) as entries:
            for entry in entries:
                suffix = Path(entry.name).suffix.lower()
                if suffix not in AUDIO_SUFFIXES:
                    continue
                try:
                    is_file = entry.is_file(follow_symlinks=True)
                except OSError:
                    continue
                if not is_file:
                    continue
                files.append(entry.name)
                if not unlimited and len(files) >= max_per_word:
                    break
        for name in sorted(files):
            rel_path = (Path("clips") / word / name).as_posix()
            _append_clip_path(
                word=word,
                rel_path=rel_path,
                train_set=train_set,
                train_counts=train_counts,
                val_counts=val_counts,
                train_files=train_files,
                val_files=val_files,
                max_per_word=max_per_word,
                unlimited=unlimited,
            )
            if not unlimited and (
                train_counts.get(word, 0) >= max_per_word
                or val_counts.get(word, 0) >= max_per_word
            ):
                continue


def build_file_splits(
    data_dir: Path,
    archive_path: Path,
    max_per_word: int,
    overwrite: bool = False,
    output_suffix: str = "",
    source: str = "auto",
) -> dict[str, object]:
    splits_dir = data_dir / "splits"
    train_words = _load_words(splits_dir / "train_words.json")
    val_words = _load_words(splits_dir / "val_words.json")

    suffix = output_suffix.strip().lstrip("_")
    suffix_part = f"_{suffix}" if suffix else ""
    train_out = splits_dir / f"train_files{suffix_part}.json"
    val_out = splits_dir / f"val_files{suffix_part}.json"
    summary_out = splits_dir / f"file_manifest_summary{suffix_part}.json"
    existing_outputs = [path for path in (train_out, val_out, summary_out) if path.exists()]
    if not overwrite and existing_outputs:
        raise FileExistsError(
            f"{', '.join(str(path) for path in existing_outputs)} already exist. "
            "Pass --overwrite to replace them."
        )

    train_set = set(train_words)
    val_set = set(val_words)
    target_words = train_set | val_set

    train_counts = {word: 0 for word in train_words}
    val_counts = {word: 0 for word in val_words}
    train_files: list[str] = []
    val_files: list[str] = []
    unlimited = max_per_word <= 0
    if source == "auto":
        source = "clips" if (data_dir / "clips").is_dir() else "archive"
    if source not in {"archive", "clips"}:
        raise ValueError(f"Unsupported source: {source}")

    logger.info(
        "Targets: %d train words, %d val words, max_per_word=%s, source=%s",
        len(train_words),
        len(val_words),
        "unlimited" if unlimited else str(max_per_word),
        source,
    )

    if source == "archive":
        _build_from_archive(
            archive_path=archive_path,
            train_set=train_set,
            val_set=val_set,
            train_counts=train_counts,
            val_counts=val_counts,
            train_files=train_files,
            val_files=val_files,
            max_per_word=max_per_word,
            unlimited=unlimited,
        )
    else:
        _build_from_clips(
            data_dir=data_dir,
            train_words=train_words,
            val_words=val_words,
            train_set=train_set,
            train_counts=train_counts,
            val_counts=val_counts,
            train_files=train_files,
            val_files=val_files,
            max_per_word=max_per_word,
            unlimited=unlimited,
        )

    _write_json(train_out, train_files)
    _write_json(val_out, val_files)

    summary = {
        "archive": str(archive_path),
        "source": source,
        "max_per_word": int(max_per_word),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "missing_train_words": [w for w, n in train_counts.items() if n == 0],
        "missing_val_words": [w for w, n in val_counts.items() if n == 0],
        "short_train_words": [w for w, n in train_counts.items() if 0 < n < max_per_word] if not unlimited else [],
        "short_val_words": [w for w, n in val_counts.items() if 0 < n < max_per_word] if not unlimited else [],
    }
    _write_json(summary_out, summary)
    logger.info("Wrote %s (%d files)", train_out, len(train_files))
    logger.info("Wrote %s (%d files)", val_out, len(val_files))
    logger.info("Summary: %s", summary_out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MSWC train/val file manifests")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mswc_en"))
    parser.add_argument("--archive", type=Path, default=Path("data/mswc_en/en.tar.gz"))
    parser.add_argument("--max-per-word", type=int, default=20)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help=(
            "Optional suffix for named manifests, e.g. max50 writes "
            "train_files_max50.json and val_files_max50.json."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--source",
        choices=["auto", "archive", "clips"],
        default="auto",
        help="Read from extracted clips when available, otherwise from the archive.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    build_file_splits(
        data_dir=args.data_dir,
        archive_path=args.archive,
        max_per_word=args.max_per_word,
        overwrite=args.overwrite,
        output_suffix=args.output_suffix,
        source=args.source,
    )


if __name__ == "__main__":
    main()
