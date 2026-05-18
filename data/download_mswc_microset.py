"""Download and prepare the official MLCommons MSWC Microset.

The Microset is a small official subset of MSWC (English + Spanish) intended
for prototyping. It is useful when Colab/local disk is not enough for MSWC
English Top500/full extraction.

Usage:
    python data/download_mswc_microset.py --language en
    python data/download_mswc_microset.py --language es
    python data/download_mswc_microset.py --language en --skip-convert
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

MICROSET_URLS = {
    "cloudflare": "https://mswc.mlcommons-storage.org/mswc_microset.tar.gz",
    "alibaba": "https://mlc-datasets.oss-cn-guangzhou.aliyuncs.com/mswc_microset.tar.gz",
    "google": "https://storage.googleapis.com/public-datasets-mswc/mswc_microset.tar.gz",
}


def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(1, retries + 1):
        resume_pos = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        try:
            with requests.get(url, stream=True, timeout=(20, 180), headers=headers) as resp:
                if resp.status_code == 416:
                    tmp.replace(dest)
                    return
                resp.raise_for_status()
                if resume_pos and resp.status_code != 206:
                    logger.warning("Server ignored Range header; restarting download for %s", dest)
                    resume_pos = 0
                total = int(resp.headers.get("content-length", 0)) + resume_pos
                mode = "ab" if resume_pos else "wb"
                with open(tmp, mode) as f, tqdm(
                    total=total,
                    initial=resume_pos,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as pbar:
                    for chunk in resp.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 5 * attempt
            logger.warning("Download failed (%s), retrying in %ss", exc, wait)
            time.sleep(wait)


def download_microset(archive: Path, mirror: str = "cloudflare") -> Path:
    if archive.exists():
        logger.info("Archive already exists: %s (%.1f MB)", archive, archive.stat().st_size / 1024**2)
        return archive

    mirrors = [mirror] + [m for m in MICROSET_URLS if m != mirror]
    last_error: Exception | None = None
    for name in mirrors:
        try:
            logger.info("Downloading MSWC Microset from %s mirror", name)
            _download_file(MICROSET_URLS[name], archive)
            return archive
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Mirror %s failed: %s", name, exc)
            archive.with_suffix(archive.suffix + ".partial").unlink(missing_ok=True)

    raise RuntimeError(f"All Microset mirrors failed: {last_error}")


def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> bool:
    target_resolved = target.resolve()
    out_path = (target / member.name).resolve()
    if not str(out_path).startswith(str(target_resolved)):
        logger.warning("Skipping unsafe tar member: %s", member.name)
        return False
    if member.isdir():
        out_path.mkdir(parents=True, exist_ok=True)
        return True
    if not member.isfile():
        logger.debug("Skipping non-file tar member: %s", member.name)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    if src is None:
        return False
    with src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return True


def extract_language(archive: Path, language: str, output_dir: Path, force: bool = False) -> None:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"mswc_microset/{language}/"
    extracted = 0
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
        if not members:
            raise RuntimeError(f"No members found for language={language!r} in {archive}")
        for member in tqdm(members, desc=f"Extracting Microset {language}"):
            rel = member.name[len(prefix):]
            if not rel:
                continue
            member.name = rel
            if _safe_extract_member(tar, member, output_dir):
                extracted += 1
    logger.info("Extracted %d members to %s", extracted, output_dir)


def _discover_audio_words(output_dir: Path) -> list[str]:
    clips = output_dir / "clips"
    return sorted(
        d.name
        for d in clips.iterdir()
        if d.is_dir() and (any(d.glob("*.wav")) or any(d.glob("*.opus")))
    )


def _infer_word_from_cell(value: str, known_words: set[str]) -> str | None:
    value = value.strip().strip('"').strip("'")
    if not value:
        return None

    lowered = value.lower()
    if lowered in known_words:
        return lowered

    parts = [p.lower() for p in value.replace("\\", "/").split("/") if p]
    for idx, part in enumerate(parts):
        if part == "clips" and idx + 1 < len(parts) and parts[idx + 1] in known_words:
            return parts[idx + 1]
        if part in known_words:
            return part

    return None


def _csv_rows(csv_path: Path) -> list[list[str]]:
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        return []

    header_tokens = {cell.strip().lower() for cell in rows[0]}
    has_header = bool(
        header_tokens
        & {
            "word",
            "words",
            "label",
            "target",
            "keyword",
            "path",
            "file",
            "filename",
            "audio",
            "audio_path",
            "clip",
            "clip_path",
            "split",
        }
    )
    return rows[1:] if has_header else rows


def _normalise_audio_rel_path(value: str, known_words: set[str]) -> str | None:
    raw = value.strip().strip('"').strip("'").replace("\\", "/")
    if not raw:
        return None

    lower = raw.lower()
    if ".opus" not in lower and ".wav" not in lower:
        return None

    parts = [p for p in raw.split("/") if p not in ("", ".", "..")]
    lower_parts = [p.lower() for p in parts]
    if "clips" in lower_parts:
        idx = lower_parts.index("clips")
        rel_parts = parts[idx:]
    else:
        word_idx = next((i for i, part in enumerate(lower_parts) if part in known_words), None)
        if word_idx is None:
            return None
        rel_parts = ["clips", *parts[word_idx:]]

    if len(rel_parts) < 3 or rel_parts[0].lower() != "clips":
        return None
    if rel_parts[1].lower() not in known_words:
        return None

    rel = Path(*rel_parts).as_posix()
    if rel.lower().endswith(".opus"):
        rel = rel[:-5] + ".wav"
    return rel


def _read_csv_entries(csv_path: Path, known_words: set[str]) -> list[tuple[str, str | None]]:
    entries: list[tuple[str, str | None]] = []
    for row in _csv_rows(csv_path):
        row_word: str | None = None
        row_path: str | None = None
        for cell in row:
            if row_path is None:
                row_path = _normalise_audio_rel_path(cell, known_words)
            if row_word is None:
                row_word = _infer_word_from_cell(cell, known_words)
        if row_word is None and row_path is not None:
            parts = row_path.split("/")
            if len(parts) >= 3 and parts[0] == "clips" and parts[1] in known_words:
                row_word = parts[1]
        if row_word is not None:
            entries.append((row_word, row_path))
    return entries


def _words_from_entries(entries: list[tuple[str, str | None]]) -> list[str]:
    return sorted({word for word, _ in entries})


def _files_from_entries(entries: list[tuple[str, str | None]]) -> list[str]:
    return sorted({path for _, path in entries if path is not None})


def _read_words_from_csv(csv_path: Path, known_words: set[str]) -> list[str]:
    found: set[str] = set()
    for word, _ in _read_csv_entries(csv_path, known_words):
        found.add(word)
    return sorted(found)


def _official_csv_word_splits(
    output_dir: Path,
    language: str,
    known_words: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]] | None:
    known = set(known_words)
    train_csv = output_dir / f"{language}_train.csv"
    dev_csv = output_dir / f"{language}_dev.csv"
    test_csv = output_dir / f"{language}_test.csv"
    splits_csv = output_dir / f"{language}_splits.csv"

    if train_csv.exists():
        train_entries = _read_csv_entries(train_csv, known)
        dev_entries = _read_csv_entries(dev_csv, known) if dev_csv.exists() else []
        test_entries = _read_csv_entries(test_csv, known) if test_csv.exists() else []
        train_words = _words_from_entries(train_entries)
        dev_words = _words_from_entries(dev_entries)
        test_words = _words_from_entries(test_entries)
        train_files = _files_from_entries(train_entries)
        dev_files = _files_from_entries(dev_entries)
        test_files = _files_from_entries(test_entries)
    elif splits_csv.exists():
        # Last-resort support for a combined CSV. If there is no explicit train
        # CSV, include every word observed in the official split manifest.
        entries = _read_csv_entries(splits_csv, known)
        train_words = _words_from_entries(entries)
        dev_words = []
        test_words = []
        train_files = _files_from_entries(entries)
        dev_files = []
        test_files = []
    else:
        return None

    # Microset CSV files are sample-level splits, not held-out-word splits:
    # the same 31 keywords can appear in train/dev/test with disjoint files.
    val_words = dev_words
    eval_words = test_words

    if train_words and dev_words and set(dev_words).issubset(set(train_words)):
        logger.info(
            "Official Microset CSV is sample-level: dev words overlap train. "
            "Using train/dev/test file manifests to avoid folder-scan leakage.",
            len(train_words),
        )
    return train_words, val_words, eval_words, train_files, dev_files, test_files


def write_word_splits(
    output_dir: Path,
    val_fraction: float,
    seed: int,
    all_words_train: bool = False,
    language: str = "en",
    split_source: str = "official",
) -> None:
    words = _discover_audio_words(output_dir)
    if not words:
        raise RuntimeError(f"Need at least 1 word, found {len(words)} in {output_dir / 'clips'}")
    if (
        not all_words_train
        and split_source == "random"
        and val_fraction > 0
        and len(words) < 2
    ):
        raise RuntimeError(f"Need at least 2 words for train/val split, found {len(words)} in {output_dir / 'clips'}")

    eval_words: list[str] = []
    train_files: list[str] = []
    val_files: list[str] = []
    eval_files: list[str] = []

    if all_words_train:
        split_source = "all_words"

    if split_source == "official":
        official = _official_csv_word_splits(output_dir, language, words)
        if official is None:
            logger.warning(
                "Official Microset CSV split files not found in %s; using all words for training.",
                output_dir,
            )
            train_words = words
            val_words = []
        else:
            train_words, val_words, eval_words, train_files, val_files, eval_files = official
            if not train_words:
                logger.warning(
                    "Official Microset CSV did not yield train words; using all words for training.",
                )
                train_words = words
                val_words = []
                eval_words = []
                train_files = []
                val_files = []
                eval_files = []
    elif split_source == "all_words" or val_fraction <= 0:
        train_words = words
        val_words: list[str] = []
    elif split_source == "random":
        if val_fraction >= 1:
            raise ValueError("--val-fraction must be < 1.0 unless --all-words-train is set")

        rng = random.Random(seed)
        shuffled = words.copy()
        rng.shuffle(shuffled)
        n_val = max(1, min(round(len(words) * val_fraction), len(words) - 1))
        val_words = sorted(shuffled[:n_val])
        train_words = sorted(shuffled[n_val:])
    else:
        raise ValueError("split_source must be one of: official, all_words, random")

    splits = output_dir / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "train_words.json").write_text(json.dumps(train_words, indent=2), encoding="utf-8")
    (splits / "val_words.json").write_text(json.dumps(val_words, indent=2), encoding="utf-8")
    (splits / "eval_words.json").write_text(json.dumps(eval_words, indent=2), encoding="utf-8")
    if train_files or val_files or eval_files:
        (splits / "train_files.json").write_text(json.dumps(train_files, indent=2), encoding="utf-8")
        (splits / "val_files.json").write_text(json.dumps(val_files, indent=2), encoding="utf-8")
        (splits / "eval_files.json").write_text(json.dumps(eval_files, indent=2), encoding="utf-8")
        logger.info(
            "Official file manifests: %d train, %d val/dev, %d eval/test files",
            len(train_files),
            len(val_files),
            len(eval_files),
        )
    else:
        for name in ("train_files.json", "val_files.json", "eval_files.json"):
            (splits / name).unlink(missing_ok=True)
    logger.info(
        "Splits (%s): %d train, %d val, %d eval -> %s",
        split_source,
        len(train_words),
        len(val_words),
        len(eval_words),
        splits,
    )


def convert_opus(output_dir: Path, workers: int, delete_opus: bool) -> None:
    clips = output_dir / "clips"
    opus_n = sum(1 for _ in clips.rglob("*.opus"))
    if opus_n == 0:
        logger.info("No OPUS files found; conversion skipped")
        return

    try:
        subprocess.run(
            ["apt-get", "install", "-qq", "ffmpeg"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass

    cmd = [
        sys.executable,
        "data/convert_opus.py",
        "--clips-dir",
        str(clips),
        "--workers",
        str(workers),
    ]
    if delete_opus:
        cmd.append("--delete-opus")
    subprocess.run(cmd, check=True)


def summarize(output_dir: Path) -> None:
    clips = output_dir / "clips"
    wav_n = sum(1 for _ in clips.rglob("*.wav"))
    opus_n = sum(1 for _ in clips.rglob("*.opus"))
    words = sorted(d for d in clips.iterdir() if d.is_dir())
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1024**2
    logger.info("Microset ready: %d words, %d WAV, %d OPUS, %.1f MB at %s",
                len(words), wav_n, opus_n, total_size, output_dir)


def _has_existing_audio(output_dir: Path) -> bool:
    clips = output_dir / "clips"
    return clips.exists() and (
        any(clips.rglob("*.wav")) or any(clips.rglob("*.opus"))
    )


def _has_splits(output_dir: Path) -> bool:
    splits = output_dir / "splits"
    return (
        (splits / "train_words.json").exists()
        and (splits / "val_words.json").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MLCommons MSWC Microset")
    parser.add_argument("--language", choices=["en", "es"], default="en")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=Path("data/mswc_microset.tar.gz"))
    parser.add_argument("--mirror", choices=list(MICROSET_URLS), default="cloudflare")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--keep-opus", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split-source",
        choices=["official", "all_words", "random"],
        default="official",
        help=(
            "How to create split JSONs. 'official' reads language CSV files from "
            "the Microset archive and falls back to all words if the CSV files are missing."
        ),
    )
    parser.add_argument(
        "--all-words-train",
        action="store_true",
        help=(
            "Put every Microset word into train_words.json and leave val_words.json empty. "
            "Recommended for the 31-word English Microset when checkpoints are selected by GSC-dev."
        ),
    )
    parser.add_argument(
        "--rewrite-splits",
        action="store_true",
        help="Rewrite split JSON files even when existing Microset audio/splits are already present.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = args.output_dir or Path(f"data/mswc_microset_{args.language}")
    if _has_existing_audio(output_dir) and not args.force:
        logger.info("Existing Microset audio found at %s; skipping archive download/extract", output_dir)
    else:
        archive = download_microset(args.archive, mirror=args.mirror)
        extract_language(archive, args.language, output_dir, force=args.force)
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
            logger.info("Deleted archive: %s", archive)

    if not args.skip_convert:
        convert_opus(output_dir, workers=args.workers, delete_opus=not args.keep_opus)
    if not _has_splits(output_dir) or args.force or args.rewrite_splits:
        write_word_splits(
            output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            all_words_train=args.all_words_train,
            language=args.language,
            split_source=args.split_source,
        )
    else:
        logger.info("Existing Microset splits found at %s", output_dir / "splits")
    summarize(output_dir)


if __name__ == "__main__":
    main()
