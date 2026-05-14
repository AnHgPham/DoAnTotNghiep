"""MSWC Drive Cache - save/load pre-converted WAV files on Google Drive.

The cache avoids repeating the expensive MSWC OPUS-to-WAV conversion on each
Colab session. It is intentionally keyed by split/extraction policy so a small
Top500 cache cannot be mistaken for a full-vocabulary cache.

Typical notebook use:
    from data.mswc_drive_cache import setup_mswc_from_drive
    _from_drive_cache = setup_mswc_from_drive(
        DRIVE_PROJECT,
        split_mode="top500",
        max_per_word=0,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

LOCAL_MSWC = Path("data/mswc_en")
LOCAL_CLIPS = LOCAL_MSWC / "clips"
SPLIT_FILES = ("train_words.json", "val_words.json", "eval_words.json")


def cache_dir_name(split_mode: str = "top500", max_per_word: int = 0) -> str:
    """Return a Drive cache directory name tied to split/extraction policy."""
    cap = "full" if int(max_per_word) <= 0 else f"mpw{int(max_per_word)}"
    return f"mswc_en_wav_{split_mode}_{cap}"


def _drive_mswc_dir(
    drive_project: str | Path,
    split_mode: str = "top500",
    max_per_word: int = 0,
) -> Path:
    return Path(drive_project) / cache_dir_name(split_mode, max_per_word)


def _count_wav_words(clips_dir: Path) -> int:
    """Count word directories that contain at least one WAV file."""
    if not clips_dir.exists():
        return 0
    return sum(
        1
        for d in clips_dir.iterdir()
        if d.is_dir() and any(d.glob("*.wav"))
    )


def _count_wavs(clips_dir: Path) -> int:
    """Count total WAV files under clips_dir."""
    if not clips_dir.exists():
        return 0
    return sum(1 for _ in clips_dir.rglob("*.wav"))


def _load_manifest(cache_dir: Path) -> dict:
    manifest = cache_dir / "manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_list(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _split_words(splits_dir: Path) -> tuple[list[str], list[str], list[str]]:
    return (
        _load_json_list(splits_dir / "train_words.json"),
        _load_json_list(splits_dir / "val_words.json"),
        _load_json_list(splits_dir / "eval_words.json"),
    )


def _discover_wav_words(clips_dir: Path) -> list[str]:
    if not clips_dir.exists():
        return []
    return sorted(
        d.name
        for d in clips_dir.iterdir()
        if d.is_dir() and any(d.glob("*.wav"))
    )


def _word_has_wav(clips_dir: Path, word: str) -> bool:
    word_dir = clips_dir / word
    return word_dir.is_dir() and any(word_dir.glob("*.wav"))


def _coverage_for_words(clips_dir: Path, words: list[str]) -> tuple[int, int, float]:
    if not words:
        return 0, 0, 0.0
    present = sum(1 for word in words if _word_has_wav(clips_dir, word))
    total = len(words)
    return present, total, present / max(total, 1)


def drive_cache_status(
    drive_project: str | Path,
    split_mode: str = "top500",
    max_per_word: int = 0,
    count_wavs: bool = False,
) -> dict:
    """Inspect a Drive cache without mutating local files.

    Google Drive's FUSE mount is very slow when recursively listing tens of
    thousands of small files. By default this avoids a full ``*.wav`` scan and
    uses the manifest/lower bound from split coverage instead. Pass
    ``count_wavs=True`` only for local/temp-dir diagnostics.
    """
    drive_mswc = _drive_mswc_dir(drive_project, split_mode, max_per_word)
    drive_clips = drive_mswc / "clips"
    drive_splits = drive_mswc / "splits"
    manifest = _load_manifest(drive_mswc)

    train_words, val_words, eval_words = _split_words(drive_splits)
    required_words = train_words + val_words
    train_present, train_total, train_cov = _coverage_for_words(
        drive_clips, train_words,
    )
    val_present, val_total, val_cov = _coverage_for_words(
        drive_clips, val_words,
    )
    required_present, required_total, required_cov = _coverage_for_words(
        drive_clips, required_words,
    )
    if count_wavs:
        n_wav = _count_wavs(drive_clips)
    else:
        n_wav = int(manifest.get("n_wav") or required_present)

    return {
        "cache_dir": str(drive_mswc),
        "clips_dir": str(drive_clips),
        "splits_dir": str(drive_splits),
        "split_mode": split_mode,
        "max_per_word": int(max_per_word),
        "has_splits": bool(train_words and val_words),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "eval_words": len(eval_words),
        "train_present": train_present,
        "train_total": train_total,
        "train_coverage": train_cov,
        "val_present": val_present,
        "val_total": val_total,
        "val_coverage": val_cov,
        "required_present": required_present,
        "required_total": required_total,
        "required_coverage": required_cov,
        "n_word_dirs": _count_wav_words(drive_clips),
        "n_wav": n_wav,
        "n_wav_exact": bool(count_wavs or manifest.get("n_wav")),
    }


def is_drive_cache_valid(
    drive_project: str | Path,
    split_mode: str = "top500",
    max_per_word: int = 0,
    min_train_val_coverage: float = 0.9,
) -> tuple[bool, dict]:
    """Validate that Drive has splits and enough WAV-backed train/val words."""
    status = drive_cache_status(drive_project, split_mode, max_per_word)
    valid = (
        status["has_splits"]
        and status["required_total"] > 0
        and status["required_coverage"] >= float(min_train_val_coverage)
    )
    status["valid"] = valid
    status["min_train_val_coverage"] = float(min_train_val_coverage)
    return valid, status


def check_drive_cache(
    drive_project: str,
    split_mode: str = "top500",
    max_per_word: int = 0,
) -> tuple[Path, Path, int]:
    """Check if Drive has cached WAV files.

    Returns:
        (drive_clips_path, drive_splits_path, n_word_dirs)
    """
    drive_mswc = _drive_mswc_dir(drive_project, split_mode, max_per_word)
    drive_clips = drive_mswc / "clips"
    drive_splits = drive_mswc / "splits"
    return drive_clips, drive_splits, _count_wav_words(drive_clips)


def _copy_tree_if_missing(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if not dst.exists():
        shutil.copytree(str(src), str(dst))
        return
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            _copy_tree_if_missing(child, target)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(child), str(target))


def _write_split_lists(
    splits_dir: Path,
    train_words: list[str],
    val_words: list[str],
    eval_words: list[str] | None = None,
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train_words.json": train_words,
        "val_words.json": val_words,
        "eval_words.json": eval_words or [],
    }
    for name, words in payloads.items():
        (splits_dir / name).write_text(
            json.dumps(words, indent=2),
            encoding="utf-8",
        )


def _candidate_split_dirs(drive_project: str | Path) -> list[Path]:
    drive_project = Path(drive_project)
    return [
        drive_project / "mswc_en" / "splits",
        drive_project / "mswc_en_wav" / "splits",
        drive_project / "mswc_en_wav_top500" / "splits",
        LOCAL_MSWC / "splits",
    ]


def repair_drive_cache_splits(
    drive_project: str | Path,
    split_mode: str = "top500",
    max_per_word: int = 0,
    min_words: int = 30,
) -> bool:
    """Repair a partial Drive WAV cache that has clips but no split JSON files.

    Older notebook versions only cached ``clips/``. Re-downloading the full
    32GB MSWC archive just to recover split files is wasteful, so this function
    first tries nearby legacy split locations and then falls back to a stable
    train/val split discovered from WAV-backed word directories.
    """
    drive_mswc = _drive_mswc_dir(drive_project, split_mode, max_per_word)
    drive_clips = drive_mswc / "clips"
    drive_splits = drive_mswc / "splits"

    if drive_splits.exists():
        train_words, val_words, _ = _split_words(drive_splits)
        available = set(_discover_wav_words(drive_clips))
        required_words = train_words + val_words
        present = sum(1 for word in required_words if word in available)
        coverage = present / max(len(required_words), 1)
        if train_words and val_words and coverage >= 0.9:
            return True

    wav_words = _discover_wav_words(drive_clips)
    if len(wav_words) < min_words:
        return False

    for candidate in _candidate_split_dirs(drive_project):
        if candidate == drive_splits:
            continue
        train_words, val_words, eval_words = _split_words(candidate)
        if not (train_words and val_words):
            continue
        available = set(wav_words)
        repaired_train = [w for w in train_words if w in available]
        repaired_val = [w for w in val_words if w in available]
        if repaired_train and repaired_val:
            _write_split_lists(drive_splits, repaired_train, repaired_val, eval_words)
            logger.info(
                "Repaired Drive cache splits from %s: %d train, %d val",
                candidate,
                len(repaired_train),
                len(repaired_val),
            )
            return True

    # Last-resort recovery: train with the words that are actually cached.
    # Keep it deterministic so future runs use the same word partition.
    n_val = max(1, min(round(len(wav_words) * 0.1), len(wav_words) - 1))
    val_words = wav_words[-n_val:]
    train_words = wav_words[:-n_val]
    _write_split_lists(drive_splits, train_words, val_words, [])
    logger.info(
        "Repaired Drive cache splits from cached WAV dirs: %d train, %d val",
        len(train_words),
        len(val_words),
    )
    return True


def _link_clips_to_drive(drive_clips: Path) -> None:
    """Link local clips to Drive, with Windows junction fallback."""
    if LOCAL_CLIPS.is_symlink():
        LOCAL_CLIPS.unlink()

    if LOCAL_CLIPS.exists():
        n_local = _count_wavs(LOCAL_CLIPS)
        manifest = _load_manifest(drive_clips.parent)
        n_drive = int(manifest.get("n_wav") or 0)
        if n_drive > 0 and n_local >= n_drive * 0.9:
            logger.info("Local clips already have %d WAV files - keeping", n_local)
            return
        logger.info(
            "Local clips incomplete (%d WAV). Replacing with Drive link...",
            n_local,
        )
        shutil.rmtree(LOCAL_CLIPS)

    LOCAL_CLIPS.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(
            str(drive_clips.resolve()),
            str(LOCAL_CLIPS),
            target_is_directory=True,
        )
        logger.info("Symlinked: %s -> %s", LOCAL_CLIPS, drive_clips)
        return
    except OSError as exc:
        if os.name != "nt":
            raise
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(LOCAL_CLIPS), str(drive_clips.resolve())],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Created junction: %s -> %s", LOCAL_CLIPS, drive_clips)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(
                "Could not link Drive cache (%s); copying clips locally instead.",
                exc,
            )
            shutil.copytree(str(drive_clips), str(LOCAL_CLIPS), dirs_exist_ok=True)


def load_from_drive(drive_clips: Path, drive_splits: Path) -> bool:
    """Symlink local clips to Drive and copy splits/metadata."""
    LOCAL_MSWC.mkdir(parents=True, exist_ok=True)

    local_splits = LOCAL_MSWC / "splits"
    if drive_splits.exists():
        _copy_tree_if_missing(drive_splits, local_splits)
        logger.info("Splits ready at %s", local_splits)

    drive_meta = drive_clips.parent / "metadata"
    local_meta = LOCAL_MSWC / "metadata"
    if drive_meta.exists():
        _copy_tree_if_missing(drive_meta, local_meta)
        logger.info("Metadata ready at %s", local_meta)

    _link_clips_to_drive(drive_clips)
    manifest = _load_manifest(drive_clips.parent)
    n_wav = manifest.get("n_wav", "unknown")
    logger.info("MSWC ready from Drive cache (%s WAV in manifest)", n_wav)
    return True


def save_to_drive(
    drive_project: str,
    split_mode: str = "top500",
    max_per_word: int = 0,
) -> bool:
    """Save local WAV files + splits + metadata to Drive for future sessions."""
    if not LOCAL_CLIPS.exists():
        logger.warning("No local clips to save")
        return False

    n_wav = _count_wavs(LOCAL_CLIPS)
    if n_wav == 0:
        logger.warning("No WAV files found in %s", LOCAL_CLIPS)
        return False

    drive_mswc = _drive_mswc_dir(drive_project, split_mode, max_per_word)
    drive_clips = drive_mswc / "clips"

    manifest = _load_manifest(drive_mswc)
    n_existing = int(manifest.get("n_wav") or 0)
    if n_existing >= n_wav:
        logger.info("Drive cache already up-to-date (%d WAV files)", n_existing)
    else:
        logger.info("Saving %d WAV files to Drive cache: %s", n_wav, drive_mswc)
        drive_mswc.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["cp", "-ru", str(LOCAL_CLIPS), str(drive_mswc) + "/"],
                check=True,
            )
            logger.info("WAV clips copied to Drive")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("cp -ru not available, using Python copy...")
            _copy_tree_if_missing(LOCAL_CLIPS, drive_clips)

    local_splits = LOCAL_MSWC / "splits"
    if local_splits.exists():
        _copy_tree_if_missing(local_splits, drive_mswc / "splits")
        logger.info("Splits saved to Drive")

    local_meta = LOCAL_MSWC / "metadata"
    if local_meta.exists():
        _copy_tree_if_missing(local_meta, drive_mswc / "metadata")
        logger.info("Metadata saved to Drive")

    train_words, val_words, eval_words = _split_words(drive_mswc / "splits")
    manifest = {
        "created_at": int(time.time()),
        "split_mode": split_mode,
        "max_per_word": int(max_per_word),
        "n_wav": n_wav,
        "n_word_dirs": _count_wav_words(LOCAL_CLIPS),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "eval_words": len(eval_words),
        "cache_dir": str(drive_mswc),
    }
    (drive_mswc / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    logger.info("Drive cache saved: %d WAV files at %s", manifest["n_wav"], drive_mswc)
    logger.info("Lan chay tiep theo se dung Drive cache tu dong!")
    return True


def download_and_convert(
    n_cpu: int = 0,
    split_mode: str = "top500",
    max_per_word: int = 0,
) -> None:
    """Run the full download + extract + OPUS-to-WAV conversion pipeline."""
    if LOCAL_CLIPS.is_symlink():
        logger.info("Removing stale local MSWC clips symlink before cache rebuild: %s", LOCAL_CLIPS)
        LOCAL_CLIPS.unlink()

    logger.info("Downloading MSWC English...")
    cmd = [sys.executable, "data/download_mswc.py"]
    if split_mode == "top500":
        cmd.append("--top500-splits")
    elif split_mode != "full":
        raise ValueError("split_mode must be 'top500' or 'full'")
    cmd.extend(["--max-per-word", str(int(max_per_word))])
    subprocess.run(cmd, check=True)

    if not LOCAL_CLIPS.exists():
        logger.error("Download failed - no clips directory created")
        return

    n_opus = sum(1 for _ in LOCAL_CLIPS.rglob("*.opus"))
    n_wav = sum(1 for _ in LOCAL_CLIPS.rglob("*.wav"))
    logger.info("After download: %d OPUS, %d WAV", n_opus, n_wav)

    if n_opus > 0 and n_wav < n_opus:
        if n_cpu <= 0:
            n_cpu = os.cpu_count() or 8
        logger.info("Converting %d OPUS -> WAV with %d workers...", n_opus, n_cpu)

        try:
            subprocess.run(
                ["apt-get", "install", "-qq", "ffmpeg"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

        subprocess.run(
            [
                sys.executable,
                "data/convert_opus.py",
                "--workers",
                str(n_cpu),
                "--delete-opus",
            ],
            check=True,
        )

    logger.info("Conversion complete: %d WAV files", _count_wavs(LOCAL_CLIPS))


def setup_mswc_from_drive(
    drive_project: str,
    split_mode: str = "top500",
    max_per_word: int = 0,
    min_train_val_coverage: float = 0.9,
    n_cpu: int = 0,
    min_words: int | None = None,
) -> bool:
    """Check Drive cache, load it if valid, otherwise build and save it.

    Args:
        drive_project: Drive output directory.
        split_mode: 'top500' or 'full'.
        max_per_word: MSWC extraction cap. 0 means unlimited/full per word.
        min_train_val_coverage: Required fraction of train+val words with WAVs.
        n_cpu: Workers for OPUS-to-WAV conversion. 0 means auto.
        min_words: Deprecated compatibility parameter. Ignored by the new
            split-aware validation.

    Returns:
        True if loaded from Drive cache; False if the full pipeline ran.
    """
    del min_words
    valid, status = is_drive_cache_valid(
        drive_project,
        split_mode=split_mode,
        max_per_word=max_per_word,
        min_train_val_coverage=min_train_val_coverage,
    )
    drive_clips = Path(status["clips_dir"])
    drive_splits = Path(status["splits_dir"])

    if valid:
        logger.info(
            "Drive WAV cache valid: %d/%d train+val words (%.1f%%), %d WAV",
            status["required_present"],
            status["required_total"],
            100.0 * status["required_coverage"],
            status["n_wav"],
        )
        logger.info(">>> SU DUNG DRIVE CACHE - skip download + extract + convert <<<")
        load_from_drive(drive_clips, drive_splits)
        return True

    if (
        status["n_word_dirs"] >= 30
        and (
            not status["has_splits"]
            or (
                status["required_total"] > 0
                and status["required_coverage"] < float(min_train_val_coverage)
            )
        )
    ):
        logger.info(
            "Drive cache has %d WAV word dirs but invalid splits "
            "(has_splits=%s, coverage=%.1f%%). Repairing splits...",
            status["n_word_dirs"],
            status["has_splits"],
            100.0 * status["required_coverage"],
        )
        if repair_drive_cache_splits(
            drive_project,
            split_mode=split_mode,
            max_per_word=max_per_word,
        ):
            valid, status = is_drive_cache_valid(
                drive_project,
                split_mode=split_mode,
                max_per_word=max_per_word,
                min_train_val_coverage=min_train_val_coverage,
            )
            drive_clips = Path(status["clips_dir"])
            drive_splits = Path(status["splits_dir"])
            if valid:
                logger.info(
                    "Drive WAV cache repaired: %d/%d train+val words (%.1f%%)",
                    status["required_present"],
                    status["required_total"],
                    100.0 * status["required_coverage"],
                )
                load_from_drive(drive_clips, drive_splits)
                return True

    if status["n_word_dirs"] > 0:
        logger.info(
            "Drive cache invalid: has_splits=%s, coverage=%.1f%%, n_words=%d. Download moi...",
            status["has_splits"],
            100.0 * status["required_coverage"],
            status["n_word_dirs"],
        )
    else:
        logger.info("Chua co Drive WAV cache. Se download + extract + convert tu dau.")

    try:
        free_gb = shutil.disk_usage("/content").free / 1024**3
        logger.info("Free disk: %.1f GB", free_gb)
        if max_per_word <= 0:
            if free_gb < 80:
                logger.error(
                    "Full extraction needs large temporary space. "
                    "Need at least ~80GB, recommended 150GB+. Current: %.1fGB",
                    free_gb,
                )
                return False
            if free_gb < 150:
                logger.warning(
                    "Full extraction may be tight with %.1fGB free. "
                    "Use a smaller cap only for smoke tests, not final results.",
                    free_gb,
                )
        elif free_gb < 35:
            logger.error("Can ~35GB cho tar. Hien tai: %.1fGB", free_gb)
            return False
    except OSError:
        pass

    download_and_convert(
        n_cpu=n_cpu,
        split_mode=split_mode,
        max_per_word=max_per_word,
    )
    save_to_drive(
        drive_project,
        split_mode=split_mode,
        max_per_word=max_per_word,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSWC Drive Cache - save/load WAV on Google Drive",
    )
    parser.add_argument(
        "--drive-project",
        type=str,
        default="/content/drive/MyDrive/DoAnTotNghiep_output",
        help="Drive project directory",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split-mode", choices=["top500", "full"], default="top500")
    parser.add_argument("--max-per-word", type=int, default=0)
    parser.add_argument("--min-train-val-coverage", type=float, default=0.9)
    parser.add_argument("--min-words", type=int, default=None, help="Deprecated")
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Only save existing local WAV to Drive (no download)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.save_only:
        save_to_drive(
            args.drive_project,
            split_mode=args.split_mode,
            max_per_word=args.max_per_word,
        )
        return

    from_cache = setup_mswc_from_drive(
        args.drive_project,
        split_mode=args.split_mode,
        max_per_word=args.max_per_word,
        min_train_val_coverage=args.min_train_val_coverage,
        n_cpu=args.workers,
        min_words=args.min_words,
    )
    if from_cache:
        logger.info("Loaded from Drive cache - no conversion needed!")
    else:
        logger.info("Full pipeline complete - WAV saved to Drive for next time")


if __name__ == "__main__":
    main()
