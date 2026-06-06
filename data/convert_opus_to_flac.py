"""Convert MSWC OPUS audio files to FLAC format.

This is intended for large Colab runs where full WAV conversion would fill the
local disk. FLAC is lossless like WAV but substantially smaller for this cache.

Usage:
    python data/convert_opus_to_flac.py --clips-dir data/mswc_en/clips
    python data/convert_opus_to_flac.py --workers 12 --batch-size 16 --delete-opus
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

_CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            creationflags=_CREATION_FLAGS,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _run_ffmpeg(cmd: list[str]) -> bool:
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            creationflags=_CREATION_FLAGS,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def convert_opus_to_flac(
    opus_path: Path,
    flac_path: Path,
    sr: int = 16000,
    compression_level: int = 3,
) -> bool:
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = flac_path.with_suffix(".flac.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(opus_path),
        "-ar", str(sr),
        "-ac", "1",
        "-compression_level", str(compression_level),
        "-f", "flac",
        "-y",
        str(tmp_path),
    ]
    ok = _run_ffmpeg(cmd)
    if ok and tmp_path.exists() and tmp_path.stat().st_size > 0:
        tmp_path.replace(flac_path)
        return True
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return False


def convert_batch(
    items: list[tuple[Path, Path]],
    sr: int,
    compression_level: int,
) -> list[bool]:
    if not items:
        return []
    if len(items) == 1:
        opus, flac = items[0]
        return [convert_opus_to_flac(opus, flac, sr, compression_level)]

    seen_dirs: set[Path] = set()
    tmp_items: list[tuple[Path, Path, Path]] = []
    for opus, flac in items:
        parent = flac.parent
        if parent not in seen_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            seen_dirs.add(parent)
        tmp = flac.with_suffix(".flac.tmp")
        if tmp.exists():
            tmp.unlink()
        tmp_items.append((opus, flac, tmp))

    cmd: list[str] = ["ffmpeg", "-loglevel", "error", "-nostdin", "-y"]
    for opus, _, _ in tmp_items:
        cmd += ["-i", str(opus)]
    for idx, (_, _, tmp) in enumerate(tmp_items):
        cmd += [
            "-map", f"{idx}:a:0",
            "-ar", str(sr),
            "-ac", "1",
            "-compression_level", str(compression_level),
            "-f", "flac",
            str(tmp),
        ]

    if _run_ffmpeg(cmd):
        results: list[bool] = []
        for _, flac, tmp in tmp_items:
            ok = tmp.exists() and tmp.stat().st_size > 0
            if ok:
                tmp.replace(flac)
            elif tmp.exists():
                tmp.unlink()
            results.append(ok)
        return results

    results = []
    for opus, flac, _ in tmp_items:
        results.append(convert_opus_to_flac(opus, flac, sr, compression_level))
    return results


def _batch_worker(
    batch: list[tuple[str, str]],
    sr: int,
    compression_level: int,
    delete_opus: bool,
) -> list[tuple[str, bool, bool]]:
    items = [(Path(opus), Path(flac)) for opus, flac in batch]
    results = convert_batch(items, sr, compression_level)
    out: list[tuple[str, bool, bool]] = []
    for (opus, _), ok in zip(items, results):
        deleted = False
        if ok and delete_opus and opus.exists():
            try:
                opus.unlink()
                deleted = True
            except OSError:
                pass
        out.append((str(opus), ok, deleted))
    return out


def _iter_opus_files(clips_dir: Path):
    for root, _, files in os.walk(clips_dir):
        for name in files:
            if name.endswith(".opus"):
                yield Path(root) / name


def convert_all(
    clips_dir: Path,
    target_sr: int = 16000,
    workers: int = 0,
    delete_opus: bool = False,
    batch_size: int = 16,
    compression_level: int = 3,
    max_pending_batches: int | None = None,
) -> tuple[int, int, int]:
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1) * 2)
    batch_size = max(1, batch_size)
    max_pending_batches = max_pending_batches or max(workers * 4, 8)

    success = 0
    fail = 0
    deleted = 0
    pending = set()
    current_batch: list[tuple[str, str]] = []

    logger.info(
        "Converting OPUS->FLAC under %s with workers=%d batch_size=%d compression=%d",
        clips_dir,
        workers,
        batch_size,
        compression_level,
    )

    pbar = tqdm(desc="Converting OPUS->FLAC", unit="file")

    def submit_batch(executor: ThreadPoolExecutor, batch: list[tuple[str, str]]) -> None:
        if batch:
            pending.add(
                executor.submit(
                    _batch_worker,
                    batch,
                    target_sr,
                    compression_level,
                    delete_opus,
                )
            )

    def drain(return_when=FIRST_COMPLETED) -> None:
        nonlocal success, fail, deleted, pending
        done, pending = wait(pending, return_when=return_when)
        for fut in done:
            for _, ok, was_deleted in fut.result():
                if ok:
                    success += 1
                else:
                    fail += 1
                if was_deleted:
                    deleted += 1
                pbar.update(1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for opus_path in _iter_opus_files(clips_dir):
            flac_path = opus_path.with_suffix(".flac")
            if flac_path.exists() and flac_path.stat().st_size > 0:
                success += 1
                if delete_opus and opus_path.exists():
                    try:
                        opus_path.unlink()
                        deleted += 1
                    except OSError:
                        pass
                pbar.update(1)
                continue

            current_batch.append((str(opus_path), str(flac_path)))
            if len(current_batch) >= batch_size:
                submit_batch(executor, current_batch)
                current_batch = []
            while len(pending) >= max_pending_batches:
                drain(FIRST_COMPLETED)

        submit_batch(executor, current_batch)
        while pending:
            drain(FIRST_COMPLETED)

    pbar.close()
    return success, fail, deleted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", type=Path, default=Path("data/mswc_en/clips"))
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--delete-opus", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not check_ffmpeg():
        raise SystemExit("ffmpeg not found. Install ffmpeg first.")
    if not args.clips_dir.exists():
        raise SystemExit(f"clips dir not found: {args.clips_dir}")

    success, fail, deleted = convert_all(
        clips_dir=args.clips_dir,
        target_sr=args.target_sr,
        workers=args.workers,
        delete_opus=args.delete_opus,
        batch_size=args.batch_size,
        compression_level=args.compression_level,
    )
    logger.info(
        "Done. success=%d fail=%d deleted_opus=%d output=%s",
        success,
        fail,
        deleted,
        args.clips_dir,
    )
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
