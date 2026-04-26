"""Convert MSWC OPUS audio files to WAV format (threaded + batched, resume-friendly).

Required on Windows where torchaudio may not support OPUS directly.
Requires ffmpeg installed and available in PATH.

Speed tricks used here:
  * ThreadPoolExecutor instead of ProcessPoolExecutor — subprocess already
    releases the GIL, and Windows `spawn` for processes is very expensive.
  * Batched ffmpeg calls: one ffmpeg invocation converts many files in a row
    (amortises the ~100-300 ms process startup cost over the batch).
  * DEVNULL instead of capture_output (no pipe allocation per call).

Usage:
    python data/convert_opus.py                       # sensible defaults
    python data/convert_opus.py --workers 16          # explicit thread count
    python data/convert_opus.py --batch-size 32       # files per ffmpeg call
    python data/convert_opus.py --batch-size 1        # disable batching
    python data/convert_opus.py --processes           # legacy process pool
    python data/convert_opus.py --delete-opus         # remove .opus on success
"""

import argparse
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

MSWC_DIR = Path("data/mswc_en")

# On Windows, suppress the flashing console windows for each ffmpeg call.
_CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
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
    except (subprocess.CalledProcessError, OSError):
        return False


def convert_opus_to_wav(opus_path: Path, wav_path: Path, sr: int = 16000) -> bool:
    """Convert a single OPUS file to WAV at target sample rate."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(opus_path),
        "-ar", str(sr),
        "-ac", "1",
        "-y",
        str(wav_path),
    ]
    ok = _run_ffmpeg(cmd)
    if not ok and wav_path.exists():
        try:
            wav_path.unlink()
        except OSError:
            pass
    return ok


def convert_batch(items: list[tuple[Path, Path]], sr: int = 16000) -> list[bool]:
    """Convert a batch of (opus, wav) pairs in a SINGLE ffmpeg invocation.

    Falls back to per-file conversion for the whole batch on failure so we
    can still report accurate per-file success/failure.
    """
    if not items:
        return []
    if len(items) == 1:
        opus, wav = items[0]
        return [convert_opus_to_wav(opus, wav, sr)]

    # Pre-create directories once per unique parent.
    seen_dirs: set[Path] = set()
    for _, wav in items:
        parent = wav.parent
        if parent not in seen_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            seen_dirs.add(parent)

    cmd: list[str] = ["ffmpeg", "-loglevel", "error", "-nostdin", "-y"]
    for opus, _ in items:
        cmd += ["-i", str(opus)]
    for idx, (_, wav) in enumerate(items):
        cmd += [
            "-map", f"{idx}:a:0",
            "-ar", str(sr),
            "-ac", "1",
            str(wav),
        ]

    if _run_ffmpeg(cmd):
        return [wav.exists() and wav.stat().st_size > 0 for _, wav in items]

    # Batch failed — retry each file individually to isolate the bad one.
    results: list[bool] = []
    for opus, wav in items:
        results.append(convert_opus_to_wav(opus, wav, sr))
    return results


def _batch_worker(
    batch: list[tuple[str, str]], sr: int, delete_opus: bool
) -> list[tuple[str, bool, bool]]:
    """Process/thread worker that converts a batch.

    Returns list of (opus_path_str, success, deleted_opus).
    """
    items = [(Path(o), Path(w)) for o, w in batch]
    results = convert_batch(items, sr)

    out: list[tuple[str, bool, bool]] = []
    for (opus, wav), ok in zip(items, results):
        deleted = False
        if ok and delete_opus and opus.exists():
            try:
                opus.unlink()
                deleted = True
            except OSError:
                pass
        out.append((str(opus), ok, deleted))
    return out


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def convert_all(
    clips_dir: Path,
    target_sr: int = 16000,
    workers: int = 0,
    delete_opus: bool = False,
    batch_size: int = 16,
    use_processes: bool = False,
) -> tuple[int, int]:
    """Convert all OPUS files in clips_dir to WAV (threaded + batched, resume-friendly).

    Args:
        clips_dir: Directory containing word subdirectories with OPUS files.
        target_sr: Target sample rate for WAV output.
        workers: Number of parallel workers. 0 = auto.
        delete_opus: Remove .opus file after successful conversion.
        batch_size: Number of files per ffmpeg invocation (1 = no batching).
        use_processes: Use ProcessPoolExecutor (legacy) instead of threads.

    Returns:
        (success_count, fail_count)
    """
    logger.info("Scanning %s for OPUS files...", clips_dir)
    opus_files = list(clips_dir.rglob("*.opus"))
    total = len(opus_files)
    if total == 0:
        logger.info("No OPUS files found in %s", clips_dir)
        return 0, 0

    pending: list[tuple[str, str]] = []
    already_done = 0
    cleanup_only: list[Path] = []  # already-converted files whose .opus should still be deleted
    for opus_path in opus_files:
        wav_path = opus_path.with_suffix(".wav")
        if wav_path.exists() and wav_path.stat().st_size > 0:
            already_done += 1
            if delete_opus and opus_path.exists():
                cleanup_only.append(opus_path)
        else:
            pending.append((str(opus_path), str(wav_path)))

    logger.info(
        "Found %d OPUS, %d already converted, %d pending, %d to cleanup",
        total, already_done, len(pending), len(cleanup_only),
    )

    # Fast path: only cleanup leftover .opus files (no conversion needed).
    if cleanup_only and not pending:
        for p in cleanup_only:
            try:
                p.unlink()
            except OSError:
                pass
        return already_done, 0

    if not pending and not cleanup_only:
        return already_done, 0

    # Delete leftover .opus files in the background-ish (cheap, just unlink calls).
    if cleanup_only:
        for p in tqdm(cleanup_only, desc="Removing already-converted .opus"):
            try:
                p.unlink()
            except OSError:
                pass

    if workers <= 0:
        # Threads: ffmpeg is mostly external — oversubscribe a bit.
        # Processes: stay close to CPU count (spawn is expensive).
        cpu = os.cpu_count() or 1
        workers = cpu * 2 if not use_processes else cpu
    workers = max(1, min(workers, max(1, len(pending))))
    batch_size = max(1, batch_size)

    logger.info(
        "Using %d %s workers, batch_size=%d",
        workers,
        "process" if use_processes else "thread",
        batch_size,
    )

    success = already_done
    fail = 0

    batches = list(_chunked(pending, batch_size))

    def _run_serial() -> None:
        nonlocal success, fail
        for batch in tqdm(batches, desc=f"Converting OPUS->WAV (batch={batch_size})"):
            for _, ok, _ in _batch_worker(batch, target_sr, delete_opus):
                if ok:
                    success += 1
                else:
                    fail += 1

    if workers == 1:
        _run_serial()
        return success, fail

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    try:
        with Executor(max_workers=workers) as ex:
            futures = [
                ex.submit(_batch_worker, batch, target_sr, delete_opus)
                for batch in batches
            ]
            desc = (
                f"Converting OPUS->WAV "
                f"({workers} {'procs' if use_processes else 'threads'}, batch={batch_size})"
            )
            # tqdm counts batches; multiply by batch for file-level feel if desired.
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    for _, ok, _ in fut.result():
                        if ok:
                            success += 1
                        else:
                            fail += 1
                except Exception as e:  # noqa: BLE001
                    logger.error("Worker error: %s", e)
                    fail += 1
    except KeyboardInterrupt:
        logger.warning("Interrupted — partial results: %d success, %d failed", success, fail)
        raise

    return success, fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MSWC OPUS to WAV (threaded + batched)")
    parser.add_argument("--clips-dir", type=str,
                        default=str(MSWC_DIR / "clips"),
                        help="Directory containing word/*.opus files")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = auto: 2x CPU for threads, 1x for processes)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Files per ffmpeg invocation (1 = one file per call, default 16)")
    parser.add_argument("--processes", action="store_true",
                        help="Use ProcessPoolExecutor instead of threads (legacy, slower on Windows)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Target sample rate")
    parser.add_argument("--delete-opus", action="store_true",
                        help="Remove .opus file after successful conversion")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not check_ffmpeg():
        logger.error(
            "ffmpeg not found. Install ffmpeg and add to PATH.\n"
            "  Windows: choco install ffmpeg  OR  download from https://ffmpeg.org/\n"
            "  Linux:   sudo apt install ffmpeg\n"
            "  Colab:   !apt-get install -qq ffmpeg"
        )
        return

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        logger.error("Clips directory not found: %s", clips_dir)
        logger.info("Run download_mswc.py first.")
        return

    success, fail = convert_all(
        clips_dir,
        target_sr=args.sr,
        workers=args.workers,
        delete_opus=args.delete_opus,
        batch_size=args.batch_size,
        use_processes=args.processes,
    )
    logger.info("Conversion complete: %d success, %d failed", success, fail)


if __name__ == "__main__":
    main()
