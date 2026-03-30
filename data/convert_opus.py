"""Convert MSWC OPUS audio files to WAV format.

Required on Windows where torchaudio may not support OPUS directly.
Requires ffmpeg installed and available in PATH.
"""

import logging
import subprocess
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

MSWC_DIR = Path("data/mswc_en")


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def convert_opus_to_wav(opus_path: Path, wav_path: Path, sr: int = 16000) -> bool:
    """Convert a single OPUS file to WAV at target sample rate.

    Args:
        opus_path: Input OPUS file.
        wav_path: Output WAV file.
        sr: Target sample rate.

    Returns:
        True if conversion succeeded.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(opus_path),
                "-ar", str(sr),
                "-ac", "1",
                "-y",
                str(wav_path),
            ],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to convert %s: %s", opus_path, e.stderr.decode())
        return False


def convert_all(clips_dir: Path, target_sr: int = 16000) -> tuple[int, int]:
    """Convert all OPUS files in clips_dir to WAV.

    Args:
        clips_dir: Directory containing word subdirectories with OPUS files.
        target_sr: Target sample rate for WAV output.

    Returns:
        (success_count, fail_count)
    """
    opus_files = list(clips_dir.rglob("*.opus"))
    if not opus_files:
        logger.info("No OPUS files found in %s", clips_dir)
        return 0, 0

    logger.info("Found %d OPUS files to convert", len(opus_files))
    success, fail = 0, 0

    for opus_path in tqdm(opus_files, desc="Converting OPUS->WAV"):
        wav_path = opus_path.with_suffix(".wav")
        if wav_path.exists():
            success += 1
            continue
        if convert_opus_to_wav(opus_path, wav_path, target_sr):
            success += 1
        else:
            fail += 1

    return success, fail


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not check_ffmpeg():
        logger.error(
            "ffmpeg not found. Install ffmpeg and add to PATH.\n"
            "  Windows: choco install ffmpeg  OR  download from https://ffmpeg.org/\n"
            "  Linux:   sudo apt install ffmpeg"
        )
        return

    clips_dir = MSWC_DIR / "clips"
    if not clips_dir.exists():
        logger.error("Clips directory not found: %s", clips_dir)
        logger.info("Run download_mswc.py first.")
        return

    success, fail = convert_all(clips_dir)
    logger.info("Conversion complete: %d success, %d failed", success, fail)


if __name__ == "__main__":
    main()
