"""Download Google Speech Commands v2 dataset.

Downloads ~2.3GB archive and extracts to data/gsc_v2/.
Contains 35 word folders + _background_noise_/.
"""

import logging
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
GSC_DIR = Path("data/gsc_v2")
ARCHIVE_PATH = Path("data/speech_commands_v0.02.tar.gz")


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Archive already exists: %s", dest)
        return

    logger.info("Downloading %s ...", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract tar.gz archive."""
    if dest.exists() and any(dest.iterdir()):
        logger.info("Already extracted: %s", dest)
        return

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s ...", archive, dest)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_file(GSC_URL, ARCHIVE_PATH)
    extract_archive(ARCHIVE_PATH, GSC_DIR)

    word_dirs = [d for d in GSC_DIR.iterdir() if d.is_dir()]
    logger.info("GSC v2 downloaded: %d directories", len(word_dirs))
    for d in sorted(word_dirs):
        n_files = len(list(d.glob("*.wav")))
        logger.info("  %s: %d files", d.name, n_files)


if __name__ == "__main__":
    main()
