"""Download MSWC English subset for training and evaluation.

Downloads the English-only audio (32.5GB OPUS) from MLCommons mirrors,
then extracts only the ~985 words needed for training/evaluation.

After download, convert OPUS->WAV: python data/convert_opus.py

Usage:
    python data/download_mswc.py                    # full pipeline
    python data/download_mswc.py --splits-only       # only metadata + splits
    python data/download_mswc.py --from-archive PATH # extract from local tar.gz
    python data/download_mswc.py --max-per-word 300
"""

import argparse
import gzip
import json
import logging
import shutil
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

MSWC_DIR = Path("data/mswc_en")
SPLITS_DIR = MSWC_DIR / "splits"
CLIPS_DIR = MSWC_DIR / "clips"

MIRRORS = {
    "cloudflare": "https://mswc.mlcommons-storage.org",
    "alibaba": "https://mlc-datasets.oss-cn-guangzhou.aliyuncs.com",
    "google": "https://storage.googleapis.com/public-datasets-mswc",
}

METADATA_URL = "https://mswc.mlcommons-storage.org/metadata.json.gz"


# ───────────────────── Metadata / Word Counts ───────────────────

def load_word_counts() -> dict[str, int]:
    """Get English word -> utterance count from MSWC metadata.

    Tries local cache, then downloads metadata.json.gz (103MB) from
    MLCommons and extracts the English section.
    """
    cache_path = MSWC_DIR / "metadata" / "en_word_counts.json"
    if cache_path.exists():
        logger.info("Loading cached word counts from %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)

    # Try building from existing clips
    if CLIPS_DIR.exists():
        counts = _count_existing_clips()
        if counts:
            _save_json(counts, cache_path)
            return counts

    # Download metadata
    metadata_gz = MSWC_DIR / "metadata" / "metadata.json.gz"
    metadata_gz.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_gz.exists():
        logger.info("Downloading MSWC metadata (103MB)...")
        _download_file(METADATA_URL, metadata_gz)

    logger.info("Parsing metadata for English...")
    with gzip.open(metadata_gz, "rt", encoding="utf-8") as f:
        metadata = json.load(f)

    en = metadata.get("en", {})
    wc = en.get("wordcounts", {})
    if not wc:
        raise RuntimeError(
            "Could not parse English word counts from metadata. "
            "File may be corrupted -- delete data/mswc_en/metadata/ and retry."
        )

    word_counts = {w: int(c) for w, c in wc.items()}
    logger.info("English: %d words, %s total clips",
                len(word_counts), f"{sum(word_counts.values()):,}")

    _save_json(word_counts, cache_path)
    return word_counts


def _count_existing_clips() -> dict[str, int]:
    counts = {}
    for d in CLIPS_DIR.iterdir():
        if d.is_dir():
            n = len(list(d.glob("*.opus")) + list(d.glob("*.wav")))
            if n > 0:
                counts[d.name] = n
    return counts


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ───────────────────── Word Splits ──────────────────────────────

def create_splits(
    word_counts: dict[str, int],
    n_train: int = 450,
    n_val: int = 50,
    min_eval_count: int = 1000,
) -> tuple[list[str], list[str], list[str]]:
    """Split words into train/val/eval pools.

    Pool 1: Top 500 by count -> n_train train + n_val val
    Pool 2: Rank 501+ with >= min_eval_count utterances -> eval
    """
    ranked = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    top500 = [w for w, _ in ranked[:500]]
    train_words = top500[:n_train]
    val_words = top500[n_train : n_train + n_val]
    eval_words = [w for w, c in ranked[500:] if c >= min_eval_count]

    logger.info("Splits: %d train, %d val, %d eval",
                len(train_words), len(val_words), len(eval_words))
    return train_words, val_words, eval_words


def save_splits(train: list[str], val: list[str], eval_: list[str]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, words in [("train_words", train), ("val_words", val), ("eval_words", eval_)]:
        path = SPLITS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(words, f, indent=2)
        logger.info("  %s: %d words -> %s", name, len(words), path)


def load_splits() -> tuple[list[str], list[str], list[str]] | None:
    paths = [SPLITS_DIR / f"{n}_words.json" for n in ("train", "val", "eval")]
    if all(p.exists() for p in paths):
        return tuple(json.loads(p.read_text()) for p in paths)
    return None


# ───────────────────── Download ─────────────────────────────────

def _download_file(url: str, dest: Path, desc: str | None = None) -> None:
    """Download a file with progress bar and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")

    # Resume support
    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

    resp = requests.get(url, stream=True, timeout=60, headers=headers)
    if resp.status_code == 416:
        # Already complete
        tmp.rename(dest)
        return
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + resume_pos
    mode = "ab" if resume_pos else "wb"

    with open(tmp, mode) as f, tqdm(
        total=total, initial=resume_pos,
        unit="B", unit_scale=True, desc=desc or dest.name,
    ) as pbar:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            pbar.update(len(chunk))

    tmp.rename(dest)
    logger.info("Downloaded: %s (%.1f GB)", dest, dest.stat().st_size / 1024**3)


def download_english_audio(mirror: str = "cloudflare") -> Path:
    """Download the MSWC English audio archive (32.5GB OPUS).

    Tries the specified mirror first, then falls back to others.

    Returns:
        Path to the downloaded archive.
    """
    archive = MSWC_DIR / "en.tar.gz"
    if archive.exists():
        logger.info("Archive already exists: %s (%.1f GB)",
                    archive, archive.stat().st_size / 1024**3)
        return archive

    # Try mirrors in order, starting with preferred
    mirror_order = [mirror] + [m for m in MIRRORS if m != mirror]

    for m in mirror_order:
        base = MIRRORS[m]
        url = f"{base}/audio/en.tar.gz"
        logger.info("Downloading MSWC English from %s mirror...", m)
        try:
            _download_file(url, archive, desc=f"en.tar.gz ({m})")
            return archive
        except Exception as e:
            logger.warning("Mirror %s failed: %s", m, e)
            archive.with_suffix(".tar.gz.partial").unlink(missing_ok=True)
            continue

    raise RuntimeError(
        "All mirrors failed. Download manually from:\n"
        "  https://mlcommons.org/datasets/multilingual-spoken-words/\n"
        f"Save as: {archive}"
    )


# ───────────────────── Extraction ───────────────────────────────

def extract_target_words(
    archive_path: Path,
    target_words: set[str],
    max_per_word: int = 200,
) -> dict[str, int]:
    """Extract only target words from the MSWC English archive.

    The archive structure is: en/clips/<word>/<filename>.opus
    We extract to: data/mswc_en/clips/<word>/<filename>.opus

    Args:
        archive_path: Path to en.tar.gz
        target_words: Set of words to extract.
        max_per_word: Max clips per word.

    Returns:
        Dict mapping word -> number of clips extracted.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing clips
    clip_counts: dict[str, int] = {}
    for w in target_words:
        d = CLIPS_DIR / w
        if d.exists():
            clip_counts[w] = len(list(d.glob("*.opus")) + list(d.glob("*.wav")))
        else:
            clip_counts[w] = 0

    words_needed = {w for w in target_words if clip_counts[w] < max_per_word}
    if not words_needed:
        logger.info("All %d words already have >= %d clips", len(target_words), max_per_word)
        return clip_counts

    logger.info("Extracting %d target words from archive (skipping %d complete)...",
                len(words_needed), len(target_words) - len(words_needed))

    extracted_total = 0
    skipped = 0

    try:
        with tarfile.open(str(archive_path), "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting target words"):
                if not member.isfile():
                    continue

                parts = Path(member.name).parts
                # Expected: en/clips/<word>/<file>.opus  or  clips/<word>/<file>.opus
                word = None
                for i, p in enumerate(parts):
                    if p == "clips" and i + 2 <= len(parts) - 1:
                        word = parts[i + 1]
                        break

                if word is None and len(parts) >= 2:
                    word = parts[-2]

                if word not in words_needed:
                    skipped += 1
                    continue

                if clip_counts.get(word, 0) >= max_per_word:
                    words_needed.discard(word)
                    continue

                dest_dir = CLIPS_DIR / word
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / Path(member.name).name

                if dest_file.exists():
                    clip_counts[word] = clip_counts.get(word, 0) + 1
                    continue

                f = tar.extractfile(member)
                if f is not None:
                    with open(dest_file, "wb") as out:
                        shutil.copyfileobj(f, out)
                    clip_counts[word] = clip_counts.get(word, 0) + 1
                    extracted_total += 1

    except Exception as e:
        logger.error("Extraction error: %s", e)

    logger.info("Extracted %d clips for %d words (skipped %d non-target files)",
                extracted_total,
                sum(1 for w in target_words if clip_counts.get(w, 0) > 0),
                skipped)
    return clip_counts


# ───────────────────── Main ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download MSWC English for training")
    parser.add_argument("--max-per-word", type=int, default=200,
                        help="Max clips per word (default: 200)")
    parser.add_argument("--n-train", type=int, default=450)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--splits-only", action="store_true",
                        help="Only download metadata and create splits")
    parser.add_argument("--from-archive", type=Path, default=None,
                        help="Extract from a local en.tar.gz instead of downloading")
    parser.add_argument("--mirror", choices=list(MIRRORS.keys()), default="cloudflare",
                        help="Preferred download mirror (default: cloudflare)")
    parser.add_argument("--keep-archive", action="store_true",
                        help="Keep the tar.gz after extraction (default: delete)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    MSWC_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Metadata + word counts
    logger.info("=" * 60)
    logger.info("Step 1: MSWC English word counts")
    logger.info("=" * 60)
    word_counts = load_word_counts()
    top5 = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top 5: %s", top5)

    # Step 2: Splits
    logger.info("=" * 60)
    logger.info("Step 2: Train/Val/Eval splits")
    logger.info("=" * 60)
    train_words, val_words, eval_words = create_splits(
        word_counts, n_train=args.n_train, n_val=args.n_val,
    )
    save_splits(train_words, val_words, eval_words)
    all_needed = set(train_words + val_words + eval_words)
    logger.info("Total words to extract: %d", len(all_needed))

    if args.splits_only:
        logger.info("Done (--splits-only). Run without flag to download audio.")
        return

    # Step 3: Download or locate archive
    logger.info("=" * 60)
    logger.info("Step 3: Audio download")
    logger.info("=" * 60)

    if args.from_archive:
        archive = args.from_archive
        if not archive.exists():
            logger.error("Archive not found: %s", archive)
            return
        logger.info("Using local archive: %s", archive)
    else:
        archive = download_english_audio(mirror=args.mirror)

    # Step 4: Extract target words
    logger.info("=" * 60)
    logger.info("Step 4: Extracting target words")
    logger.info("=" * 60)
    stats = extract_target_words(archive, all_needed, max_per_word=args.max_per_word)

    # Optionally delete archive
    if not args.keep_archive and not args.from_archive:
        logger.info("Deleting archive to save space...")
        archive.unlink(missing_ok=True)
        logger.info("Deleted: %s", archive)

    # Step 5: Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for name, words in [("Train", train_words), ("Val", val_words), ("Eval", eval_words)]:
        total = sum(stats.get(w, 0) for w in words)
        with_data = sum(1 for w in words if stats.get(w, 0) > 0)
        logger.info("  %s: %d words (%d with data, %d clips)",
                    name, len(words), with_data, total)

    opus_count = sum(1 for f in CLIPS_DIR.rglob("*.opus"))
    wav_count = sum(1 for f in CLIPS_DIR.rglob("*.wav"))
    logger.info("\nFiles: %d OPUS, %d WAV", opus_count, wav_count)
    logger.info("Clips directory: %s", CLIPS_DIR)

    if opus_count > 0:
        logger.info(
            "\nNext steps:\n"
            "  1. Convert OPUS -> WAV:  python data/convert_opus.py\n"
            "  2. Train:                python scripts/train.py --config configs/default.yaml"
        )
    elif wav_count > 0:
        logger.info("\nReady for training!\n"
                    "  python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
