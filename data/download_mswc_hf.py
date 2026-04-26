"""Download MSWC English from HuggingFace (WAV 16kHz).

Downloads tar.gz shards one by one, extracts only target words,
then discards the shard. Audio is WAV@16kHz -- no conversion needed.

This avoids needing the full 55GB dataset: only the ~763 target words
are kept (max 200 clips each ≈ 5GB).

Usage:
    python data/download_mswc_hf.py
    python data/download_mswc_hf.py --max-per-word 300
    python data/download_mswc_hf.py --splits-only
"""

import argparse
import gzip
import json
import logging
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm

logger = logging.getLogger(__name__)

MSWC_DIR = Path("data/mswc_en")
SPLITS_DIR = MSWC_DIR / "splits"
CLIPS_DIR = MSWC_DIR / "clips"

HF_REPO = "MLCommons/ml_spoken_words"
METADATA_URL = "https://mswc.mlcommons-storage.org/metadata.json.gz"
AUDIO_URLS = [
    "https://mswc.mlcommons-storage.org/audio/en.tar.gz",       # Cloudflare
    "https://mlc-datasets.oss-cn-guangzhou.aliyuncs.com/audio/en.tar.gz",  # Alibaba
    "https://storage.googleapis.com/public-datasets-mswc/audio/en.tar.gz", # Google
]


# ───────────────────── Metadata ─────────────────────────────────

def download_metadata() -> dict[str, int]:
    """Download MSWC metadata to get English word counts.

    Tries local cache first, then MLCommons Cloudflare mirror.

    Returns:
        Dict mapping word -> utterance count for English.
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

    # Download metadata from MLCommons
    logger.info("Downloading MSWC metadata (103MB)...")
    metadata_gz_path = MSWC_DIR / "metadata" / "metadata.json.gz"
    metadata_gz_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_gz_path.exists():
        try:
            resp = requests.get(METADATA_URL, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(metadata_gz_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="metadata.json.gz",
            ) as pbar:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except Exception as e:
            logger.error("Failed to download metadata: %s", e)
            logger.info("Falling back to scanning HuggingFace repo...")
            return _fallback_word_counts()

    # Parse English section
    logger.info("Parsing metadata for English...")
    with gzip.open(metadata_gz_path, "rt", encoding="utf-8") as f:
        metadata = json.load(f)

    en_meta = metadata.get("en", {})
    word_counts_raw = en_meta.get("wordcounts", en_meta.get("word_counts", {}))

    if not word_counts_raw:
        logger.warning("No English word counts in metadata, trying 'filenames' key...")
        filenames = en_meta.get("filenames", {})
        if filenames:
            word_counts_raw = {w: len(flist) for w, flist in filenames.items()}

    if not word_counts_raw:
        logger.error("Could not parse English word counts from metadata")
        return _fallback_word_counts()

    word_counts = {w: int(c) for w, c in word_counts_raw.items()}
    logger.info("English: %d unique words, %s total clips",
                len(word_counts), f"{sum(word_counts.values()):,}")

    _save_json(word_counts, cache_path)
    return word_counts


def _count_existing_clips() -> dict[str, int]:
    """Count WAV files in existing clips directory."""
    counts = {}
    for word_dir in CLIPS_DIR.iterdir():
        if word_dir.is_dir():
            n = len(list(word_dir.glob("*.wav")))
            if n > 0:
                counts[word_dir.name] = n
    return counts


def _fallback_word_counts() -> dict[str, int]:
    """Fallback: use built-in word list from download_mswc.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_mswc", Path(__file__).parent / "download_mswc.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    words = mod._get_builtin_word_list()
    return {w["word"]: w["count"] for w in words}


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved -> %s", path)


# ───────────────────── Word Splits ──────────────────────────────

def create_splits(
    word_counts: dict[str, int],
    n_train: int = 450,
    n_val: int = 50,
    min_eval_count: int = 1000,
) -> tuple[list[str], list[str], list[str]]:
    """Split words into train/val/eval pools."""
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    top500 = [w for w, _ in sorted_words[:500]]
    train_words = top500[:n_train]
    val_words = top500[n_train : n_train + n_val]

    eval_words = [
        w for w, c in sorted_words[500:] if c >= min_eval_count
    ]

    logger.info(
        "Splits: %d train, %d val, %d eval (from %d total words)",
        len(train_words), len(val_words), len(eval_words), len(sorted_words),
    )
    return train_words, val_words, eval_words


def save_splits(train: list[str], val: list[str], eval_: list[str]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, words in [("train_words", train), ("val_words", val), ("eval_words", eval_)]:
        path = SPLITS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(words, f, indent=2)
        logger.info("  %s: %d words -> %s", name, len(words), path)


# ───────────────────── Audio Download ───────────────────────────

def _extract_keyword(filename: str) -> str | None:
    """Extract keyword from MSWC filename like 'hello_common_voice_en_12345.wav'."""
    marker = "_common_voice_"
    idx = filename.find(marker)
    if idx > 0:
        return filename[:idx]
    # Fallback: assume keyword is the parent directory name
    return None


def _list_hf_shards(split: str = "train", fmt: str = "wav") -> list[str]:
    """List tar.gz shard paths in HuggingFace repo for a given split."""
    api = HfApi()
    path = f"data/{fmt}/en/{split}/audio"
    files = api.list_repo_tree(HF_REPO, repo_type="dataset", path_in_repo=path)
    return sorted(
        [f.path for f in files if hasattr(f, 'size') and f.path.endswith(".tar.gz")],
    )


def download_audio_from_hf(
    target_words: set[str],
    max_per_word: int = 200,
    fmt: str = "wav",
    splits: list[str] | None = None,
    delete_after_extract: bool = True,
) -> dict[str, int]:
    """Download audio by streaming through HuggingFace tar.gz shards.

    For each shard:
      1. Download via huggingface_hub (cached/resumable)
      2. Stream through tar, extracting only target word clips
      3. Optionally delete shard from cache to save space

    Args:
        target_words: Words to download audio for.
        max_per_word: Maximum clips per word.
        fmt: Audio format ('wav' or 'opus').
        splits: HF splits to process (default: train, dev, test).
        delete_after_extract: Remove downloaded shards after extraction.

    Returns:
        Dict mapping word -> clip count.
    """
    if splits is None:
        splits = ["train", "dev", "test"]

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Track existing clips
    clip_counts: dict[str, int] = defaultdict(int)
    for word in target_words:
        word_dir = CLIPS_DIR / word
        if word_dir.exists():
            clip_counts[word] = len(list(word_dir.glob("*.wav")))

    words_done = {w for w in target_words if clip_counts[w] >= max_per_word}
    words_needed = target_words - words_done

    if not words_needed:
        logger.info("All %d words already have >= %d clips!", len(target_words), max_per_word)
        return dict(clip_counts)

    logger.info(
        "Need clips for %d words (%d already complete with >= %d clips)",
        len(words_needed), len(words_done), max_per_word,
    )

    for split in splits:
        if not words_needed:
            break

        logger.info("Listing shards for split '%s'...", split)
        try:
            shard_paths = _list_hf_shards(split, fmt)
        except Exception as e:
            logger.warning("Failed to list shards for %s: %s", split, e)
            continue

        logger.info("Found %d shards for split '%s'", len(shard_paths), split)

        for shard_idx, shard_path in enumerate(shard_paths):
            if not words_needed:
                logger.info("All words complete! Stopping early.")
                break

            logger.info(
                "[%s %d/%d] Downloading %s ...",
                split, shard_idx + 1, len(shard_paths), shard_path.split("/")[-1],
            )

            try:
                local_path = hf_hub_download(
                    HF_REPO,
                    shard_path,
                    repo_type="dataset",
                )
            except Exception as e:
                logger.warning("Failed to download shard %s: %s", shard_path, e)
                continue

            # Extract target words from this shard
            n_extracted = _extract_from_tar(
                local_path, target_words, words_needed, clip_counts, max_per_word,
            )

            # Update words_needed
            newly_done = {w for w in words_needed if clip_counts[w] >= max_per_word}
            words_needed -= newly_done
            words_done |= newly_done

            logger.info(
                "  Extracted %d clips. Progress: %d/%d words complete.",
                n_extracted, len(words_done), len(target_words),
            )

            if delete_after_extract:
                try:
                    Path(local_path).unlink(missing_ok=True)
                except Exception:
                    pass

    return dict(clip_counts)


def _extract_from_tar(
    tar_path: str,
    target_words: set[str],
    words_needed: set[str],
    clip_counts: dict[str, int],
    max_per_word: int,
) -> int:
    """Extract target word clips from a single tar.gz shard."""
    n_extracted = 0
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                filename = Path(member.name).name
                keyword = _extract_keyword(filename)

                if keyword is None:
                    # Try parent directory name
                    parts = Path(member.name).parts
                    if len(parts) >= 2:
                        keyword = parts[-2]

                if keyword not in words_needed:
                    continue

                if clip_counts[keyword] >= max_per_word:
                    continue

                word_dir = CLIPS_DIR / keyword
                word_dir.mkdir(parents=True, exist_ok=True)

                dest = word_dir / filename
                if dest.exists():
                    clip_counts[keyword] += 1
                    continue

                try:
                    f = tar.extractfile(member)
                    if f is not None:
                        with open(dest, "wb") as out:
                            shutil.copyfileobj(f, out)
                        clip_counts[keyword] += 1
                        n_extracted += 1
                except Exception as e:
                    logger.debug("Failed to extract %s: %s", member.name, e)

    except Exception as e:
        logger.error("Failed to read tar %s: %s", tar_path, e)

    return n_extracted


# ───────────────────── Main ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MSWC English from HuggingFace (WAV 16kHz)"
    )
    parser.add_argument("--max-per-word", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=450)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--splits-only", action="store_true",
                        help="Only generate word splits, skip audio download")
    parser.add_argument("--keep-shards", action="store_true",
                        help="Keep downloaded tar.gz shards (default: delete after extraction)")
    parser.add_argument("--format", choices=["wav", "opus"], default="wav",
                        help="Audio format to download (default: wav, no conversion needed)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    MSWC_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get word counts
    logger.info("=" * 60)
    logger.info("Step 1: Getting MSWC English word counts")
    logger.info("=" * 60)
    word_counts = download_metadata()
    logger.info("Total English words: %d", len(word_counts))
    top10 = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 words: %s", top10)

    # Step 2: Create splits
    logger.info("=" * 60)
    logger.info("Step 2: Creating train/val/eval splits")
    logger.info("=" * 60)
    train_words, val_words, eval_words = create_splits(
        word_counts, n_train=args.n_train, n_val=args.n_val,
    )
    save_splits(train_words, val_words, eval_words)

    if args.splits_only:
        logger.info("Skipping audio download (--splits-only)")
        return

    # Step 3: Download audio
    logger.info("=" * 60)
    logger.info("Step 3: Downloading audio from HuggingFace")
    logger.info("=" * 60)

    all_needed = set(train_words + val_words + eval_words)
    logger.info("Target: %d words, max %d clips each", len(all_needed), args.max_per_word)

    stats = download_audio_from_hf(
        target_words=all_needed,
        max_per_word=args.max_per_word,
        fmt=args.format,
        delete_after_extract=not args.keep_shards,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for name, words in [("Train", train_words), ("Val", val_words), ("Eval", eval_words)]:
        total_clips = sum(stats.get(w, 0) for w in words)
        with_data = sum(1 for w in words if stats.get(w, 0) > 0)
        logger.info("  %s: %d words (%d with data, %d clips)", name, len(words), with_data, total_clips)

    logger.info("\nSplits: %s", SPLITS_DIR)
    logger.info("Clips:  %s", CLIPS_DIR)

    if args.format == "wav":
        logger.info("\nAudio is WAV@16kHz -- NO conversion needed!")
    else:
        logger.info("\nAudio is OPUS -- run: python data/convert_opus.py")

    logger.info("Next: python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
