"""Download MSWC English subset for training and evaluation.

Downloads top 500 words (450 train + 50 val) + 263 eval words = 763 total.
MSWC files are in OPUS format at 48kHz -- use convert_opus.py afterward.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MSWC_DIR = Path("data/mswc_en")
MSWC_BASE_URL = "https://storage.googleapis.com/mswc-corpus/en"


def get_word_list() -> list[dict]:
    """Fetch and parse the MSWC English word list with utterance counts.

    Returns:
        List of dicts with 'word' and 'count' keys, sorted by count descending.
    """
    # Placeholder: actual implementation depends on MSWC metadata format.
    # The real MSWC provides a TSV or JSON with word -> utterance_count.
    raise NotImplementedError(
        "MSWC word list fetching requires the actual MSWC metadata. "
        "Download the English metadata from MLCommons and place it in "
        f"{MSWC_DIR}/metadata/. Then update this function."
    )


def filter_words(
    all_words: list[dict],
    n_train: int = 450,
    n_val: int = 50,
    n_eval_min_utterances: int = 1000,
) -> tuple[list[str], list[str], list[str]]:
    """Split words into train, validation, and evaluation pools.

    Args:
        all_words: Word list sorted by utterance count descending.
        n_train: Number of words for training.
        n_val: Number of words for validation.
        n_eval_min_utterances: Minimum utterances for eval words.

    Returns:
        (train_words, val_words, eval_words) lists of word strings.
    """
    top500 = all_words[:500]
    train_words = [w["word"] for w in top500[:n_train]]
    val_words = [w["word"] for w in top500[n_train : n_train + n_val]]

    remaining = all_words[500:]
    eval_words = [
        w["word"] for w in remaining if w["count"] >= n_eval_min_utterances
    ]

    logger.info(
        "Word split: %d train, %d val, %d eval",
        len(train_words),
        len(val_words),
        len(eval_words),
    )
    return train_words, val_words, eval_words


def download_word_clips(word: str, dest_dir: Path) -> int:
    """Download all audio clips for a given word.

    Returns:
        Number of clips downloaded.
    """
    word_dir = dest_dir / "clips" / word
    word_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: actual download logic depends on MSWC hosting structure.
    # Each word has clips stored as OPUS files.
    logger.info("TODO: Download clips for word '%s' to %s", word, word_dir)
    return 0


def save_splits(
    train_words: list[str],
    val_words: list[str],
    eval_words: list[str],
    dest_dir: Path,
) -> None:
    """Save word splits to JSON for reproducibility."""
    splits_dir = dest_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, words in [
        ("train", train_words),
        ("val", val_words),
        ("eval", eval_words),
    ]:
        path = splits_dir / f"{name}_words.json"
        with open(path, "w") as f:
            json.dump(words, f, indent=2)
        logger.info("Saved %d words to %s", len(words), path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    MSWC_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("MSWC download script initialized.")
    logger.info("Target directory: %s", MSWC_DIR)
    logger.info(
        "NOTE: This script requires MSWC metadata from MLCommons. "
        "Download English metadata first, then update get_word_list()."
    )


if __name__ == "__main__":
    main()
