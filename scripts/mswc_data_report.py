"""Report MSWC extraction scale for Colab/research runs.

This script is intentionally lightweight: it checks the word split JSON files,
counts extracted audio per word, and optionally compares against MSWC metadata
word counts when available. Use it before long training jobs to avoid
accidentally benchmarking a capped debug cache such as ``mpw200``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_words(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return [str(w) for w in payload]


def _count_audio(word_dir: Path) -> int:
    if not word_dir.is_dir():
        return 0
    return sum(1 for _ in word_dir.glob("*.wav")) + sum(1 for _ in word_dir.glob("*.opus"))


def _load_metadata_counts(data_dir: Path) -> dict[str, int]:
    path = data_dir / "metadata" / "en_word_counts.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(k): int(v) for k, v in payload.items()}


def _print_split_report(
    name: str,
    words: list[str],
    clips_dir: Path,
    metadata_counts: dict[str, int],
    top_n: int,
) -> None:
    counts = {word: _count_audio(clips_dir / word) for word in words}
    present = {word: count for word, count in counts.items() if count > 0}
    total = sum(counts.values())
    avg = total / len(words) if words else 0.0
    nonzero_avg = total / len(present) if present else 0.0

    print(f"\n[{name}]")
    print(f"words: {len(words)}")
    print(f"words_with_audio: {len(present)}")
    print(f"audio_files: {total:,}")
    print(f"avg_files_per_word: {avg:,.1f}")
    print(f"avg_files_per_audio_word: {nonzero_avg:,.1f}")
    if present:
        values = list(present.values())
        print(f"min/max_audio_word: {min(values):,} / {max(values):,}")
        count_hist: dict[int, int] = {}
        for value in values:
            count_hist[value] = count_hist.get(value, 0) + 1
        most_common_count, most_common_n = max(
            count_hist.items(),
            key=lambda item: item[1],
        )
        print(
            "most_common_word_count: "
            f"{most_common_count:,} files on {most_common_n}/{len(values)} words",
        )

    if metadata_counts and words:
        expected = {word: metadata_counts.get(word, 0) for word in words}
        expected_total = sum(expected.values())
        coverage = total / expected_total if expected_total else 0.0
        print(f"metadata_expected_files: {expected_total:,}")
        print(f"actual_vs_metadata_coverage: {coverage:.1%}")

    capped_words = sum(1 for count in present.values() if 0 < count <= 200)
    if words and capped_words / max(len(words), 1) > 0.80:
        print("WARNING: most words have <=200 files. This looks like an mpw200 debug cache, not full extraction.")
    if present and most_common_n / max(len(present), 1) > 0.80 and most_common_count <= 1000:
        print(
            "WARNING: most words have the exact same low file count. "
            "This looks like a capped extraction such as mpw500/mpw1000, not full extraction.",
        )

    ranked = sorted(present.items(), key=lambda item: item[1], reverse=True)
    if ranked:
        print(f"top_{min(top_n, len(ranked))}:")
        for word, count in ranked[:top_n]:
            expected = metadata_counts.get(word)
            suffix = f" / metadata={expected:,}" if expected is not None else ""
            print(f"  {word}: {count:,}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report MSWC extracted audio scale")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mswc_en"))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    data_dir = args.data_dir
    clips_dir = data_dir / "clips"
    splits_dir = data_dir / "splits"

    if not clips_dir.exists():
        raise SystemExit(f"Missing clips directory: {clips_dir}")

    train_words = _load_words(splits_dir / "train_words.json")
    val_words = _load_words(splits_dir / "val_words.json")
    eval_words = _load_words(splits_dir / "eval_words.json")
    metadata_counts = _load_metadata_counts(data_dir)

    print(f"data_dir: {data_dir}")
    print(f"clips_dir: {clips_dir}")
    print(f"metadata_counts: {'yes' if metadata_counts else 'no'}")

    _print_split_report("train", train_words, clips_dir, metadata_counts, args.top_n)
    _print_split_report("val", val_words, clips_dir, metadata_counts, args.top_n)
    if eval_words:
        _print_split_report("eval", eval_words, clips_dir, metadata_counts, args.top_n)
    _print_split_report("train+val", train_words + val_words, clips_dir, metadata_counts, args.top_n)


if __name__ == "__main__":
    main()
