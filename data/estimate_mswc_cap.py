"""Estimate an MSWC per-word cap for a target number of files.

This script expects MSWC metadata and train/val word split files to already
exist. On Colab, run ``data/download_mswc.py --splits-only`` first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _load_words(path: Path) -> list[str]:
    obj = _load_json(path)
    if isinstance(obj, dict):
        if "words" in obj:
            obj = obj["words"]
        else:
            obj = list(obj.keys())
    return [str(x) for x in obj]


def _estimate_for_cap(
    word_counts: dict[str, int],
    train_words: list[str],
    val_words: list[str],
    cap: int,
) -> dict[str, int]:
    train = sum(min(int(word_counts.get(w, 0)), cap) for w in train_words)
    val = sum(min(int(word_counts.get(w, 0)), cap) for w in val_words)
    return {
        "cap": cap,
        "train_files": train,
        "val_files": val,
        "total_files": train + val,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mswc_en"))
    parser.add_argument("--target-files", type=int, default=6_000_000)
    parser.add_argument("--min-cap", type=int, default=180)
    parser.add_argument("--max-cap", type=int, default=220)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-selected-only", action="store_true")
    args = parser.parse_args()

    if args.min_cap <= 0 or args.max_cap <= 0:
        raise SystemExit("caps must be positive")
    if args.min_cap > args.max_cap:
        raise SystemExit("--min-cap must be <= --max-cap")
    if args.step <= 0:
        raise SystemExit("--step must be positive")

    metadata_path = args.data_dir / "metadata" / "en_word_counts.json"
    train_words_path = args.data_dir / "splits" / "train_words.json"
    val_words_path = args.data_dir / "splits" / "val_words.json"

    missing = [p for p in (metadata_path, train_words_path, val_words_path) if not p.exists()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise SystemExit(
            "missing required files: "
            f"{joined}. Run: python data/download_mswc.py --splits-only"
        )

    word_counts = {str(k): int(v) for k, v in _load_json(metadata_path).items()}
    train_words = _load_words(train_words_path)
    val_words = _load_words(val_words_path)

    caps = list(range(args.min_cap, args.max_cap + 1, args.step))
    if caps[-1] != args.max_cap:
        caps.append(args.max_cap)

    estimates = [
        _estimate_for_cap(word_counts, train_words, val_words, cap)
        for cap in caps
    ]
    selected = next(
        (row for row in estimates if row["total_files"] >= args.target_files),
        estimates[-1],
    )
    hit_target = selected["total_files"] >= args.target_files

    report = {
        "target_files": args.target_files,
        "selected_cap": selected["cap"],
        "selected_train_files": selected["train_files"],
        "selected_val_files": selected["val_files"],
        "selected_total_files": selected["total_files"],
        "hit_target": hit_target,
        "train_words": len(train_words),
        "val_words": len(val_words),
        "total_words": len(train_words) + len(val_words),
        "estimates": estimates,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if args.print_selected_only:
        print(report["selected_cap"])
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
