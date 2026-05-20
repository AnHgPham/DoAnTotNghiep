"""Aggregate confusion and per-word error tables from evaluation result JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COUNT_FIELDS = [
    "total",
    "correct",
    "rejected",
    "confused",
    "false_accept",
    "correct_reject",
]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _safe_rate(num: int, den: int) -> float | None:
    return None if den <= 0 else num / den


def aggregate_per_word(results: dict[str, Any]) -> dict[str, dict[str, int | float | None]]:
    totals: dict[str, Counter] = defaultdict(Counter)
    for run in results.get("per_run", []):
        for word, counts in run.get("per_word", {}).items():
            for field in COUNT_FIELDS:
                totals[word][field] += int(counts.get(field, 0) or 0)

    rows = {}
    for word, counts in totals.items():
        known_total = counts["correct"] + counts["rejected"] + counts["confused"]
        unknown_total = counts["false_accept"] + counts["correct_reject"]
        rows[word] = {
            **{field: int(counts[field]) for field in COUNT_FIELDS},
            "keyword_recall_at_far": _safe_rate(int(counts["correct"]), int(known_total)),
            "false_accept_rate": _safe_rate(int(counts["false_accept"]), int(unknown_total)),
            "accuracy_at_far": _safe_rate(
                int(counts["correct"] + counts["correct_reject"]),
                int(counts["total"]),
            ),
        }
    return rows


def aggregate_confusion(results: dict[str, Any]) -> Counter[tuple[str, str]]:
    confusion: Counter[tuple[str, str]] = Counter()
    for run in results.get("per_run", []):
        for true_label, pred_counts in run.get("confusion", {}).items():
            for pred_label, count in pred_counts.items():
                if true_label != pred_label:
                    confusion[(true_label, pred_label)] += int(count)
    return confusion


def write_per_word_csv(rows: dict[str, dict[str, int | float | None]], path: Path) -> None:
    fieldnames = [
        "word",
        *COUNT_FIELDS,
        "keyword_recall_at_far",
        "false_accept_rate",
        "accuracy_at_far",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for word, row in sorted(rows.items()):
            writer.writerow({"word": word, **row})


def write_markdown(
    rows: dict[str, dict[str, int | float | None]],
    confusion: Counter[tuple[str, str]],
    path: Path,
    top_k: int,
) -> None:
    lines = ["# Result Error Analysis", ""]
    lines.append("## Per-word summary")
    lines.append("")
    lines.append("| Word | Total | Recall@FAR | False Accept Rate | Accuracy@FAR | Rejected | Confused |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    sorted_rows = sorted(
        rows.items(),
        key=lambda item: (
            item[1]["accuracy_at_far"] is None,
            item[1]["accuracy_at_far"] if item[1]["accuracy_at_far"] is not None else 1.0,
            item[0],
        ),
    )
    for word, row in sorted_rows:
        lines.append(
            f"| `{word}` | {row['total']} | {_pct(row['keyword_recall_at_far'])} | "
            f"{_pct(row['false_accept_rate'])} | {_pct(row['accuracy_at_far'])} | "
            f"{row['rejected']} | {row['confused']} |"
        )

    lines.extend(["", f"## Top {top_k} confusion pairs", ""])
    lines.append("| True | Predicted | Count |")
    lines.append("|---|---|---:|")
    for (true_label, pred_label), count in confusion.most_common(top_k):
        lines.append(f"| `{true_label}` | `{pred_label}` | {count} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-word and confusion errors")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/microset"))
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    results = json.loads(args.result_json.read_text(encoding="utf-8"))
    if not results.get("per_run"):
        raise SystemExit("Result JSON has no per_run data; re-run evaluation with current protocol output.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_word = aggregate_per_word(results)
    confusion = aggregate_confusion(results)
    write_per_word_csv(per_word, args.out_dir / "per_word_errors.csv")
    write_markdown(per_word, confusion, args.out_dir / "error_analysis.md", args.top_k)
    print(f"Wrote {args.out_dir / 'per_word_errors.csv'}")
    print(f"Wrote {args.out_dir / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
