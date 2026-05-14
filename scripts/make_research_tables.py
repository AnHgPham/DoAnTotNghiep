"""Create compact Markdown tables from evaluation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    name = path.stem.replace("_results", "")
    return (
        f"| {name} | {data.get('auc', 0):.4f} | {data.get('eer', 0):.4f} | "
        f"{data.get('open_set_acc_at_1far', 0):.4f} | "
        f"{data.get('open_set_acc_at_5far', data.get('open_set_acc_at_far', 0)):.4f} | "
        f"{data.get('frr_at_5far', data.get('frr_at_far', 0)):.4f} | "
        f"{data.get('keyword_acc', 0):.4f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build research metric table")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    print("| Experiment | AUC | EER | ACC@1% FAR | ACC@5% FAR | FRR@5% FAR | KW-ACC |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for path in args.paths:
        print(row(path))


if __name__ == "__main__":
    main()
