"""Create a template table for the GSC 17/17 open-set demo.

The live values are produced by the UI/API at demo time. This script writes the
stable split and reporting policy so the thesis/report text stays consistent.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "reports" / "project_status" / "open_set_17_17_table.md"

KNOWN = [
    "yes", "stop", "happy", "bird", "dog", "tree", "marvin", "four", "learn",
    "wow", "sheila", "zero", "down", "left", "right", "off", "three",
]
UNKNOWN = [
    "no", "go", "up", "on", "one", "two", "five", "six", "seven", "eight",
    "nine", "bed", "cat", "house", "backward", "forward", "follow",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join([
            "# Open-Set 17/17 Demo Table",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Known words | {', '.join(KNOWN)} |",
            f"| Unknown words | {', '.join(UNKNOWN)} |",
            "| Heldout | visual |",
            "| Candidate labels | 17 known words only |",
            "| Recommended policy | Guard ON, Per-class OFF, accept margin 0.05 |",
            "| Reporting status | Demo-level sampled evaluation, not a replacement for GSC test100 |",
            "",
            "## Metrics To Fill From UI",
            "",
            "| Run | Balanced | KW-ACC | Unknown Reject ACC | FAR | False Reject | Threshold | Guard | Per-class | Margin |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
            "| Best Balanced | TBD | TBD | TBD | TBD | TBD | TBD | ON | OFF | 0.05 |",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
