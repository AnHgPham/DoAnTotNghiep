"""Build publication-ready result tables from GSC evaluation JSON files.

The script is intentionally small and file-based so it can be run in Colab or
locally after copying result JSON folders from Drive.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METRIC_FIELDS = [
    "open_set_acc_at_1far",
    "open_set_acc_at_5far",
    "frr_at_5far",
    "auc",
    "eer",
    "keyword_acc",
    "f1",
]

HEADERS = [
    "Model",
    "Split",
    "Runs",
    "ACC@1% FAR",
    "ACC@5% FAR",
    "FRR@5% FAR",
    "AUC",
    "EER",
    "Keyword ACC",
    "F1",
]


@dataclass(frozen=True)
class ResultRow:
    label: str
    split: str
    runs: int | None
    metrics: dict[str, float | None]
    source: str


def _get_metric(data: dict[str, Any], key: str) -> float | None:
    aliases = {
        "open_set_acc_at_5far": ["open_set_acc_at_5far", "open_set_acc_at_far"],
        "frr_at_5far": ["frr_at_5far", "frr_at_far"],
    }
    for candidate in aliases.get(key, [key]):
        value = data.get(candidate)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _infer_label(folder_name: str) -> str:
    name = folder_name.lower()
    if "dscnn" in name:
        return "DSCNN-L Triplet"
    if "edgespot_full" in name and "scaf_ge2e" in name:
        return "EdgeSpotFull T4 SCAF+GE2E"
    if "edgespot_full" in name and "scaf" in name:
        return "EdgeSpotFull T4 SCAF"
    return folder_name.replace("_", " ")


def _infer_split(folder_name: str, data: dict[str, Any]) -> str:
    name = folder_name.lower()
    if "test" in name:
        return "test"
    if "dev" in name or "val" in name:
        return "dev"
    return str(data.get("split", "unknown"))


def _infer_runs(folder_name: str, data: dict[str, Any]) -> int | None:
    for pattern in (r"(?:test|dev|val)(\d+)", r"runs?[_-]?(\d+)"):
        match = re.search(pattern, folder_name.lower())
        if match:
            return int(match.group(1))
    if isinstance(data.get("per_run"), list):
        return len(data["per_run"])
    value = data.get("n_runs")
    return int(value) if isinstance(value, int) else None


def _row_sort_key(row: ResultRow) -> tuple[int, int, str]:
    label_order = {
        "DSCNN-L Triplet": 0,
        "EdgeSpotFull T4 SCAF": 1,
        "EdgeSpotFull T4 SCAF+GE2E": 2,
    }
    split_order = {"dev": 0, "test": 1}
    return (
        label_order.get(row.label, 99),
        split_order.get(row.split, 99),
        row.source,
    )


def row_from_json(path: Path) -> ResultRow:
    data = json.loads(path.read_text(encoding="utf-8"))
    folder = path.parent.name
    return ResultRow(
        label=_infer_label(folder),
        split=_infer_split(folder, data),
        runs=_infer_runs(folder, data),
        metrics={key: _get_metric(data, key) for key in METRIC_FIELDS},
        source=str(path),
    )


def rows_from_manifest(path: Path) -> list[ResultRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("experiments", []):
        metrics = {
            key: (float(item[key]) if item.get(key) is not None else None)
            for key in METRIC_FIELDS
        }
        rows.append(
            ResultRow(
                label=str(item["label"]),
                split=str(item.get("split", "unknown")),
                runs=int(item["runs"]) if item.get("runs") is not None else None,
                metrics=metrics,
                source=str(item.get("result_json", path)),
            )
        )
    return rows


def discover_json(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("gsc_edgespot_exact_k10_results.json"))


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _raw(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def markdown_table(rows: Iterable[ResultRow]) -> str:
    lines = [
        "| " + " | ".join(HEADERS) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row.metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    row.label,
                    row.split,
                    str(row.runs) if row.runs is not None else "-",
                    _pct(m["open_set_acc_at_1far"]),
                    _pct(m["open_set_acc_at_5far"]),
                    _pct(m["frr_at_5far"]),
                    _pct(m["auc"]),
                    _pct(m["eer"]),
                    _pct(m["keyword_acc"]),
                    _pct(m["f1"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def latex_table(rows: Iterable[ResultRow]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Split & Runs & ACC@1\%FAR & ACC@5\%FAR & FRR@5\%FAR & AUC & EER & KW-ACC & F1 \\",
        r"\midrule",
    ]
    for row in rows:
        m = row.metrics
        lines.append(
            " & ".join(
                [
                    row.label,
                    row.split,
                    str(row.runs) if row.runs is not None else "-",
                    _pct(m["open_set_acc_at_1far"]).replace("%", r"\%"),
                    _pct(m["open_set_acc_at_5far"]).replace("%", r"\%"),
                    _pct(m["frr_at_5far"]).replace("%", r"\%"),
                    _pct(m["auc"]).replace("%", r"\%"),
                    _pct(m["eer"]).replace("%", r"\%"),
                    _pct(m["keyword_acc"]).replace("%", r"\%"),
                    _pct(m["f1"]).replace("%", r"\%"),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_csv(rows: Iterable[ResultRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "split",
                "runs",
                "open_set_acc_at_1far",
                "open_set_acc_at_5far",
                "frr_at_5far",
                "auc",
                "eer",
                "keyword_acc",
                "f1",
                "source",
            ]
        )
        for row in rows:
            m = row.metrics
            writer.writerow(
                [
                    row.label,
                    row.split,
                    row.runs if row.runs is not None else "",
                    _raw(m["open_set_acc_at_1far"]),
                    _raw(m["open_set_acc_at_5far"]),
                    _raw(m["frr_at_5far"]),
                    _raw(m["auc"]),
                    _raw(m["eer"]),
                    _raw(m["keyword_acc"]),
                    _raw(m["f1"]),
                    row.source,
                ]
            )


def build_rows(paths: list[Path], results_dir: Path | None, manifest: Path | None) -> list[ResultRow]:
    json_paths = list(paths)
    if results_dir is not None and results_dir.exists():
        json_paths.extend(discover_json(results_dir))

    rows = [row_from_json(path) for path in sorted(set(json_paths))]
    if not rows and manifest is not None and manifest.exists():
        rows = rows_from_manifest(manifest)

    return sorted(rows, key=_row_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Markdown/CSV/LaTeX result tables")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional result JSON files")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/microset"))
    parser.add_argument("--profile", default="microset_en")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = args.manifest
    if manifest is None and args.profile == "microset_en":
        manifest = Path("reports/microset/locked_results_manifest.json")

    rows = build_rows(args.paths, args.results_dir, manifest)
    if not rows:
        raise SystemExit("No result JSON files found and no manifest fallback available.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "result_table.md"
    csv_path = args.out_dir / "result_table.csv"
    tex_path = args.out_dir / "result_table.tex"

    md_path.write_text(markdown_table(rows), encoding="utf-8")
    write_csv(rows, csv_path)
    tex_path.write_text(latex_table(rows), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
