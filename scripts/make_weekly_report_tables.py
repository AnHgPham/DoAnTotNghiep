"""Write stable Markdown result tables used by weekly reports and thesis drafts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.artifacts import discover_artifacts, format_percent


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def record_by_id(status: dict, record_id: str) -> dict:
    for record in status.get("records", []):
        if record.get("id") == record_id:
            return record
    return {}


def main() -> None:
    status = discover_artifacts(PROJECT_ROOT)
    out = PROJECT_ROOT / "reports" / "project_status"

    micro = record_by_id(status, "microset_edgespot_t4_scaf_ge2e_epoch05")
    top = record_by_id(status, "top500_edgespot_t4_scaf_ge2e_epoch13")

    micro_metrics = micro.get("metrics", {})
    top_metrics = top.get("metrics", {})

    write(
        out / "microset_table.md",
        "\n".join([
            "# Microset Result Table",
            "",
            "| Model | Status | ACC@5%FAR | KW-ACC | F1 | AUC | EER |",
            "|---|---|---:|---:|---:|---:|---:|",
            "| EdgeSpotFull T4 + SCAF+GE2E epoch05 | official locked | {acc5} | {kw} | {f1} | {auc} | {eer} |".format(
                acc5=format_percent(micro_metrics.get("acc_at_5far")),
                kw=format_percent(micro_metrics.get("keyword_acc")),
                f1=format_percent(micro_metrics.get("f1")),
                auc=format_percent(micro_metrics.get("auc")),
                eer=format_percent(micro_metrics.get("eer")),
            ),
            "",
        ]),
    )

    write(
        out / "top500_epoch13_table.md",
        "\n".join([
            "# Top500 Epoch13 Dev30 Table",
            "",
            "| Model | Status | ACC@1%FAR | ACC@5%FAR | FRR@5%FAR | AUC | F1 |",
            "|---|---|---:|---:|---:|---:|---:|",
            "| EdgeSpotFull T4 + SCAF+GE2E epoch13 | local preliminary | {acc1} | {acc5} | {frr5} | {auc} | {f1} |".format(
                acc1=format_percent(top_metrics.get("acc_at_1far")),
                acc5=format_percent(top_metrics.get("acc_at_5far")),
                frr5=format_percent(top_metrics.get("frr_at_5far")),
                auc=format_percent(top_metrics.get("auc")),
                f1=format_percent(top_metrics.get("f1")),
            ),
            "",
        ]),
    )

    print(f"Wrote result tables to {out}")


if __name__ == "__main__":
    main()
