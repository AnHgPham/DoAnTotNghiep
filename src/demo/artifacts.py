"""Artifact discovery and reporting helpers for the KWS demo.

This module intentionally avoids importing torch or model code so it can be used
from small documentation/status scripts and from the FastAPI backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


MICROSET_FINAL_METRICS = {
    "acc_at_5far": 0.8611777777777777,
    "keyword_acc": 0.7766181818181818,
    "f1": 0.8241123365438194,
    "auc": 0.9560512436363637,
    "eer": 0.11543854545454545,
    "frr_at_5far": 0.21392727272727277,
}

TOP500_EPOCH13_DEV30_METRICS = {
    "acc_at_1far": 0.8667777777777776,
    "acc_at_5far": 0.8887222222222222,
    "frr_at_5far": 0.20357575757575755,
    "auc": 0.9511623272727273,
    "f1": 0.8171095216583741,
}


@dataclass
class ArtifactRecord:
    id: str
    label: str
    status: str
    role: str
    path: str
    exists: bool
    evidence_type: str
    metrics: dict[str, Any]
    notes_vi: str
    notes_en: str


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        for match in matches:
            if match.exists():
                return match
    return None


def result_metrics(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    data = read_json(path)
    if not data:
        return fallback
    return {
        "acc_at_1far": data.get("open_set_acc_at_1far"),
        "acc_at_5far": data.get("open_set_acc_at_5far"),
        "frr_at_5far": data.get("frr_at_5far"),
        "auc": data.get("auc"),
        "eer": data.get("eer"),
        "keyword_acc": data.get("keyword_acc"),
        "f1": data.get("f1"),
    }


def discover_artifacts(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    microset_manifest = project_root / "reports" / "microset" / "locked_results_manifest.json"
    microset_ckpt = first_existing(project_root, [
        "server/**/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt",
    ])
    microset_result = first_existing(project_root, [
        "server/**/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json",
    ])

    top500_epoch13_ckpt = (
        project_root / "server" / "final_kws_artifacts_package" / "checkpoints" /
        "edgespot_full_t4_scaf_ge2e_top500_full_v1" / "epoch_13.pt"
    )
    top500_epoch13_result = (
        project_root / "server" / "final_kws_artifacts_package" / "results" /
        "edgespot_full_t4_scaf_ge2e_top500_full_v1_epoch13_dev30" /
        "gsc_edgespot_exact_k10_results.json"
    )
    top500_package_manifest = (
        project_root / "server" / "final_kws_artifacts_package" / "manifest.json"
    )

    records = [
        ArtifactRecord(
            id="microset_edgespot_t4_scaf_ge2e_epoch05",
            label="Microset EdgeSpotFull T4 + SCAF+GE2E epoch05",
            status="official_locked",
            role="thesis_main_result",
            path=str(microset_ckpt or microset_manifest),
            exists=bool((microset_ckpt and microset_ckpt.exists()) or microset_manifest.exists()),
            evidence_type="checkpoint_result_manifest",
            metrics=result_metrics(microset_result, MICROSET_FINAL_METRICS) if microset_result else MICROSET_FINAL_METRICS,
            notes_vi=(
                "Mốc chính cho thesis hiện tại. Kết quả này đến từ các thử nghiệm Microset "
                "và là cơ sở chọn EdgeSpotFull T4 + SCAF+GE2E để mở rộng sang Top500."
            ),
            notes_en=(
                "Current thesis anchor. Microset experiments selected EdgeSpotFull T4 + "
                "SCAF+GE2E before scaling the same direction to Top500."
            ),
        ),
        ArtifactRecord(
            id="top500_edgespot_t4_scaf_ge2e_epoch13",
            label="Top500 EdgeSpotFull T4 + SCAF+GE2E epoch13",
            status="available_local",
            role="demo_and_preliminary_top500",
            path=str(top500_epoch13_ckpt),
            exists=top500_epoch13_ckpt.exists(),
            evidence_type="checkpoint_dev30_result",
            metrics=result_metrics(top500_epoch13_result, TOP500_EPOCH13_DEV30_METRICS),
            notes_vi=(
                "Checkpoint Top500 chắc chắn đang có ở local. Run bị dừng ở epoch 13 do "
                "giới hạn Colab/session/units, nên dùng cho demo và phân tích sơ bộ."
            ),
            notes_en=(
                "Local Top500 checkpoint available. The run stopped at epoch 13 because of "
                "Colab/session/unit limits, so it is used for demo and preliminary analysis."
            ),
        ),
        ArtifactRecord(
            id="top500_edgespot_t4_scaf_ge2e_epoch25_historical",
            label="Top500 EdgeSpotFull T4 + SCAF+GE2E epoch25 historical run",
            status="historical_colab_log",
            role="progress_history_not_locked_artifact",
            path=str(top500_package_manifest),
            exists=top500_package_manifest.exists(),
            evidence_type="package_manifest_or_log",
            metrics={},
            notes_vi=(
                "Lần chạy Top500 trước đó có log tốt hơn nhưng không có checkpoint/result "
                "local đầy đủ trong package hiện tại, nên chỉ ghi trong lịch sử tiến độ."
            ),
            notes_en=(
                "An earlier Top500 run had promising logs, but the current package does not "
                "contain complete local checkpoint/result artifacts, so it is progress history."
            ),
        ),
    ]

    package_manifest = read_json(top500_package_manifest) or {}
    return {
        "generated_from": str(project_root),
        "records": [asdict(record) for record in records],
        "microset_manifest": str(microset_manifest),
        "top500_package_manifest": str(top500_package_manifest),
        "top500_package": package_manifest,
    }


def format_percent(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def artifact_markdown(status: dict[str, Any], lang: str = "vi") -> str:
    title = "Project Artifact Status" if lang == "en" else "Trạng Thái Artifact Dự Án"
    lines = [f"# {title}", ""]
    if lang == "en":
        lines.extend([
            "This file summarizes which results are backed by local artifacts and how each result should be used.",
            "",
            "| Artifact | Status | Role | Evidence | ACC@5%FAR | KW-ACC | F1 | Notes |",
            "|---|---|---|---|---:|---:|---:|---|",
        ])
    else:
        lines.extend([
            "File này tổng hợp kết quả nào đang có artifact local và nên dùng mỗi kết quả trong ngữ cảnh nào.",
            "",
            "| Artifact | Trạng thái | Vai trò | Bằng chứng | ACC@5%FAR | KW-ACC | F1 | Ghi chú |",
            "|---|---|---|---|---:|---:|---:|---|",
        ])

    for record in status.get("records", []):
        metrics = record.get("metrics", {})
        note = record.get("notes_en" if lang == "en" else "notes_vi", "")
        evidence = "yes" if record.get("exists") else "missing"
        lines.append(
            "| {label} | {status} | {role} | {evidence} | {acc5} | {kw} | {f1} | {note} |".format(
                label=record.get("label", ""),
                status=record.get("status", ""),
                role=record.get("role", ""),
                evidence=evidence,
                acc5=format_percent(metrics.get("acc_at_5far")),
                kw=format_percent(metrics.get("keyword_acc")),
                f1=format_percent(metrics.get("f1")),
                note=note.replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)
