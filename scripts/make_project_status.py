"""Generate project artifact status files for thesis/demo reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.artifacts import artifact_markdown, discover_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Repository root")
    parser.add_argument(
        "--out-dir",
        default="reports/project_status",
        help="Output directory for generated status files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    status = discover_artifacts(project_root)
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "result_story_vi.md").write_text(
        artifact_markdown(status, lang="vi"),
        encoding="utf-8",
    )
    (out_dir / "result_story_en.md").write_text(
        artifact_markdown(status, lang="en"),
        encoding="utf-8",
    )

    claim_lines = [
        "# Claim Matrix",
        "",
        "| Statement | Evidence | Status | Use in Thesis | Use in Email | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for record in status["records"]:
        if record["status"] == "official_locked":
            thesis = "yes"
            email = "yes"
        elif record["status"] == "available_local":
            thesis = "progress/preliminary"
            email = "yes"
        else:
            thesis = "no"
            email = "progress only"
        claim_lines.append(
            "| {statement} | {evidence} | {status} | {thesis} | {email} | {notes} |".format(
                statement=record["label"],
                evidence=record["path"].replace("|", "/"),
                status=record["status"],
                thesis=thesis,
                email=email,
                notes=record["notes_en"].replace("|", "/"),
            )
        )
    claim_lines.append("")
    (out_dir / "claim_matrix.md").write_text("\n".join(claim_lines), encoding="utf-8")

    print(f"Wrote project status to {out_dir}")


if __name__ == "__main__":
    main()
