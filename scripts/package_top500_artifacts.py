"""Package final Top500 checkpoints and result artifacts.

This is intended for Colab: create the ZIP directly on Google Drive first,
then optionally download it from the notebook. That way a runtime reset after
packaging does not lose the final model/result files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


METRIC_KEYS = [
    "open_set_acc_at_1far",
    "open_set_acc_at_5far",
    "frr_at_5far",
    "auc",
    "eer",
    "keyword_acc",
    "f1",
]


def _copy_if_exists(src: Path, drive_root: Path, package_root: Path) -> dict:
    rel = src.relative_to(drive_root)
    dst = package_root / rel
    if not src.exists():
        return {"path": str(src), "package_path": str(rel), "exists": False}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": str(src), "package_path": str(rel), "exists": True}


def _read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {key: payload.get(key) for key in METRIC_KEYS if key in payload}


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package Top500 final checkpoint/results into a Drive ZIP.",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/DoAnTotNghiep_output"),
    )
    parser.add_argument(
        "--run-tag",
        default="edgespot_full_t4_scaf_ge2e_top500_full_v1",
    )
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/DoAnTotNghiep_output/packages/"
            "edgespot_top500_final_package.zip",
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Create a partial package even if the final checkpoint/result is missing.",
    )
    args = parser.parse_args()

    drive_root = args.drive_root
    epoch_name = f"epoch_{args.epoch:02d}.pt"
    result_prefix = f"{args.run_tag}_epoch{args.epoch:02d}"

    ckpt_dir = drive_root / "checkpoints" / args.run_tag
    result_dev_dir = drive_root / "results" / f"{result_prefix}_dev30"
    result_test_dir = drive_root / "results" / f"{result_prefix}_test100"

    required = [
        ckpt_dir / epoch_name,
        result_test_dir / "gsc_edgespot_exact_k10_results.json",
    ]
    artifacts = required + [
        ckpt_dir / "best.pt",
        ckpt_dir / "latest.pt",
        result_test_dir / "gsc_edgespot_exact_det_curve.png",
        result_dev_dir / "gsc_edgespot_exact_k10_results.json",
        result_dev_dir / "gsc_edgespot_exact_det_curve.png",
    ]

    with tempfile.TemporaryDirectory(prefix="edgespot_top500_pkg_") as tmp:
        package_root = Path(tmp) / "edgespot_top500_final_package"
        package_root.mkdir(parents=True, exist_ok=True)
        copied = [_copy_if_exists(path, drive_root, package_root) for path in artifacts]
        missing_required = [str(path) for path in required if not path.exists()]
        if missing_required and not args.allow_missing:
            print("Missing required artifacts:")
            for path in missing_required:
                print("  ", path)
            raise SystemExit(2)

        test_metrics = _read_metrics(required[1])
        manifest = {
            "run_tag": args.run_tag,
            "epoch": args.epoch,
            "checkpoint": str(required[0]),
            "test100_result": str(required[1]),
            "output_zip": str(args.output_zip),
            "metrics": test_metrics,
            "artifacts": copied,
        }
        (package_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        lines = [
            "# EdgeSpot Top500 Final Package",
            "",
            f"Run tag: `{args.run_tag}`",
            f"Selected checkpoint: `{epoch_name}`",
            "",
            "Final test100 metrics:",
        ]
        for key in METRIC_KEYS:
            if key in test_metrics:
                lines.append(f"- `{key}` = {_format_metric(test_metrics[key])}")
        lines.extend(
            [
                "",
                "Use this package for demo/reporting. It does not contain the MSWC dataset.",
            ],
        )
        (package_root / "README_TOP500_FINAL.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        args.output_zip.parent.mkdir(parents=True, exist_ok=True)
        if args.output_zip.exists():
            args.output_zip.unlink()
        zip_base = args.output_zip.with_suffix("")
        zip_path = shutil.make_archive(str(zip_base), "zip", package_root)

    print("Package ZIP:", zip_path)
    print("Copied artifacts:")
    for item in copied:
        status = "OK" if item["exists"] else "MISSING"
        print(f"  {status}: {item['package_path']}")


if __name__ == "__main__":
    main()
