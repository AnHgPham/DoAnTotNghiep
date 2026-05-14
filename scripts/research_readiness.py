"""Check whether the repository has enough artifacts for a paper-grade report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _count_wavs(data_dir: Path) -> tuple[int, int]:
    clips = data_dir / "clips"
    root = clips if clips.exists() else data_dir
    if not root.exists():
        return 0, 0
    word_dirs = [p for p in root.iterdir() if p.is_dir()]
    wav_n = sum(1 for _ in root.rglob("*.wav"))
    words_with_wav = sum(1 for p in word_dirs if any(p.glob("*.wav")))
    return words_with_wav, wav_n


def _has_splits(data_dir: Path) -> bool:
    splits = data_dir / "splits"
    return (
        (splits / "train_words.json").exists()
        and (splits / "val_words.json").exists()
    )


def _find_result_json(results_root: Path, run_tag: str) -> list[Path]:
    if not results_root.exists():
        return []
    candidates = []
    for path in results_root.rglob("*_results.json"):
        if run_tag in str(path):
            candidates.append(path)
    return sorted(candidates)


def _best_metric(paths: list[Path], metric: str) -> float | None:
    vals = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metric in data:
            vals.append(float(data[metric]))
    return max(vals) if vals else None


def _status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def main() -> None:
    parser = argparse.ArgumentParser(description="Research artifact readiness check")
    parser.add_argument("--manifest", type=Path, default=Path("configs/research_experiments.yaml"))
    parser.add_argument("--data-profile", default="top500_full")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    manifest = _load_yaml(args.manifest)
    profiles = manifest.get("data_profiles", {})
    if args.data_profile not in profiles:
        raise SystemExit(f"Unknown data profile: {args.data_profile}. Choices: {sorted(profiles)}")

    profile = profiles[args.data_profile]
    data_dir = Path(profile["data_dir"])
    data_label = args.data_profile
    if args.data_profile == "top500_full":
        data_label = "top500_full"

    print("# Research Readiness")
    print(f"manifest: {args.manifest}")
    print(f"data_profile: {args.data_profile}")
    print(f"data_dir: {data_dir}")
    print()

    words_with_wav, wav_n = _count_wavs(data_dir)
    print("## Data")
    print(f"- exists: {_status(data_dir.exists())}")
    print(f"- splits: {_status(_has_splits(data_dir))}")
    print(f"- words_with_wav: {words_with_wav}")
    print(f"- wav_files: {wav_n}")
    print(f"- allowed_claim: {profile.get('allowed_claim', '')}")
    print(f"- forbidden_claim: {profile.get('forbidden_claim', '')}")
    print()

    print("## Required Repo Artifacts")
    missing_artifacts = []
    for rel in manifest.get("artifact_requirements", {}).get("required", []):
        ok = Path(rel).exists()
        print(f"- {_status(ok)} {rel}")
        if not ok:
            missing_artifacts.append(rel)
    print()

    print("## Experiments")
    required_missing = []
    experiments = manifest.get("experiments", [])
    for exp in experiments:
        template = exp.get("run_tag_template", exp["id"] + "_{data_label}")
        run_tag = template.format(data_label=data_label)
        ckpt = args.checkpoints_root / run_tag / "best.pt"
        result_paths = _find_result_json(args.results_root, run_tag)
        result_1far = _best_metric(result_paths, "open_set_acc_at_1far")
        priority = exp.get("priority", "recommended")
        ckpt_ok = ckpt.exists()
        result_ok = bool(result_paths)
        print(f"- {exp['id']} [{priority}]")
        print(f"  run_tag: {run_tag}")
        print(f"  checkpoint: {_status(ckpt_ok)} {ckpt}")
        print(f"  results: {_status(result_ok)} ({len(result_paths)} json)")
        if result_1far is not None:
            print(f"  best_ACC@1%FAR: {result_1far:.4f}")
        if priority == "required" and (not ckpt_ok or not result_ok):
            required_missing.append(exp["id"])
    print()

    print("## Targets")
    target_failures = []
    for name, target in manifest.get("targets", {}).items():
        metric = target.get("metric")
        min_value = target.get("min_value")
        scope = target.get("scope")
        if not metric or min_value is None or (scope and scope != args.data_profile):
            continue
        vals = []
        for exp in experiments:
            run_tag = exp.get("run_tag_template", exp["id"] + "_{data_label}").format(data_label=data_label)
            val = _best_metric(_find_result_json(args.results_root, run_tag), metric)
            if val is not None:
                vals.append(val)
        best = max(vals) if vals else None
        ok = best is not None and best >= float(min_value)
        print(f"- {name}: {_status(ok)} {metric}>={float(min_value):.4f} best={best}")
        if not ok:
            target_failures.append(name)
    print()

    ready = (
        data_dir.exists()
        and _has_splits(data_dir)
        and wav_n > 0
        and not missing_artifacts
        and not required_missing
        and not target_failures
    )
    print("## Summary")
    print(f"paper_ready: {ready}")
    if not ready:
        print("next_actions:")
        if not data_dir.exists() or wav_n == 0:
            print("- prepare data profile and run scripts/mswc_data_report.py")
        if missing_artifacts:
            print("- restore missing repo artifacts")
        if required_missing:
            print(f"- run required experiments: {', '.join(required_missing)}")
        if target_failures:
            print(f"- improve or rerun target experiments: {', '.join(target_failures)}")


if __name__ == "__main__":
    main()
