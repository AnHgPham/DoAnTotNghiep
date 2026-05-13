"""Deep confusion analysis on GSC test set.

Reads ``results/confusion_matrix_best_v2.json`` (or recomputes if missing)
and produces:

  - ``results/confusion_analysis.md``: human-readable report with phonetic
    grouping, per-cluster accuracy, asymmetry analysis.
  - ``results/hard_pairs.json``: machine-readable list of hard pairs that
    Phase 2 (training-side hard-pair mining) will consume.

Usage:
    python scripts/analyze_confusion.py
    python scripts/analyze_confusion.py --recompute    # re-run classification
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CM_PATH = PROJECT_ROOT / "results" / "confusion_matrix_best_v2_margin1.0_colab.json"
ANALYSIS_PATH = PROJECT_ROOT / "results" / "confusion_analysis.md"
HARD_PAIRS_PATH = PROJECT_ROOT / "results" / "hard_pairs.json"

PHONETIC_GROUPS: dict[str, list[str]] = {
    "vowel-o (go/no/down/dog)":         ["go", "no", "down", "dog"],
    "fricative-f (four/forward/follow/five)":["four", "forward", "follow", "five"],
    "tree/three (sibilant onset)":       ["three", "tree"],
    "bed/bird (vowel + r/d)":            ["bed", "bird"],
    "off/on/up (back-vowel + nasal)":    ["off", "on", "up", "house"],
    "left/right (lateral)":              ["left", "right"],
    "marvin/sheila/learn (rare names)":  ["marvin", "sheila", "learn", "happy", "wow"],
    "numbers (0-9)":                     ["zero", "one", "two", "three", "four",
                                          "five", "six", "seven", "eight", "nine"],
    "animals (cat/dog/bird)":            ["cat", "dog", "bird"],
    "directions (up/down/left/right/forward/backward)":
        ["up", "down", "left", "right", "forward", "backward"],
}

MIN_PAIR_COUNT = 5
HARD_PAIR_KEEP = 30


def load_or_compute() -> dict:
    if not CM_PATH.exists():
        print(f"  Confusion matrix not found at {CM_PATH}")
        print("  -> Run scripts/compute_confusion_matrix.py first.")
        sys.exit(1)
    return json.loads(CM_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Re-run classification before analysis")
    args = parser.parse_args()

    if args.recompute:
        print("Recomputing confusion matrix...")
        import subprocess
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "compute_confusion_matrix.py")],
            check=True,
        )

    cm_data = load_or_compute()
    labels: list[str] = cm_data["labels"]
    cm = cm_data["confusion_matrix"]
    per_class = cm_data["per_class_accuracy"]
    overall = cm_data["overall_accuracy"]

    word_to_idx = {w: i for i, w in enumerate(labels)}

    pair_counts: dict[tuple[str, str], int] = {}
    for i, true_w in enumerate(labels):
        for j, pred_w in enumerate(labels):
            if i == j:
                continue
            if cm[i][j] >= MIN_PAIR_COUNT:
                pair_counts[(true_w, pred_w)] = cm[i][j]

    asymmetric = []
    seen_pairs: set[frozenset[str]] = set()
    for (a, b), count_ab in pair_counts.items():
        key = frozenset({a, b})
        if key in seen_pairs:
            continue
        count_ba = pair_counts.get((b, a), 0)
        if abs(count_ab - count_ba) >= MIN_PAIR_COUNT:
            asymmetric.append({
                "a": a, "b": b,
                "ab": count_ab, "ba": count_ba,
                "diff": count_ab - count_ba,
            })
        seen_pairs.add(key)
    asymmetric.sort(key=lambda r: abs(r["diff"]), reverse=True)

    cluster_acc: list[dict] = []
    for name, members in PHONETIC_GROUPS.items():
        present = [m for m in members if m in labels]
        if len(present) < 2:
            continue
        intra = 0
        intra_total = 0
        for m in present:
            i = word_to_idx[m]
            for n in present:
                if m == n:
                    continue
                j = word_to_idx[n]
                intra += cm[i][j]
                intra_total += cm[i][j]
            intra_total += cm[i][word_to_idx[m]]
        accs = [per_class[m] for m in present]
        avg_acc = float(sum(accs) / len(accs))
        cluster_acc.append({
            "cluster": name,
            "members": present,
            "avg_accuracy": avg_acc,
            "intra_cluster_errors": intra,
            "members_n_test": intra_total,
        })
    cluster_acc.sort(key=lambda r: r["avg_accuracy"])

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    hard_pairs = [
        {"true": t, "pred": p, "count": c}
        for (t, p), c in sorted_pairs[:HARD_PAIR_KEEP]
    ]

    weights: dict[tuple[str, str], float] = {}
    for entry in hard_pairs:
        key = frozenset({entry["true"], entry["pred"]})
        a, b = sorted(key)
        weights[(a, b)] = weights.get((a, b), 0) + entry["count"]
    total_pair_count = sum(weights.values())
    pair_weights = {
        f"{a}|{b}": round(c / max(total_pair_count, 1), 4)
        for (a, b), c in weights.items()
    }

    HARD_PAIRS_PATH.write_text(json.dumps({
        "checkpoint": cm_data.get("checkpoint", ""),
        "epoch": cm_data.get("epoch", -1),
        "min_pair_count": MIN_PAIR_COUNT,
        "hard_pairs_directional": hard_pairs,
        "hard_pairs_undirected_weights": pair_weights,
    }, indent=2), encoding="utf-8")

    md = ["# Confusion Analysis (best_v2_margin1.0_colab.pt)", ""]
    md.append(f"- **Checkpoint**: `{cm_data.get('checkpoint', '?')}` (epoch {cm_data.get('epoch','?')})")
    md.append(f"- **Overall closed-set top-1 accuracy**: **{overall:.4f}**")
    md.append(f"- **Number of keywords**: {len(labels)}")
    md.append(f"- **Enroll/Test**: {cm_data.get('n_enroll','?')} / {cm_data.get('max_test_per_word','?')} samples per word")
    md.append("")

    md.append("## 1. Per-class accuracy (sorted, ascending)")
    md.append("")
    md.append("| Word | Accuracy | Top-3 confusion targets |")
    md.append("|------|---------:|-------------------------|")
    for w, acc in sorted(per_class.items(), key=lambda x: x[1]):
        i = word_to_idx[w]
        confs = []
        wrong = sorted(
            [(j, cm[i][j]) for j in range(len(labels)) if j != i and cm[i][j] > 0],
            key=lambda x: x[1], reverse=True,
        )[:3]
        for j, c in wrong:
            confs.append(f"`{labels[j]}` x{c}")
        md.append(f"| `{w}` | {acc:.3f} | {', '.join(confs) or '-'} |")
    md.append("")

    md.append("## 2. Cluster-level accuracy (phonetic grouping)")
    md.append("")
    md.append("Members in the same cluster share phonetic features and are most"
              " susceptible to mutual confusion. Sorted ascending by average per-class"
              " accuracy.")
    md.append("")
    md.append("| Cluster | Members | Avg accuracy |")
    md.append("|---------|---------|------------:|")
    for c in cluster_acc:
        md.append(f"| {c['cluster']} | {', '.join(c['members'])} | {c['avg_accuracy']:.3f} |")
    md.append("")

    md.append(f"## 3. Asymmetric confusion (where A->B count differs from B->A by >= {MIN_PAIR_COUNT})")
    md.append("")
    md.append("Indicates encoder bias (model favors one word over another, often due"
              " to training-set imbalance).")
    md.append("")
    md.append("| A | B | A->B | B->A | Δ |")
    md.append("|---|---|-----:|-----:|--:|")
    for r in asymmetric[:15]:
        md.append(f"| `{r['a']}` | `{r['b']}` | {r['ab']} | {r['ba']} | {r['diff']:+d} |")
    md.append("")

    md.append(f"## 4. Top {HARD_PAIR_KEEP} hard pairs (directional, count >= {MIN_PAIR_COUNT})")
    md.append("")
    md.append("| # | True | Predicted | Count |")
    md.append("|---|------|-----------|------:|")
    for i, entry in enumerate(hard_pairs, 1):
        md.append(f"| {i} | `{entry['true']}` | `{entry['pred']}` | {entry['count']} |")
    md.append("")

    md.append("## 5. Recommended actions for Phase 2 retrain")
    md.append("")
    md.append("- **Hard-pair mining**: bias `EpisodicBatchSampler` so episodes more"
              " often contain both elements of the top hard pairs above.")
    md.append("- **Class balancing**: words with strong asymmetry (A->B much higher"
              " than B->A) likely have unequal sample count in MSWC -> oversample the"
              " minority class.")
    md.append("- **Augmentation focus**: per-class accuracy < 0.50 keywords (`follow`,"
              " `sheila`, `down`, `three`, `four`, `no`, `off`, `bed`) need more"
              " augmentation diversity (speed perturb, pitch shift, noise mixing).")
    md.append("")
    md.append(f"**Hard-pair JSON saved**: `{HARD_PAIRS_PATH.relative_to(PROJECT_ROOT)}`"
              f" — Phase 2 training script will read from here.")

    ANALYSIS_PATH.write_text("\n".join(md), encoding="utf-8")

    print(f"Overall accuracy: {overall:.4f}  ({len(labels)} keywords)")
    print(f"\nWeakest 5 keywords:")
    for w, acc in sorted(per_class.items(), key=lambda x: x[1])[:5]:
        print(f"  {w:>10s}: {acc:.3f}")
    print(f"\nStrongest 5 keywords:")
    for w, acc in sorted(per_class.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {w:>10s}: {acc:.3f}")
    print(f"\nTop 10 hard pairs (directional):")
    for entry in hard_pairs[:10]:
        print(f"  {entry['true']:>10s} -> {entry['pred']:<10s}  count={entry['count']}")
    print(f"\nReports saved:")
    print(f"  {ANALYSIS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {HARD_PAIRS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
