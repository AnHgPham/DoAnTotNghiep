"""K-shot ablation: AUC/ACC/F1 across k=1,3,5,10,20 on GSC fixed.

Usage:
    python scripts/compare_kshot.py
"""

import json
import logging
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.open_ncm import OpenNCMClassifier
from src.evaluation.gsc import GSCFewShotProvider
from src.evaluation.protocols import EvaluationProtocol
from src.models.dscnn import DSCNN

K_VALUES = [1, 3, 5, 10, 20]
N_RUNS = 3
PROTOCOL = ("gsc", "fixed")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DSCNN(model_size=cfg["model"]["architecture"][-1]).to(device)
    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?'):.4f}")

    provider = GSCFewShotProvider(cfg["data"]["gsc_dir"])
    rows = []

    print("\n" + "=" * 90)
    print(
        f"{'k':>3} | {'AUC':>10} | {'EER':>10} | {'FRR@5%FAR':>10} | "
        f"{'ACC@5%FAR':>10} | {'KW-ACC':>10} | {'F1':>10}"
    )
    print("-" * 90)

    for k in K_VALUES:
        protocol = EvaluationProtocol(
            dataset=PROTOCOL[0],
            mode=PROTOCOL[1],
            n_runs=N_RUNS,
            n_way=cfg["evaluation"]["n_way"],
            k_shot=k,
            seed=cfg["seed"],
        )
        classifier = OpenNCMClassifier()
        result = protocol.evaluate(
            encoder, classifier, provider, device=device,
            target_far=cfg["evaluation"]["target_far"],
        )
        row = {
            "k": k,
            "auc": result["auc"],
            "eer": result["eer"],
            "frr_at_far": result["frr_at_far"],
            "open_set_acc_at_far": result["open_set_acc_at_far"],
            "keyword_acc": result["keyword_acc"],
            "f1": result["f1"],
        }
        rows.append(row)
        print(
            f"{k:>3} | {row['auc']:>10.4f} | {row['eer']:>10.4f} | {row['frr_at_far']:>10.4f} | "
            f"{row['open_set_acc_at_far']:>10.4f} | {row['keyword_acc']:>10.4f} | {row['f1']:>10.4f}"
        )

    print("=" * 90)

    output_path = Path("results/kshot_ablation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
