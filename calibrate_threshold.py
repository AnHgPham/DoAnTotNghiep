"""Tinh threshold toi uu cho demo bang cach test tren GSC.

Enroll cac word phan biet (yes,no,...) -> tinh L2 distance cho:
  - Positive: same word (different samples)
  - Negative: different word
Roi tim threshold sao cho FAR + FRR nho nhat (EER point)
        + threshold sao cho FAR=5% (strict mode)
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.models.dscnn import DSCNN
from src.features.mfcc import MFCCExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
CKPT = Path("checkpoints/best.pt")
GSC = Path("data/gsc_v2")

# Test config
TEST_WORDS  = ["yes", "no", "stop", "go", "up", "down", "left", "right", "on", "off"]
import os
N_ENROLL    = int(os.environ.get("N_ENROLL", 5))      # so sample dung de enroll moi word
N_TEST_POS  = 30     # so sample test cho moi word
N_TEST_NEG  = 30     # so sample test cua cac word khac (per word)

print(f"Loading model from {CKPT}...")
encoder = DSCNN(model_size="L", feature_mode="NORM", input_shape=(47, 10))
ckpt = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
encoder.load_state_dict(ckpt["model_state_dict"])
encoder = encoder.to(DEVICE).eval()
mfcc_ext = MFCCExtractor(n_mfcc=40, num_features=10, sample_rate=SR)
print(f"  Epoch: {ckpt.get('epoch','?')}, Loss: {ckpt.get('loss',0):.6f}")

def load_wav(path):
    w, sr = torchaudio.load(str(path))
    if sr != SR:
        w = torchaudio.transforms.Resample(sr, SR)(w)
    if w.shape[-1] < SR:
        w = F.pad(w, (0, SR - w.shape[-1]))
    return w[..., :SR]

def embed(wav):
    mfcc = mfcc_ext.extract(wav).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = F.normalize(encoder(mfcc), p=2, dim=-1)
    return emb.squeeze(0).cpu()

# === Enroll prototypes ===
prototypes = {}
test_pool = {}
print(f"\nEnrolling with {N_ENROLL} samples per word...")
for word in TEST_WORDS:
    files = sorted((GSC / word).glob("*.wav"))
    if len(files) < N_ENROLL + N_TEST_POS:
        print(f"  {word}: skip (only {len(files)} files)")
        continue
    enroll_files = files[:N_ENROLL]
    test_files   = files[N_ENROLL : N_ENROLL + N_TEST_POS]

    embs = [embed(load_wav(f)) for f in enroll_files]
    prototypes[word] = torch.stack(embs).mean(0)

    test_pool[word] = [embed(load_wav(f)) for f in test_files]
    print(f"  {word}: enrolled ({N_ENROLL}), test pool ({N_TEST_POS})")

# === Compute positive + negative distances ===
print("\nComputing distances...")
pos_dist = []  # same word
neg_dist = []  # different word

for word, embs in test_pool.items():
    proto_self = prototypes[word]
    other_protos = torch.stack([p for w, p in prototypes.items() if w != word])

    for e in embs:
        # positive: distance to own prototype
        d_self = torch.norm(e - proto_self, p=2).item()
        pos_dist.append(d_self)

        # negative: minimum distance to other prototypes (closest wrong match)
        d_others = torch.norm(other_protos - e.unsqueeze(0), p=2, dim=-1)
        neg_dist.append(d_others.min().item())

pos = np.array(pos_dist)
neg = np.array(neg_dist)

print(f"\n=== Distance distribution ===")
print(f"Positive (same word):    min={pos.min():.4f}  mean={pos.mean():.4f}  max={pos.max():.4f}  std={pos.std():.4f}")
print(f"Negative (closest wrong): min={neg.min():.4f}  mean={neg.mean():.4f}  max={neg.max():.4f}  std={neg.std():.4f}")
print(f"Overlap region: {max(pos.min(), neg.min()):.4f} - {min(pos.max(), neg.max()):.4f}")

# === Find optimal thresholds ===
print(f"\n=== Threshold sweep ===")
print(f"{'Threshold':>10} {'TPR (recall)':>15} {'FPR (FAR)':>12} {'Accuracy':>10}")
print("-" * 50)

thresholds = np.arange(0.3, 1.5, 0.05)
results = []
for t in thresholds:
    tp = (pos <= t).sum()
    fn = (pos > t).sum()
    fp = (neg <= t).sum()
    tn = (neg > t).sum()
    tpr = tp / (tp + fn) if (tp+fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    results.append((t, tpr, fpr, acc))
    if t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        print(f"{t:>10.2f} {tpr:>15.4f} {fpr:>12.4f} {acc:>10.4f}")

# EER point (FAR ~ FRR)
eer_t = min(results, key=lambda r: abs(r[2] - (1 - r[1])))
print(f"\n=== Recommended thresholds ===")
print(f"EER point (balanced):  threshold={eer_t[0]:.2f}  TPR={eer_t[1]:.4f}  FAR={eer_t[2]:.4f}  ACC={eer_t[3]:.4f}")

# Best accuracy
best_acc = max(results, key=lambda r: r[3])
print(f"Best accuracy:         threshold={best_acc[0]:.2f}  TPR={best_acc[1]:.4f}  FAR={best_acc[2]:.4f}  ACC={best_acc[3]:.4f}")

# Strict (low FAR)
strict = [r for r in results if r[2] <= 0.05]
if strict:
    strict_t = max(strict, key=lambda r: r[1])  # max TPR with FAR<=5%
    print(f"Strict (FAR<=5%):      threshold={strict_t[0]:.2f}  TPR={strict_t[1]:.4f}  FAR={strict_t[2]:.4f}  ACC={strict_t[3]:.4f}")

print(f"\n>>> SUGGESTION: Set demo threshold to {eer_t[0]:.2f} (balanced) or {best_acc[0]:.2f} (best accuracy)")
