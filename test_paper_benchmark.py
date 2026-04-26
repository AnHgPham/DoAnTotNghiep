"""Reproduce paper benchmark: enroll va test ca 2 deu lay tu GSC test set.

So sanh voi Rusci 2023:
  - 10-shot 10-way open-set: 63% acc (TL+openNCM) - 76% acc (TL+Dproto)
  - FRR @ FAR=5%: ~30%
"""
import random
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

# Paper setup: 10-way 10-shot
TARGET_WORDS  = ["yes", "no", "stop", "go", "up", "down", "left", "right", "on", "off"]
# Unknown words (negative set) - cac word khong enroll
UNKNOWN_WORDS = ["bed", "bird", "cat", "dog", "four", "five", "eight", "nine", "zero", "one",
                 "two", "three", "seven", "six", "tree", "happy", "house", "marvin", "sheila", "wow"]
K_SHOT = 10  # Enroll 10 samples per target word
N_POS_TEST = 50  # Test samples per target word
N_NEG_TEST = 20  # Unknown samples per unknown word
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

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

# --- Enroll prototypes tu GSC (khong phai mic) ---
print(f"\nEnrolling {K_SHOT} samples/word from GSC (same distribution as test)...")
prototypes = {}
used_files = {}  # de tranh dung lai file khi test
for word in TARGET_WORDS:
    files = sorted((GSC / word).glob("*.wav"))
    random.shuffle(files)
    enroll_files = files[:K_SHOT]
    used_files[word] = set(str(f) for f in enroll_files)
    embs = [embed(load_wav(f)) for f in enroll_files]
    prototypes[word] = torch.stack(embs).mean(0)

print(f"Enrolled {len(prototypes)} keywords, {K_SHOT} samples each")

# --- Collect positive tests (known words, different samples) ---
print(f"\nCollecting {N_POS_TEST} test samples per target word...")
pos_embs = []  # (embedding, true_label)
for word in TARGET_WORDS:
    files = sorted((GSC / word).glob("*.wav"))
    random.shuffle(files)
    count = 0
    for f in files:
        if str(f) in used_files[word]:
            continue
        pos_embs.append((embed(load_wav(f)), word))
        count += 1
        if count >= N_POS_TEST:
            break

# --- Collect negative tests (unknown words) ---
print(f"Collecting negative (unknown) samples...")
neg_embs = []
for word in UNKNOWN_WORDS:
    word_dir = GSC / word
    if not word_dir.exists():
        continue
    files = sorted(word_dir.glob("*.wav"))
    random.shuffle(files)
    for f in files[:N_NEG_TEST]:
        neg_embs.append(embed(load_wav(f)))

print(f"  Positive: {len(pos_embs)} samples across {len(TARGET_WORDS)} keywords")
print(f"  Negative: {len(neg_embs)} unknown samples")

# --- openNCM: classify by nearest prototype ---
print(f"\n=== openNCM Classification ===")
proto_names = list(prototypes.keys())
proto_stack = torch.stack([prototypes[w] for w in proto_names])  # (n_proto, dim)

def classify(e, threshold):
    d = torch.norm(proto_stack - e.unsqueeze(0), p=2, dim=-1)  # (n_proto,)
    min_d, min_i = d.min(0)
    return proto_names[min_i.item()] if min_d.item() <= threshold else "unknown", min_d.item()

# --- Find threshold at FAR=5% ---
neg_dists = [torch.norm(proto_stack - e.unsqueeze(0), p=2, dim=-1).min().item() for e in neg_embs]
neg_dists_sorted = sorted(neg_dists)
far_5_idx = int(0.05 * len(neg_dists_sorted))
threshold_5far = neg_dists_sorted[far_5_idx]
print(f"Threshold @ FAR=5%: {threshold_5far:.4f}")

# --- Evaluate at FAR=5% ---
correct_known = 0
wrong_known = 0
rejected_known = 0
for emb, true_label in pos_embs:
    pred, d = classify(emb, threshold_5far)
    if pred == "unknown":
        rejected_known += 1
    elif pred == true_label:
        correct_known += 1
    else:
        wrong_known += 1

# FAR on negatives
far_count = sum(1 for e in neg_embs if classify(e, threshold_5far)[0] != "unknown")
far_actual = far_count / len(neg_embs)

total_known = len(pos_embs)
keyword_acc = correct_known / total_known
frr = rejected_known / total_known

print(f"\n=== Results (paper-style, 10-way {K_SHOT}-shot open-set) ===")
print(f"  Keyword Accuracy (correct/total known):  {keyword_acc:.4f} ({correct_known}/{total_known})")
print(f"  Wrong predictions (known-known errors):  {wrong_known/total_known:.4f}")
print(f"  FRR (rejected known):                    {frr:.4f}")
print(f"  FAR (accepted unknown):                  {far_actual:.4f}")
print(f"  Open-set ACC @ FAR=5%:                   {correct_known/total_known:.4f}")

# Also compute at different thresholds
print(f"\n=== Paper comparison (Rusci 2023) ===")
print(f"  Paper (TL+NORM+openNCM, 10-shot):   63% accuracy @ FAR=5%")
print(f"  Paper (TL+NORM+Dproto, 10-shot):    76% accuracy @ FAR=5% (best)")
print(f"  YOUR MODEL (15/40 epochs):          {keyword_acc*100:.1f}% accuracy @ FAR=5%")

if keyword_acc >= 0.63:
    print(f"\n  >>> YOUR MODEL EXCEEDS paper baseline! (+{(keyword_acc-0.63)*100:.1f}%)")
if keyword_acc >= 0.76:
    print(f"  >>> YOUR MODEL MATCHES best paper result!")
