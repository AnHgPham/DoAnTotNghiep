"""Quick demo: Full KWS pipeline with synthetic audio.

Demonstrates the entire pipeline end-to-end:
  Audio -> MFCC -> DSCNN-L -> Embedding -> Enroll Prototypes -> Predict -> Open-Set Reject

Uses untrained model with synthetic audio -- predictions are random,
but proves the architecture works correctly.

Usage:
    python demo_quick.py
"""

import torch
import torch.nn.functional as F

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.classifiers.open_ncm import OpenNCMClassifier


def generate_synthetic_audio(n_samples: int = 1, duration_sec: float = 1.0) -> torch.Tensor:
    """Generate random audio waveforms simulating speech."""
    sr = 16000
    length = int(sr * duration_sec)
    return torch.randn(n_samples, 1, length) * 0.3


def print_header(text: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_step(step: int, text: str) -> None:
    print(f"\n  [{step}] {text}")


def main() -> None:
    print_header("Few-Shot Open-Set Keyword Spotting Demo")
    print("  Architecture: DSCNN-L (276ch, 5 DS blocks)")
    print("  NOTE: Using untrained model -- predictions are random")
    print("        Train on MSWC for real results")

    # --- Step 1: Initialize ---
    print_step(1, "Initializing components...")

    extractor = MFCCExtractor()
    encoder = DSCNN(model_size="L", feature_mode="NORM")
    encoder.eval()
    classifier = OpenNCMClassifier()

    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"      DSCNN-L parameters: {param_count:,}")
    print(f"      Embedding dim: {encoder.embedding_dim}")

    # --- Step 2: Enrollment (Few-Shot) ---
    print_step(2, "Enrolling keywords (5-shot each)...")

    keywords = ["yes", "no", "stop"]
    k_shot = 5
    prototypes = []

    for word in keywords:
        support_audio = generate_synthetic_audio(k_shot)
        support_mfcc = extractor.extract_batch(support_audio)
        print(f"      '{word}': audio {tuple(support_audio.shape)} -> MFCC {tuple(support_mfcc.shape)}")

        with torch.no_grad():
            embeddings = encoder(support_mfcc)
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        prototype = embeddings.mean(dim=0)
        prototypes.append(prototype)
        print(f"              -> {k_shot} embeddings -> prototype {tuple(prototype.shape)}")

    proto_tensor = torch.stack(prototypes)
    classifier.set_prototypes(proto_tensor, keywords)
    classifier.threshold = 1.4  # manual threshold for demo
    print(f"\n      Enrolled {len(keywords)} keywords, threshold = {classifier.threshold}")

    # --- Step 3: Query -- Known keyword ---
    print_step(3, "Testing with known-like queries...")

    for word in keywords:
        query_audio = generate_synthetic_audio(1)
        query_mfcc = extractor.extract_batch(query_audio)

        with torch.no_grad():
            query_emb = encoder(query_mfcc)
            query_emb = F.normalize(query_emb, p=2, dim=-1)

        pred, dist = classifier.predict(query_emb.squeeze(0))
        distances = classifier.get_distances(query_emb.squeeze(0))

        status = "ACCEPTED" if pred != "unknown" else "REJECTED"
        color_start = "\033[92m" if pred != "unknown" else "\033[91m"
        color_end = "\033[0m"

        print(f"      Query -> {color_start}{status}: '{pred}' (dist={dist:.4f}){color_end}")
        print(f"               Distances: {', '.join(f'{k}={v:.4f}' for k, v in distances.items())}")

    # --- Step 4: Query -- Unknown word (open-set rejection) ---
    print_step(4, "Testing open-set rejection (unknown word)...")

    for i in range(3):
        unknown_audio = generate_synthetic_audio(1)
        unknown_mfcc = extractor.extract_batch(unknown_audio)

        with torch.no_grad():
            unknown_emb = encoder(unknown_mfcc)
            unknown_emb = F.normalize(unknown_emb, p=2, dim=-1)

        pred, dist = classifier.predict(unknown_emb.squeeze(0))
        distances = classifier.get_distances(unknown_emb.squeeze(0))

        status = "REJECTED (unknown)" if pred == "unknown" else f"ACCEPTED as '{pred}'"
        color_start = "\033[91m" if pred == "unknown" else "\033[93m"
        color_end = "\033[0m"

        print(f"      Unknown #{i+1} -> {color_start}{status} (dist={dist:.4f}){color_end}")

    # --- Step 5: Pipeline shapes summary ---
    print_step(5, "Pipeline shape verification")

    wav = torch.randn(1, 16000)
    mfcc = extractor.extract(wav)
    mfcc_batch = mfcc.unsqueeze(0)
    with torch.no_grad():
        emb = encoder(mfcc_batch)
        emb_norm = F.normalize(emb, p=2, dim=-1)

    print(f"      Audio:     (1, 16000)  -- 1 second @ 16kHz")
    print(f"      MFCC:      {tuple(mfcc.shape)}    -- 47 frames x 10 features")
    print(f"      DSCNN in:  {tuple(mfcc_batch.shape)} -- batch dim added")
    print(f"      Embedding: {tuple(emb.shape)}      -- 276-dim raw")
    print(f"      L2-normed: {tuple(emb_norm.shape)}  -- unit sphere")
    print(f"      L2 norm:   {emb_norm.norm(dim=-1).item():.6f} (should be 1.0)")

    # --- Summary ---
    print_header("Demo Complete")
    print("  Pipeline:  WAV -> MFCC(47,10) -> DSCNN-L(276) -> L2-norm -> Classify")
    print(f"  Model:     DSCNN-L ({param_count:,} params)")
    print(f"  Keywords:  {keywords}")
    print(f"  Threshold: {classifier.threshold}")
    print()
    print("  Next steps:")
    print("    1. Download data:  python data/download_gsc.py")
    print("    2. Train model:    python scripts/train.py --config configs/default.yaml")
    print("    3. Evaluate:       python scripts/evaluate.py --checkpoint checkpoints/best.pt")
    print()


if __name__ == "__main__":
    main()
