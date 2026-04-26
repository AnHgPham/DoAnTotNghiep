"""Benchmark EXT-1 denoiser: KWS accuracy across SNR levels with/without denoising.

Mixes GSC test queries with GSC background noise at fixed SNR levels, runs the
GSC fixed protocol both with raw noisy audio and after spectral-gating denoising,
and reports the AUC / ACC delta.

Usage:
    python scripts/benchmark_denoiser.py
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.open_ncm import OpenNCMClassifier
from src.enhancements.denoiser import Denoiser
from src.evaluation.gsc import GSCFewShotProvider
from src.evaluation.protocols import EvaluationProtocol
from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN

SR = 16000
NOISE_DIR = Path("data/gsc_v2/_background_noise_")
SNR_LEVELS = [0, 5, 10, 20]  # plus clean baseline
N_RUNS = 3


def load_noise_clips() -> list[torch.Tensor]:
    """Load all GSC background noise clips as 1D tensors at 16kHz."""
    clips: list[torch.Tensor] = []
    for path in sorted(NOISE_DIR.glob("*.wav")):
        wav, sr = torchaudio.load(str(path))
        if sr != SR:
            wav = torchaudio.transforms.Resample(sr, SR)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        clips.append(wav.squeeze(0))
    return clips


def mix_at_snr(
    clean: torch.Tensor, noise: torch.Tensor, snr_db: float, rng: random.Random,
) -> torch.Tensor:
    """Mix a clean clip with a random noise excerpt at the requested SNR (dB)."""
    if clean.dim() == 2:
        clean = clean.squeeze(0)

    if noise.numel() <= clean.numel():
        repeat = (clean.numel() // noise.numel()) + 1
        noise = noise.repeat(repeat)
    start = rng.randint(0, max(0, noise.numel() - clean.numel()))
    noise_chunk = noise[start:start + clean.numel()]

    rms_clean = torch.sqrt(torch.mean(clean ** 2)).clamp_min(1e-6)
    rms_noise = torch.sqrt(torch.mean(noise_chunk ** 2)).clamp_min(1e-6)
    scale = rms_clean / (rms_noise * (10 ** (snr_db / 20.0)))
    return (clean + scale * noise_chunk).clamp(-1.0, 1.0)


class NoisyMFCCProvider:
    """Wrap a GSCFewShotProvider so that QUERY clips are mixed with noise."""

    def __init__(
        self,
        base: GSCFewShotProvider,
        snr_db: float | None,
        denoiser: Denoiser | None,
        rng: random.Random,
    ):
        self.base = base
        self.snr_db = snr_db  # None means clean (no noise mixing)
        self.denoiser = denoiser
        self.rng = rng
        self.extractor = MFCCExtractor()
        self.noise_clips = load_noise_clips()

    def validate_words(self, words: list[str], min_support: int = 5) -> None:
        self.base.validate_words(words, min_support=min_support)

    def get_support_samples(self, word: str, n_samples: int, seed: int = 42):
        return self.base.get_support_samples(word, n_samples, seed=seed)

    def get_query_samples(self, word: str, max_samples: int = 50):
        paths = self.base._test_files.get(word, [])
        if not paths:
            return self.base.get_query_samples(word, max_samples=max_samples)
        if len(paths) > max_samples:
            paths = random.Random(42).sample(paths, max_samples)

        mfccs = []
        names = []
        for p in paths:
            wav, sr = torchaudio.load(str(p))
            if sr != SR:
                wav = torchaudio.transforms.Resample(sr, SR)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if wav.shape[-1] < SR:
                wav = F.pad(wav, (0, SR - wav.shape[-1]))
            wav = wav[..., :SR]

            if self.snr_db is None:
                processed = wav
            else:
                noise = self.rng.choice(self.noise_clips)
                processed = mix_at_snr(
                    wav.squeeze(0), noise, self.snr_db, self.rng,
                ).unsqueeze(0)

            if self.denoiser is not None and self.denoiser.enabled:
                processed = self.denoiser.denoise(processed)

            mfcc = self.extractor.extract(processed)
            mfccs.append(mfcc.unsqueeze(0))
            names.append(p.name)
        return torch.cat(mfccs, dim=0), names


def evaluate(
    encoder: torch.nn.Module,
    base_provider: GSCFewShotProvider,
    snr_db: float | None,
    use_denoiser: bool,
    cfg: dict,
    device: torch.device,
) -> dict:
    rng = random.Random(cfg["seed"])
    denoiser = Denoiser(backend="spectral_gate", enabled=True) if use_denoiser else None
    provider = NoisyMFCCProvider(base_provider, snr_db, denoiser, rng)
    protocol = EvaluationProtocol(
        dataset="gsc",
        mode="fixed",
        n_runs=N_RUNS,
        n_way=cfg["evaluation"]["n_way"],
        k_shot=cfg["evaluation"]["k_shot"],
        seed=cfg["seed"],
    )
    classifier = OpenNCMClassifier()
    return protocol.evaluate(
        encoder,
        classifier,
        provider,
        device=device,
        target_far=cfg["evaluation"]["target_far"],
    )


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

    base_provider = GSCFewShotProvider(cfg["data"]["gsc_dir"])

    print("\n" + "=" * 90)
    print(f"{'Condition':<22} | {'AUC':>8} | {'EER':>8} | {'ACC@5%FAR':>10} | {'KW-ACC':>8} | {'F1':>8}")
    print("-" * 90)

    rows = []
    conditions: list[tuple[float | None, str]] = [(s, f"SNR={s}dB") for s in SNR_LEVELS]
    conditions.append((None, "clean (no noise)"))
    for snr, snr_label in conditions:
        for use_dn in (False, True):
            label = f"{snr_label}, dn={'on' if use_dn else 'off'}"
            res = evaluate(encoder, base_provider, snr, use_dn, cfg, device)
            rows.append({
                "snr_db": snr if snr is not None else "clean",
                "denoiser": "on" if use_dn else "off",
                **{k: res[k] for k in ["auc", "eer", "frr_at_far", "open_set_acc_at_far", "keyword_acc", "f1"]},
            })
            print(
                f"{label:<26} | {res['auc']:>8.4f} | {res['eer']:>8.4f} | "
                f"{res['open_set_acc_at_far']:>10.4f} | {res['keyword_acc']:>8.4f} | {res['f1']:>8.4f}"
            )

    print("=" * 90)
    output_path = Path("results/denoiser_ablation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
