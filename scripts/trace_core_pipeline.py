"""Print the core KWS tensor flow for study and debugging.

This script uses randomly initialized encoders. Its distances are educational
only and are not evaluation results.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.mel import MelSpectrogramExtractor
from src.features.mfcc import MFCCExtractor
from src.features.pcen import PCEN
from src.models.dscnn import DSCNN
from src.models.edgespot_full import EdgeSpotFull


SAMPLE_RATE = 16_000


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_wave(frequency: float, phase: float = 0.0) -> torch.Tensor:
    time = torch.arange(SAMPLE_RATE, dtype=torch.float32) / SAMPLE_RATE
    waveform = 0.20 * torch.sin(2.0 * math.pi * frequency * time + phase)
    return waveform.unsqueeze(0)


def main() -> None:
    torch.manual_seed(7)
    waveform = make_wave(440.0)
    print("1. waveform:", tuple(waveform.shape), "= (channel, samples)")

    mfcc_extractor = MFCCExtractor()
    mel_extractor = MelSpectrogramExtractor(log=False)
    mfcc = mfcc_extractor.extract(waveform).unsqueeze(0)
    mel = mel_extractor.extract(waveform).unsqueeze(0)
    pcen = PCEN(n_channels=40)(mel)

    print("2. MFCC:", tuple(mfcc.shape), "= (batch, channel, frames, coefficients)")
    print("3. raw mel:", tuple(mel.shape), "= (batch, channel, mel bins, frames)")
    print("4. PCEN:", tuple(pcen.shape), "= same geometry as raw mel")

    dscnn_mfcc = DSCNN(model_size="L", input_shape=(47, 10), use_pcen=False).eval()
    dscnn_pcen = DSCNN(model_size="L", input_shape=(40, 101), use_pcen=True).eval()
    edgespot_pcen = EdgeSpotFull(tau=4, embedding_dim=64, use_pcen=True).eval()

    with torch.no_grad():
        emb_dscnn_mfcc = F.normalize(dscnn_mfcc(mfcc), p=2, dim=-1)
        emb_dscnn_pcen = F.normalize(dscnn_pcen(mel), p=2, dim=-1)
        emb_edgespot_pcen = F.normalize(edgespot_pcen(mel), p=2, dim=-1)

    print(
        "5. DSCNN-L + MFCC embedding:",
        tuple(emb_dscnn_mfcc.shape),
        f"params={parameter_count(dscnn_mfcc):,}",
    )
    print(
        "6. DSCNN-L + PCEN embedding:",
        tuple(emb_dscnn_pcen.shape),
        f"params={parameter_count(dscnn_pcen):,}",
    )
    print(
        "7. EdgeSpotFull T4 + PCEN embedding:",
        tuple(emb_edgespot_pcen.shape),
        f"params={parameter_count(edgespot_pcen):,}",
    )

    support_waves = torch.stack(
        [make_wave(440.0 + index, phase=index * 0.1) for index in range(10)]
    )
    query_wave = make_wave(445.0).unsqueeze(0)
    support_mel = mel_extractor.extract_batch(support_waves)
    query_mel = mel_extractor.extract_batch(query_wave)

    with torch.no_grad():
        support_embeddings = F.normalize(dscnn_pcen(support_mel), p=2, dim=-1)
        prototype = support_embeddings.mean(dim=0)
        query_embedding = F.normalize(dscnn_pcen(query_mel), p=2, dim=-1).squeeze(0)
        distance = torch.dist(query_embedding, prototype, p=2).item()

    print("8. support embeddings:", tuple(support_embeddings.shape), "= 10-shot")
    print("9. keyword prototype:", tuple(prototype.shape), "= mean over support")
    print("10. query-to-prototype L2 distance:", f"{distance:.6f}")
    print("Note: models are random here; the distance is not an accuracy result.")


if __name__ == "__main__":
    main()
