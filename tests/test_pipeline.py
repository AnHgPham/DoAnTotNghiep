"""End-to-end pipeline integration test.

Tests: WAV (1,16000) -> MFCC (1,47,10) -> DSCNN-L (1,276) -> L2-norm (1,276)
"""

import torch
import torch.nn.functional as F

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN
from src.streaming.enrollment import EmbeddingBackend


def test_full_pipeline():
    wav = torch.randn(1, 16000)

    extractor = MFCCExtractor()
    mfcc = extractor.extract(wav)
    assert mfcc.shape == (1, 47, 10)

    mfcc_batch = mfcc.unsqueeze(0)  # (1, 1, 47, 10)

    model = DSCNN(model_size="L")
    model.eval()
    with torch.no_grad():
        embedding = model(mfcc_batch)
    assert embedding.shape == (1, 276)

    embedding_norm = F.normalize(embedding, p=2, dim=-1)
    assert torch.allclose(
        embedding_norm.norm(dim=-1), torch.ones(1), atol=1e-5
    )


def test_full_pipeline_batch():
    wavs = torch.randn(4, 1, 16000)

    extractor = MFCCExtractor()
    mfcc = extractor.extract_batch(wavs)
    assert mfcc.shape == (4, 1, 47, 10)

    model = DSCNN(model_size="L")
    model.eval()
    with torch.no_grad():
        embeddings = model(mfcc)
    assert embeddings.shape == (4, 276)

    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    norms = embeddings_norm.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_pipeline_dscnn_s():
    wav = torch.randn(1, 16000)
    extractor = MFCCExtractor()
    mfcc = extractor.extract(wav).unsqueeze(0)

    model = DSCNN(model_size="S")
    model.eval()
    with torch.no_grad():
        embedding = model(mfcc)
    assert embedding.shape == (1, 64)


def test_pipeline_different_audio_lengths():
    for length in [8000, 16000, 24000, 32000]:
        wav = torch.randn(1, length)
        extractor = MFCCExtractor()
        mfcc = extractor.extract(wav).unsqueeze(0)
        assert mfcc.shape == (1, 1, 47, 10)

        model = DSCNN(model_size="L")
        model.eval()
        with torch.no_grad():
            embedding = model(mfcc)
        assert embedding.shape == (1, 276)


def test_embedding_backend_batch_matches_single_inference():
    torch.manual_seed(7)
    waveforms = [torch.randn(1, 16000) for _ in range(3)]
    backend = EmbeddingBackend(
        DSCNN(model_size="S"),
        MFCCExtractor(),
        device="cpu",
    )

    batched = backend.embed_batch(waveforms)
    individual = torch.stack([backend.embed(waveform) for waveform in waveforms])

    assert batched.shape == individual.shape == (3, 64)
    assert torch.allclose(batched, individual, atol=1e-5, rtol=1e-5)
