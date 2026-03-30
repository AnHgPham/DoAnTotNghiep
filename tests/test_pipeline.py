"""End-to-end pipeline integration test.

Tests: WAV (1,16000) -> MFCC (1,49,10) -> DSCNN-L (1,276) -> L2-norm (1,276)
"""

import torch
import torch.nn.functional as F

from src.features.mfcc import MFCCExtractor
from src.models.dscnn import DSCNN


def test_full_pipeline():
    wav = torch.randn(1, 16000)

    extractor = MFCCExtractor()
    mfcc = extractor.extract(wav)
    assert mfcc.shape == (1, 49, 10)

    mfcc_batch = mfcc.unsqueeze(0)  # (1, 1, 49, 10)

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
    assert mfcc.shape == (4, 1, 49, 10)

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
        assert mfcc.shape == (1, 1, 49, 10)

        model = DSCNN(model_size="L")
        model.eval()
        with torch.no_grad():
            embedding = model(mfcc)
        assert embedding.shape == (1, 276)
