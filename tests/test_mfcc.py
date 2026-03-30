"""Tests for MFCCExtractor."""

import torch
from src.features.mfcc import MFCCExtractor


def test_mfcc_shape():
    ext = MFCCExtractor()
    wav = torch.randn(1, 16000)
    mfcc = ext.extract(wav)
    assert mfcc.shape == (1, 47, 10)


def test_mfcc_short_audio():
    ext = MFCCExtractor()
    wav = torch.randn(1, 8000)
    mfcc = ext.extract(wav)
    assert mfcc.shape == (1, 47, 10)


def test_mfcc_long_audio():
    ext = MFCCExtractor()
    wav = torch.randn(1, 24000)
    mfcc = ext.extract(wav)
    assert mfcc.shape == (1, 47, 10)


def test_mfcc_batch():
    ext = MFCCExtractor()
    wavs = torch.randn(4, 1, 16000)
    mfcc = ext.extract_batch(wavs)
    assert mfcc.shape == (4, 1, 47, 10)


def test_mfcc_num_features():
    ext = MFCCExtractor(num_features=10)
    wav = torch.randn(1, 16000)
    mfcc = ext.extract(wav)
    assert mfcc.shape[-1] == 10


def test_mfcc_deterministic():
    ext = MFCCExtractor()
    wav = torch.randn(1, 16000)
    mfcc1 = ext.extract(wav)
    mfcc2 = ext.extract(wav)
    assert torch.equal(mfcc1, mfcc2)


def test_mfcc_1d_input():
    ext = MFCCExtractor()
    wav = torch.randn(16000)
    mfcc = ext.extract(wav)
    assert mfcc.shape == (1, 47, 10)
