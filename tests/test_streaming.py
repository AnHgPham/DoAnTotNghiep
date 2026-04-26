"""Tests for the streaming/VAD module (EXT-2)."""

import pytest
import torch
import torch.nn.functional as F


class TestStreamingKWS:
    """Tests for StreamingKWS without VAD (unit-testable without model download)."""

    def _make_mock_encoder(self):
        """Create a mock encoder that returns fixed-size embeddings."""
        from src.models.dscnn import DSCNN
        encoder = DSCNN(model_size="L")
        encoder.eval()
        return encoder

    def _make_mock_extractor(self):
        from src.features.mfcc import MFCCExtractor
        return MFCCExtractor()

    def test_process_file_no_vad(self):
        from src.streaming.vad_engine import StreamingKWS

        encoder = self._make_mock_encoder()
        extractor = self._make_mock_extractor()

        engine = StreamingKWS(
            encoder=encoder,
            mfcc_extractor=extractor,
            vad=None,
            window_size=16000,
            stride=8000,
        )

        waveform = torch.randn(1, 48000)  # 3 seconds

        proto = torch.randn(276)
        proto = F.normalize(proto, p=2, dim=0)
        prototypes = {"test_word": proto}

        results = engine.process_file(waveform, prototypes, threshold=2.0)

        assert len(results) > 0
        for r in results:
            assert "t_start" in r
            assert "t_end" in r
            assert "label" in r
            assert "distance" in r
            assert "detected" in r
            assert "speech_prob" in r
            assert r["is_speech"] is True  # no VAD = always speech

    def test_process_file_window_count(self):
        from src.streaming.vad_engine import StreamingKWS

        encoder = self._make_mock_encoder()
        extractor = self._make_mock_extractor()

        engine = StreamingKWS(
            encoder=encoder,
            mfcc_extractor=extractor,
            vad=None,
            window_size=16000,
            stride=8000,
        )

        waveform = torch.randn(1, 32000)  # 2 seconds
        proto = F.normalize(torch.randn(276), p=2, dim=0)
        results = engine.process_file(waveform, {"word": proto}, threshold=2.0)

        expected_windows = (32000 - 16000) // 8000 + 1  # = 3
        assert len(results) == expected_windows

    def test_process_file_empty_prototypes(self):
        from src.streaming.vad_engine import StreamingKWS

        encoder = self._make_mock_encoder()
        extractor = self._make_mock_extractor()

        engine = StreamingKWS(
            encoder=encoder, mfcc_extractor=extractor,
            vad=None, window_size=16000, stride=8000,
        )

        waveform = torch.randn(1, 32000)
        results = engine.process_file(waveform, {}, threshold=1.0)

        assert len(results) > 0
        assert all(not r["detected"] for r in results)

    def test_process_chunk_buffering(self):
        from src.streaming.vad_engine import StreamingKWS

        encoder = self._make_mock_encoder()
        extractor = self._make_mock_extractor()

        engine = StreamingKWS(
            encoder=encoder, mfcc_extractor=extractor,
            vad=None, window_size=16000, stride=8000,
        )

        proto = F.normalize(torch.randn(276), p=2, dim=0)

        result = engine.process_chunk(torch.randn(8000), {"word": proto}, threshold=2.0)
        assert result is None  # buffer not full yet

        result = engine.process_chunk(torch.randn(8000), {"word": proto}, threshold=2.0)
        assert result is not None  # buffer now has 16000 samples

    def test_reset(self):
        from src.streaming.vad_engine import StreamingKWS

        encoder = self._make_mock_encoder()
        extractor = self._make_mock_extractor()

        engine = StreamingKWS(
            encoder=encoder, mfcc_extractor=extractor,
            vad=None, window_size=16000, stride=8000,
        )

        engine.process_chunk(torch.randn(8000), {}, threshold=1.0)
        engine.reset()
        assert len(engine._buffer) == 0
