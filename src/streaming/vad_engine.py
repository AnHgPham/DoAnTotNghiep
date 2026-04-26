"""Streaming keyword spotting engine with Silero VAD (EXT-2).

Implements a sliding-window keyword detection pipeline gated by
Voice Activity Detection. Only windows containing speech are
processed by the KWS encoder, reducing false activations and
computation on silence/noise segments.
"""

import logging
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class SileroVAD:
    """Silero Voice Activity Detection wrapper.

    Downloads the Silero VAD model on first use (~2 MB).

    Args:
        threshold: Speech probability threshold (0.0–1.0).
        min_speech_ms: Minimum speech duration to consider valid.
        device: Torch device.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        device: torch.device | None = None,
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.device = device or torch.device("cpu")

        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Silero VAD loaded (threshold=%.2f)", threshold)

    def is_speech(self, waveform: torch.Tensor) -> tuple[bool, float]:
        """Check whether a waveform chunk contains speech.

        Args:
            waveform: (T,) or (1, T) audio tensor at 16kHz.

        Returns:
            (contains_speech, speech_probability) tuple.
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.to(self.device)

        with torch.no_grad():
            prob = self.model(waveform, SAMPLE_RATE).item()

        return prob >= self.threshold, prob

    def get_speech_timestamps(
        self, waveform: torch.Tensor
    ) -> list[dict[str, int]]:
        """Get speech segments with start/end sample indices.

        Args:
            waveform: (T,) or (1, T) audio tensor at 16kHz.

        Returns:
            List of dicts with 'start' and 'end' sample indices.
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        get_speech_timestamps = self.utils[0]
        timestamps = get_speech_timestamps(
            waveform.to(self.device),
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            sampling_rate=SAMPLE_RATE,
        )
        return timestamps

    def reset_states(self) -> None:
        """Reset VAD internal states for a new audio stream."""
        self.model.reset_states()


class StreamingKWS:
    """Streaming keyword spotting with VAD gating.

    Processes audio through a sliding window, using VAD to skip
    non-speech segments before running the KWS encoder.

    Args:
        encoder: DSCNN model (eval mode).
        mfcc_extractor: MFCCExtractor instance.
        vad: SileroVAD instance (None to disable VAD gating).
        window_size: Detection window in samples (default 16000 = 1s).
        stride: Window stride in samples (default 8000 = 0.5s).
        device: Torch device for encoder inference.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        mfcc_extractor: Any,
        vad: SileroVAD | None = None,
        window_size: int = 16000,
        stride: int = 8000,
        device: torch.device | None = None,
    ):
        self.encoder = encoder
        self.mfcc_extractor = mfcc_extractor
        self.vad = vad
        self.window_size = window_size
        self.stride = stride
        self.device = device or torch.device("cpu")

        self._buffer: deque[float] = deque(maxlen=window_size * 4)

    def _extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embedding from a 1-second waveform."""
        if waveform.shape[-1] < SAMPLE_RATE:
            waveform = F.pad(waveform, (0, SAMPLE_RATE - waveform.shape[-1]))
        elif waveform.shape[-1] > SAMPLE_RATE:
            waveform = waveform[..., :SAMPLE_RATE]

        mfcc = self.mfcc_extractor.extract(waveform)  # (1, 47, 10)
        mfcc = mfcc.unsqueeze(0).to(self.device)       # (1, 1, 47, 10)

        with torch.no_grad():
            embedding = self.encoder(mfcc)              # (1, 276)
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.squeeze(0).cpu()               # (276,)

    def process_file(
        self,
        waveform: torch.Tensor,
        prototypes: dict[str, torch.Tensor],
        threshold: float = 0.8,
    ) -> list[dict]:
        """Process an audio file with sliding window + VAD.

        Args:
            waveform: (1, T) or (T,) full audio tensor at 16kHz.
            prototypes: Dict mapping keyword label -> (D,) prototype embedding.
            threshold: L2 distance threshold for keyword acceptance.

        Returns:
            List of detection dicts with keys: t_start, t_end, label,
            distance, detected, speech_prob, is_speech.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        results = []

        if self.vad is not None:
            self.vad.reset_states()

        pos = 0
        while pos + self.window_size <= total_samples:
            segment = waveform[..., pos : pos + self.window_size]
            t_start = pos / SAMPLE_RATE
            t_end = (pos + self.window_size) / SAMPLE_RATE

            speech_prob = 1.0
            is_speech = True

            if self.vad is not None:
                is_speech, speech_prob = self.vad.is_speech(segment)

            if is_speech and prototypes:
                emb = self._extract_embedding(segment)

                min_dist = float("inf")
                min_label = "unknown"
                for label, proto in prototypes.items():
                    d = torch.dist(emb, proto, p=2).item()
                    if d < min_dist:
                        min_dist = d
                        min_label = label

                detected = min_dist <= threshold
                results.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "label": min_label if detected else "—",
                    "distance": min_dist,
                    "detected": detected,
                    "speech_prob": speech_prob,
                    "is_speech": True,
                })
            else:
                results.append({
                    "t_start": t_start,
                    "t_end": t_end,
                    "label": "—",
                    "distance": float("inf"),
                    "detected": False,
                    "speech_prob": speech_prob,
                    "is_speech": is_speech,
                })

            pos += self.stride

        return results

    def process_chunk(
        self,
        chunk: torch.Tensor,
        prototypes: dict[str, torch.Tensor],
        threshold: float = 0.8,
    ) -> dict | None:
        """Process a single streaming chunk, buffering internally.

        Args:
            chunk: (T,) audio samples to append to the buffer.
            prototypes: Keyword prototypes.
            threshold: Detection threshold.

        Returns:
            Detection result dict if a full window was processed, else None.
        """
        self._buffer.extend(chunk.tolist())

        if len(self._buffer) < self.window_size:
            return None

        window = torch.tensor(list(self._buffer)[-self.window_size :]).unsqueeze(0)

        speech_prob = 1.0
        is_speech = True
        if self.vad is not None:
            is_speech, speech_prob = self.vad.is_speech(window)

        if not is_speech or not prototypes:
            return {
                "label": "—",
                "distance": float("inf"),
                "detected": False,
                "speech_prob": speech_prob,
                "is_speech": is_speech,
            }

        emb = self._extract_embedding(window)
        min_dist = float("inf")
        min_label = "unknown"
        for label, proto in prototypes.items():
            d = torch.dist(emb, proto, p=2).item()
            if d < min_dist:
                min_dist = d
                min_label = label

        detected = min_dist <= threshold
        return {
            "label": min_label if detected else "—",
            "distance": min_dist,
            "detected": detected,
            "speech_prob": speech_prob,
            "is_speech": True,
        }

    def reset(self) -> None:
        """Reset internal buffer and VAD state."""
        self._buffer.clear()
        if self.vad is not None:
            self.vad.reset_states()
