"""Speaker verification gate using ECAPA-TDNN (Optional Extension).

Verifies that the detected keyword was spoken by an enrolled speaker,
adding a speaker identity check on top of keyword spotting.
Uses SpeechBrain's pre-trained ECAPA-TDNN model.
"""

import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class SpeakerVerifier:
    """Speaker verification using ECAPA-TDNN embeddings.

    Downloads the pre-trained model on first use (~80 MB).

    Args:
        model_source: HuggingFace model ID or local path.
        save_dir: Directory to cache downloaded model.
        threshold: Cosine similarity threshold for speaker acceptance.
        device: Torch device for inference.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        save_dir: str = "pretrained_models/ecapa-tdnn",
        threshold: float = 0.25,
        device: torch.device | None = None,
    ):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "speechbrain is required for SpeakerVerifier. "
                "Install with: pip install speechbrain"
            ) from exc

        self.device = device or torch.device("cpu")
        self.threshold = threshold

        self.model = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=save_dir,
            run_opts={"device": str(self.device)},
        )
        logger.info("ECAPA-TDNN speaker model loaded from %s", model_source)

        self.enrolled_embedding: torch.Tensor | None = None
        self.enrolled_samples: list[torch.Tensor] = []

    def _extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from waveform.

        Args:
            waveform: (1, T) audio tensor at 16kHz.

        Returns:
            (1, D) speaker embedding tensor.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        embedding = self.model.encode_batch(waveform)
        return embedding.squeeze(1).cpu()  # (1, D)

    def enroll(self, waveform: torch.Tensor) -> int:
        """Enroll a speaker sample.

        Args:
            waveform: (1, T) audio tensor at 16kHz.

        Returns:
            Number of enrolled samples so far.
        """
        emb = self._extract_embedding(waveform)
        self.enrolled_samples.append(emb)
        stacked = torch.cat(self.enrolled_samples, dim=0)  # (N, D)
        self.enrolled_embedding = stacked.mean(dim=0, keepdim=True)  # (1, D)
        return len(self.enrolled_samples)

    def clear(self) -> None:
        """Remove all enrolled speaker samples."""
        self.enrolled_samples.clear()
        self.enrolled_embedding = None

    def verify(self, waveform: torch.Tensor) -> tuple[bool, float]:
        """Verify whether waveform matches the enrolled speaker.

        Args:
            waveform: (1, T) audio tensor at 16kHz.

        Returns:
            (is_same_speaker, cosine_similarity) tuple.
        """
        if self.enrolled_embedding is None:
            return True, 1.0  # no speaker enrolled = always accept

        query_emb = self._extract_embedding(waveform)

        similarity = torch.nn.functional.cosine_similarity(
            query_emb, self.enrolled_embedding
        ).item()

        return similarity >= self.threshold, similarity

    @property
    def is_enrolled(self) -> bool:
        """Whether at least one speaker sample has been enrolled."""
        return self.enrolled_embedding is not None

    @property
    def num_samples(self) -> int:
        """Number of enrolled speaker samples."""
        return len(self.enrolled_samples)


class SpeakerGate:
    """Wrapper that gates KWS predictions with speaker verification.

    If the speaker gate is enabled and a speaker is enrolled, keyword
    detections are only accepted when the speaker matches.

    Args:
        enabled: Whether the speaker gate is active.
        threshold: Cosine similarity threshold.
        device: Torch device.
    """

    def __init__(
        self,
        enabled: bool = False,
        threshold: float = 0.25,
        device: torch.device | None = None,
    ):
        self.enabled = enabled
        self._verifier: SpeakerVerifier | None = None
        self._threshold = threshold
        self._device = device

    def _ensure_verifier(self) -> SpeakerVerifier:
        if self._verifier is None:
            self._verifier = SpeakerVerifier(
                threshold=self._threshold, device=self._device
            )
        return self._verifier

    def enroll(self, waveform: torch.Tensor) -> int:
        """Enroll a speaker sample."""
        return self._ensure_verifier().enroll(waveform)

    def clear(self) -> None:
        """Remove enrolled speaker."""
        if self._verifier is not None:
            self._verifier.clear()

    def check(self, waveform: torch.Tensor) -> tuple[bool, float]:
        """Check if waveform passes the speaker gate.

        Returns:
            (accepted, similarity). Always (True, 1.0) if gate is disabled.
        """
        if not self.enabled or self._verifier is None:
            return True, 1.0
        if not self._verifier.is_enrolled:
            return True, 1.0
        return self._verifier.verify(waveform)

    def set_enabled(self, enabled: bool) -> None:
        """Toggle speaker gate on/off."""
        self.enabled = enabled
