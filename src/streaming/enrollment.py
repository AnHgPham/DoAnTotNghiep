"""Robust few-shot enrollment utilities for streaming KWS.

The training pipeline produces an embedding model. This module handles the
deployment-side problem: turning 3-5 user recordings into stable prototypes,
per-keyword thresholds, and quality diagnostics without retraining.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
EPS = 1e-8


def _mono(waveform: torch.Tensor) -> torch.Tensor:
    """Return waveform as a mono ``(1, T)`` float tensor."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.float().detach().cpu()


def pad_or_trim(waveform: torch.Tensor, length: int = SAMPLE_RATE) -> torch.Tensor:
    """Pad or trim right side to a fixed length."""
    waveform = _mono(waveform)
    if waveform.shape[-1] < length:
        return F.pad(waveform, (0, length - waveform.shape[-1]))
    return waveform[..., :length]


def _frame_rms(waveform: torch.Tensor, frame: int, hop: int) -> tuple[torch.Tensor, torch.Tensor]:
    mono = _mono(waveform).squeeze(0)
    if mono.numel() < frame:
        rms = torch.sqrt(torch.mean(mono.pow(2)) + EPS).view(1)
        return rms, torch.tensor([0])

    starts = torch.arange(0, mono.numel() - frame + 1, hop)
    frames = torch.stack([mono[int(s): int(s) + frame] for s in starts])
    rms = torch.sqrt(torch.mean(frames.pow(2), dim=1) + EPS)
    return rms, starts


@dataclass
class AudioQuality:
    """Quality diagnostics for one enrollment utterance."""

    duration_ms: float
    active_ms: float
    rms_dbfs: float
    peak: float
    clipped_fraction: float
    snr_proxy_db: float
    accepted: bool
    reason: str = "ok"


def estimate_quality(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    min_active_ms: int = 120,
    max_active_ms: int = 1400,
) -> AudioQuality:
    """Estimate quality from simple signal statistics.

    The SNR value is a proxy computed from high/low frame-energy quantiles. It
    is not a calibrated physical SNR, but it is useful for rejecting obviously
    bad enrollment recordings.
    """
    wav = _mono(waveform)
    mono = wav.squeeze(0)
    duration_ms = mono.numel() / sample_rate * 1000.0
    peak = float(mono.abs().max().item()) if mono.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(mono.pow(2)) + EPS).item()) if mono.numel() else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, EPS))
    clipped_fraction = float((mono.abs() >= 0.98).float().mean().item()) if mono.numel() else 0.0

    frame = int(sample_rate * 0.03)
    hop = int(sample_rate * 0.01)
    frame_rms, _ = _frame_rms(wav, frame, hop)
    if frame_rms.numel() == 0:
        active_ms = 0.0
        snr_proxy = 0.0
    else:
        max_rms = float(frame_rms.max().item())
        active = frame_rms >= max(0.0005, max_rms * 0.12)
        active_ms = float(active.sum().item() * hop / sample_rate * 1000.0)
        low = float(torch.quantile(frame_rms, 0.20).item())
        high = float(torch.quantile(frame_rms, 0.90).item())
        snr_proxy = 20.0 * math.log10(max(high, EPS) / max(low, EPS))

    accepted = True
    reason = "ok"
    if peak < 0.01 or rms_dbfs < -45.0:
        accepted, reason = False, "too_quiet"
    elif clipped_fraction > 0.01:
        accepted, reason = False, "clipped"
    elif active_ms < min_active_ms:
        accepted, reason = False, "too_short"
    elif active_ms > max_active_ms:
        accepted, reason = False, "too_long"

    return AudioQuality(
        duration_ms=duration_ms,
        active_ms=active_ms,
        rms_dbfs=rms_dbfs,
        peak=peak,
        clipped_fraction=clipped_fraction,
        snr_proxy_db=snr_proxy,
        accepted=accepted,
        reason=reason,
    )


def crop_to_active_region(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    pad_ms: int = 160,
    min_length_ms: int = 450,
) -> tuple[torch.Tensor, tuple[int, int], AudioQuality]:
    """Crop around active speech using frame energy.

    Returns ``(cropped_waveform, (start_sample, end_sample), quality)``. If no
    reliable active region is found, the original waveform is returned.
    """
    wav = _mono(waveform)
    quality = estimate_quality(wav, sample_rate=sample_rate)
    mono = wav.squeeze(0)
    if mono.numel() == 0:
        return wav, (0, 0), quality

    frame = int(sample_rate * 0.03)
    hop = int(sample_rate * 0.01)
    rms, starts = _frame_rms(wav, frame, hop)
    if rms.numel() == 0:
        return wav, (0, mono.numel()), quality

    max_rms = float(rms.max().item())
    if max_rms <= 1e-6:
        return wav, (0, mono.numel()), quality

    active = torch.where(rms >= max(0.0005, max_rms * 0.12))[0]
    if active.numel() == 0:
        return wav, (0, mono.numel()), quality

    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, int(starts[int(active[0])].item()) - pad)
    end = min(mono.numel(), int(starts[int(active[-1])].item()) + frame + pad)

    min_len = int(sample_rate * min_length_ms / 1000)
    if end - start < min_len:
        center = (start + end) // 2
        start = max(0, center - min_len // 2)
        end = min(mono.numel(), start + min_len)
        start = max(0, end - min_len)

    return wav[..., start:end], (start, end), quality


def make_enrollment_views(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    max_views: int = 5,
    shift_ms: int = 80,
) -> list[torch.Tensor]:
    """Create deterministic inference-time augmentations for one utterance."""
    base = pad_or_trim(waveform, sample_rate)
    views = [base]
    shift = int(sample_rate * shift_ms / 1000)
    if shift > 0:
        views.append(F.pad(base[..., shift:], (0, shift)))
        views.append(F.pad(base[..., :-shift], (shift, 0)))
    views.append((base * 0.80).clamp(-1.0, 1.0))
    views.append((base * 1.20).clamp(-1.0, 1.0))
    return views[:max(1, max_views)]


class EmbeddingBackend:
    """Small wrapper around encoder + feature extractor."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        feature_extractor,
        device: torch.device | str | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.device = torch.device(device or "cpu")
        self.sample_rate = sample_rate
        self.encoder.to(self.device).eval()

    @torch.no_grad()
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = pad_or_trim(waveform, self.sample_rate)
        features = self.feature_extractor.extract(wav).unsqueeze(0).to(self.device)
        emb = self.encoder(features)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.squeeze(0).cpu()

    @torch.no_grad()
    def embed_many(self, waveforms: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack([self.embed(w) for w in waveforms])

    @torch.no_grad()
    def embed_batch(self, waveforms: Sequence[torch.Tensor]) -> torch.Tensor:
        """Embed several waveforms in a single batched encoder forward.

        Numerically identical to calling :meth:`embed` on each waveform
        (encoder is in eval mode, so per-sample outputs are batch-invariant),
        but amortizes feature-extraction and forward-pass overhead.
        Returns a ``(N, d)`` tensor of L2-normalized embeddings on CPU.
        """
        if not waveforms:
            return torch.empty(0)
        feats = [
            self.feature_extractor.extract(pad_or_trim(w, self.sample_rate)).unsqueeze(0)
            for w in waveforms
        ]
        batch = torch.cat(feats, dim=0).to(self.device)
        emb = self.encoder(batch)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.cpu()


@dataclass
class KeywordProfile:
    label: str
    prototype: torch.Tensor
    exemplars: torch.Tensor
    threshold: float
    support_distance_mean: float
    support_distance_std: float
    qualities: list[AudioQuality] = field(default_factory=list)


@dataclass
class MatchResult:
    label: str
    detected: bool
    distance: float
    threshold: float
    margin: float
    confidence: float
    second_label: str | None
    distances: dict[str, float]


class EnrollmentProfile:
    """Collection of enrolled keywords with per-word thresholds."""

    def __init__(self, keywords: Mapping[str, KeywordProfile] | None = None):
        self.keywords = dict(keywords or {})

    @property
    def labels(self) -> list[str]:
        return sorted(self.keywords)

    def _distance_to_keyword(
        self,
        embedding: torch.Tensor,
        profile: KeywordProfile,
        prototype_weight: float = 0.70,
    ) -> float:
        emb = F.normalize(embedding.detach().cpu(), p=2, dim=0)
        proto = F.normalize(profile.prototype.detach().cpu(), p=2, dim=0)
        proto_dist = torch.dist(emb, proto, p=2).item()
        if profile.exemplars.numel() == 0:
            return proto_dist
        exemplars = F.normalize(profile.exemplars.detach().cpu(), p=2, dim=-1)
        exemplar_dist = torch.cdist(emb.view(1, -1), exemplars).min().item()
        return prototype_weight * proto_dist + (1.0 - prototype_weight) * exemplar_dist

    def score(
        self,
        embedding: torch.Tensor,
        min_margin: float = 0.05,
        threshold_scale: float = 1.0,
    ) -> MatchResult:
        if not self.keywords:
            return MatchResult("unknown", False, float("inf"), 0.0, 0.0, 0.0, None, {})

        distances = {
            label: self._distance_to_keyword(embedding, profile)
            for label, profile in self.keywords.items()
        }
        ordered = sorted(distances.items(), key=lambda item: item[1])
        best_label, best_dist = ordered[0]
        second_label = ordered[1][0] if len(ordered) > 1 else None
        second_dist = ordered[1][1] if len(ordered) > 1 else best_dist + 2.0
        margin = second_dist - best_dist
        threshold = float(self.keywords[best_label].threshold) * threshold_scale
        detected = best_dist <= threshold and margin >= min_margin
        dist_score = max(0.0, 1.0 - best_dist / max(threshold, EPS))
        margin_score = max(0.0, min(1.0, margin / 0.50))
        confidence = 0.75 * dist_score + 0.25 * margin_score
        return MatchResult(
            label=best_label if detected else "unknown",
            detected=detected,
            distance=float(best_dist),
            threshold=threshold,
            margin=float(margin),
            confidence=float(confidence),
            second_label=second_label,
            distances={k: float(v) for k, v in distances.items()},
        )

    def to_dict(self) -> dict:
        return {
            "keywords": {
                label: {
                    "prototype": profile.prototype.tolist(),
                    "exemplars": profile.exemplars.tolist(),
                    "threshold": profile.threshold,
                    "support_distance_mean": profile.support_distance_mean,
                    "support_distance_std": profile.support_distance_std,
                    "qualities": [q.__dict__ for q in profile.qualities],
                }
                for label, profile in self.keywords.items()
            }
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "EnrollmentProfile":
        keywords = {}
        for label, item in payload.get("keywords", {}).items():
            qualities = [AudioQuality(**q) for q in item.get("qualities", [])]
            keywords[label] = KeywordProfile(
                label=label,
                prototype=torch.tensor(item["prototype"], dtype=torch.float32),
                exemplars=torch.tensor(item.get("exemplars", []), dtype=torch.float32),
                threshold=float(item["threshold"]),
                support_distance_mean=float(item.get("support_distance_mean", 0.0)),
                support_distance_std=float(item.get("support_distance_std", 0.0)),
                qualities=qualities,
            )
        return cls(keywords)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "EnrollmentProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_enrollment_profile(
    samples: Mapping[str, Sequence[torch.Tensor]],
    backend: EmbeddingBackend,
    impostor_waveforms: Sequence[torch.Tensor] | None = None,
    views_per_sample: int = 5,
    threshold_alpha: float = 2.0,
    threshold_floor: float = 0.35,
    threshold_ceil: float = 1.25,
    target_far: float = 0.01,
) -> EnrollmentProfile:
    """Build robust keyword profiles from a few enrollment examples."""
    impostor_embeddings: torch.Tensor | None = None
    if impostor_waveforms:
        impostor_views = []
        for wav in impostor_waveforms:
            cropped, _, _ = crop_to_active_region(wav, backend.sample_rate)
            impostor_views.append(pad_or_trim(cropped, backend.sample_rate))
        impostor_embeddings = backend.embed_many(impostor_views) if impostor_views else None

    keyword_profiles: dict[str, KeywordProfile] = {}
    for label, wavs in samples.items():
        views: list[torch.Tensor] = []
        qualities: list[AudioQuality] = []
        for wav in wavs:
            cropped, _, quality = crop_to_active_region(wav, backend.sample_rate)
            qualities.append(quality)
            views.extend(make_enrollment_views(cropped, backend.sample_rate, max_views=views_per_sample))

        if not views:
            continue

        exemplars = backend.embed_many(views)
        prototype = F.normalize(exemplars.mean(dim=0), p=2, dim=0)
        support_dists = torch.cdist(exemplars, prototype.view(1, -1)).squeeze(1)
        mean_d = float(support_dists.mean().item())
        std_d = float(support_dists.std(unbiased=False).item()) if support_dists.numel() > 1 else 0.0
        threshold = mean_d + threshold_alpha * max(std_d, 1e-3)

        if impostor_embeddings is not None and impostor_embeddings.numel() > 0:
            impostor_dists = torch.cdist(impostor_embeddings, prototype.view(1, -1)).squeeze(1)
            impostor_thr = float(torch.quantile(impostor_dists, min(max(target_far, 0.0), 1.0)).item())
            threshold = min(threshold, impostor_thr)

        threshold = max(threshold_floor, min(threshold_ceil, threshold))
        keyword_profiles[label] = KeywordProfile(
            label=label,
            prototype=prototype,
            exemplars=exemplars,
            threshold=threshold,
            support_distance_mean=mean_d,
            support_distance_std=std_d,
            qualities=qualities,
        )

    return EnrollmentProfile(keyword_profiles)
