"""Wav2Vec2 teacher wrapper used for offline embedding precompute."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Wav2Vec2Teacher(nn.Module):
    """Frozen Wav2Vec2 encoder with a projection head.

    The projection head can be loaded from a checkpoint after teacher training.
    Without a checkpoint it is randomly initialized, which is acceptable only
    for smoke tests and pipeline validation, not for final KD experiments.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        layer: int = 16,
        embedding_dim: int = 64,
        head_checkpoint: str | None = None,
    ):
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "Wav2Vec2 teacher precompute requires transformers. "
                "Install with: pip install transformers"
            ) from exc

        self.model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.layer = int(layer)
        hidden_size = int(self.model.config.hidden_size)
        self.proj = nn.Linear(hidden_size, embedding_dim)
        self.embedding_dim = int(embedding_dim)

        if head_checkpoint:
            state = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
            state_dict = state.get("projection_state_dict", state)
            self.proj.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Embed waveform batch shaped ``(B, T)`` or ``(B, 1, T)``."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        out = self.model(waveform, output_hidden_states=True)
        hidden_states = out.hidden_states
        layer = min(max(self.layer, 0), len(hidden_states) - 1)
        hidden = hidden_states[layer]
        pooled = hidden.mean(dim=1)
        return F.normalize(self.proj(pooled), p=2, dim=-1)
