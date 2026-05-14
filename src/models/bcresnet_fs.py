"""BC-ResNet few-shot baseline encoder.

This model is intentionally simpler than EdgeSpotFull: it keeps the mel/PCEN
frontend and broadcasted residual blocks, but removes temporal attention. It is
useful as a clean architecture baseline between DSCNN and EdgeSpot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.pcen import PCEN
from src.models.edgespot_lite import BCResLiteBlock, _make_divisible


class BCResNetFS(nn.Module):
    """Compact BC-ResNet-style encoder for few-shot KWS."""

    def __init__(
        self,
        tau: int = 4,
        embedding_dim: int = 64,
        use_pcen: bool = True,
    ):
        super().__init__()
        self.tau = int(tau)
        self.width_mult = self.tau
        self.embedding_dim = int(embedding_dim)
        self.use_pcen = bool(use_pcen)

        width = max(1, self.tau)
        c0 = _make_divisible(16 * width)
        c1 = _make_divisible(8 * width)
        c2 = _make_divisible(12 * width)
        c3 = _make_divisible(16 * width)
        c4 = _make_divisible(24 * width)

        self.pcen = PCEN(n_channels=40, per_channel=False) if use_pcen else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            BCResLiteBlock(c0, c1, stride=(1, 1), dilation=(1, 1)),
            BCResLiteBlock(c1, c1, stride=(1, 1), dilation=(1, 1)),
            BCResLiteBlock(c1, c2, stride=(2, 1), dilation=(1, 2)),
            BCResLiteBlock(c2, c2, stride=(1, 1), dilation=(1, 2)),
            BCResLiteBlock(c2, c3, stride=(2, 1), dilation=(1, 4)),
            BCResLiteBlock(c3, c3, stride=(1, 1), dilation=(1, 4)),
            BCResLiteBlock(c3, c4, stride=(1, 1), dilation=(1, 8)),
            BCResLiteBlock(c4, c4, stride=(1, 1), dilation=(1, 8)),
        )
        self.head = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=(5, 5), padding=(0, 2), groups=c4, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ``(B, 1, 40, 101)`` mel features."""
        x = self.pcen(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x).squeeze(2)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.output(x)
