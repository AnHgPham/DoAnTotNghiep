"""EdgeSpot-style full encoder for few-shot keyword spotting.

This module keeps the implementation practical for the current codebase while
matching the architectural ingredients needed for paper-grade experiments:
40x101 mel input, trainable PCEN, fused early temporal blocks, BC-ResNet-style
blocks, temporal positional depthwise Conv1D, single-head SDPA, and a 64-D
embedding head. MFCC input is supported as an ablation path when PCEN is off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.pcen import PCEN
from src.models.edgespot_lite import (
    BCResLiteBlock,
    FusedTemporalBlock,
    SamePadDepthwiseConv1d,
    _make_divisible,
)


class EdgeSpotFull(nn.Module):
    """EdgeSpot reproduction encoder.

    Args:
        tau: Width multiplier, corresponding to EdgeSpot-1..4.
        embedding_dim: Output embedding dimension.
        use_pcen: Apply trainable PCEN before the convolutional encoder.
    """

    def __init__(
        self,
        tau: int = 4,
        embedding_dim: int = 64,
        use_pcen: bool = True,
        dropout: float = 0.0,
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
        c4 = _make_divisible(20 * width)
        c5 = _make_divisible(32 * width)

        self.pcen = PCEN(n_channels=40, per_channel=False) if use_pcen else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._stage(FusedTemporalBlock, c0, c1, n_blocks=2, stride=(1, 1), dilation=(1, 1))
        self.stage2 = self._stage(FusedTemporalBlock, c1, c2, n_blocks=2, stride=(2, 1), dilation=(1, 2))
        self.stage3 = self._stage(BCResLiteBlock, c2, c3, n_blocks=4, stride=(2, 1), dilation=(1, 4))
        self.stage4 = self._stage(BCResLiteBlock, c3, c4, n_blocks=4, stride=(1, 1), dilation=(1, 8))

        self.freq_collapse = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=(5, 5), padding=(0, 2), groups=c4, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )

        self.positional = SamePadDepthwiseConv1d(c5, kernel_size=16)
        self.to_attention_dim = nn.Linear(c5, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.temporal_head = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, groups=embedding_dim),
            nn.PReLU(),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.PReLU(),
        )
        self.output = nn.Linear(embedding_dim, embedding_dim)

    @staticmethod
    def _stage(
        block_cls: type[nn.Module],
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: tuple[int, int],
        dilation: tuple[int, int],
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            block_cls(in_channels, out_channels, stride=stride, dilation=dilation)
        ]
        for _ in range(n_blocks - 1):
            blocks.append(block_cls(out_channels, out_channels, stride=(1, 1), dilation=dilation))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ``(B, 1, F, T)`` features.

        The original EdgeSpot-style path uses ``(F, T) = (40, 101)`` mel
        features. MFCC ablations use a shorter time axis and can leave more
        than one frequency row after ``freq_collapse``, so we average over the
        remaining frequency dimension instead of assuming it is exactly 1.
        """
        x = self.pcen(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.freq_collapse(x).mean(dim=2)  # (B, C, T)
        x = x + self.positional(x)
        x = x.transpose(1, 2)
        x = self.to_attention_dim(x)
        attn, _ = self.attention(x, x, x, need_weights=False)
        x = self.attn_norm(x + attn)
        x = self.temporal_head(x.transpose(1, 2))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.output(x)
