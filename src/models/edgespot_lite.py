"""EdgeSpot-lite encoder for few-shot keyword spotting.

This is a practical reimplementation of the ideas used in the EdgeSpot paper:
40x101 mel input, optional trainable PCEN, fused early temporal blocks, temporal
relative positional convolution, and single-head temporal self-attention.

It intentionally omits the heavyweight Wav2Vec2 teacher distillation path so it
can be trained in the existing project pipeline first.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.pcen import PCEN


def _make_divisible(value: int, divisor: int = 4) -> int:
    return max(divisor, int(round(value / divisor) * divisor))


class SamePadDepthwiseConv1d(nn.Module):
    """Depthwise Conv1d with explicit same padding for even kernels."""

    def __init__(self, channels: int, kernel_size: int = 16):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        return self.conv(F.pad(x, (left, right)))


class FusedTemporalBlock(nn.Module):
    """Early fused block: frequency depthwise path plus regular temporal conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
    ):
        super().__init__()
        freq_dilation, time_dilation = dilation
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(3, 1),
                stride=(stride[0], 1),
                padding=(freq_dilation, 0),
                dilation=(freq_dilation, 1),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 3),
                stride=(1, stride[1]),
                padding=(0, time_dilation),
                dilation=(1, time_dilation),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if stride != (1, 1) or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class BCResLiteBlock(nn.Module):
    """Small BC-ResNet-style depthwise separable residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
    ):
        super().__init__()
        freq_dilation, time_dilation = dilation
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(3, 1),
                stride=(stride[0], 1),
                padding=(freq_dilation, 0),
                dilation=(freq_dilation, 1),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, 3),
                stride=(1, stride[1]),
                padding=(0, time_dilation),
                dilation=(1, time_dilation),
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != (1, 1) or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


def _stack_blocks(
    block_cls: type[nn.Module],
    in_channels: int,
    out_channels: int,
    n_blocks: int,
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[nn.Sequential, int]:
    blocks = [
        block_cls(
            in_channels,
            out_channels,
            stride=stride,
            dilation=dilation,
        ),
    ]
    for _ in range(n_blocks - 1):
        blocks.append(
            block_cls(
                out_channels,
                out_channels,
                stride=(1, 1),
                dilation=dilation,
            ),
        )
    return nn.Sequential(*blocks), out_channels


class EdgeSpotLite(nn.Module):
    """EdgeSpot-inspired mel/PCEN encoder.

    Args:
        width_mult: Width multiplier. ``4`` approximates the paper's largest
            compact variant while still remaining much smaller than DSCNN-L.
        embedding_dim: Output embedding size.
        use_pcen: Enable trainable PCEN frontend.
    """

    def __init__(
        self,
        width_mult: int = 4,
        embedding_dim: int = 64,
        use_pcen: bool = True,
    ):
        super().__init__()
        self.width_mult = int(width_mult)
        self.embedding_dim = int(embedding_dim)
        self.use_pcen = use_pcen

        tau = self.width_mult
        c0 = _make_divisible(16 * tau)
        c1 = _make_divisible(8 * tau)
        c2 = _make_divisible(12 * tau)
        c3 = _make_divisible(16 * tau)
        c4 = _make_divisible(20 * tau)
        c5 = _make_divisible(32 * tau)

        self.pcen = PCEN(n_channels=40, per_channel=False) if use_pcen else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        self.stage1, cur = _stack_blocks(
            FusedTemporalBlock, c0, c1, n_blocks=2, stride=(1, 1), dilation=(1, 1),
        )
        self.stage2, cur = _stack_blocks(
            FusedTemporalBlock, cur, c2, n_blocks=2, stride=(2, 1), dilation=(1, 2),
        )
        self.stage3, cur = _stack_blocks(
            BCResLiteBlock, cur, c3, n_blocks=4, stride=(2, 1), dilation=(1, 4),
        )
        self.stage4, cur = _stack_blocks(
            BCResLiteBlock, cur, c4, n_blocks=4, stride=(1, 1), dilation=(1, 8),
        )

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
            batch_first=True,
        )
        self.attn_act = nn.PReLU()
        self.temporal_head = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, groups=embedding_dim),
            nn.PReLU(),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for input ``(B, 1, 40, 101)`` mel features."""
        x = self.pcen(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.freq_collapse(x).squeeze(2)  # (B, C, T)
        x = x + self.positional(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.to_attention_dim(x)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = self.attn_act(x)
        x = self.temporal_head(x.transpose(1, 2))
        return x.mean(dim=-1)
