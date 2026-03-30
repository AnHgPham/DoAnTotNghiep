"""DSCNN encoder following Rusci et al. architecture exactly.

DSCNN-L: 276 channels, 1 initial conv + 5 DS blocks, embedding_dim=276.
DSCNN-S: 64 channels, 1 initial conv + 4 DS blocks, embedding_dim=64.

Key implementation details matching Rusci source code:
- Padding via nn.ZeroPad2d (asymmetric), NOT Conv2d padding parameter
- All Conv2d use bias=True (PyTorch default)
- Last DS block uses LayerNorm([C, H, W], elementwise_affine=False) on 3D shape
- L2 normalization is NOT applied inside the model
"""

import math

import torch
import torch.nn as nn

# fmt: off
MODEL_SIZE_INFO = {
    "L": [
        6, 276,
        10, 4, 2, 1,       # Initial conv: kernel=(10,4), stride=(2,1)
        276, 3, 3, 2, 2,   # DS Block 1: 276ch, kernel=3x3, stride=(2,2)
        276, 3, 3, 1, 1,   # DS Block 2: stride=(1,1)
        276, 3, 3, 1, 1,   # DS Block 3
        276, 3, 3, 1, 1,   # DS Block 4
        276, 3, 3, 1, 1,   # DS Block 5
    ],
    "S": [
        5, 64,
        10, 4, 2, 2,       # Initial conv: kernel=(10,4), stride=(2,2) -- NOTE: S uses (2,2) not (2,1)
        64, 3, 3, 2, 2,    # DS Block 1
        64, 3, 3, 1, 1,    # DS Block 2
        64, 3, 3, 1, 1,    # DS Block 3
        64, 3, 3, 1, 1,    # DS Block 4
    ],
}
# fmt: on


def _compute_padding(
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> tuple[int, int, int, int]:
    """Compute asymmetric padding for 'SAME' convolution (Rusci style).

    Returns:
        (left, right, top, bottom) padding for nn.ZeroPad2d.
    """
    out_h = math.ceil(input_h / stride_h)
    out_w = math.ceil(input_w / stride_w)

    pad_h = max((out_h - 1) * stride_h + kernel_h - input_h, 0)
    pad_w = max((out_w - 1) * stride_w + kernel_w - input_w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)


def _conv_output_size(input_size: int, kernel: int, stride: int, pad_total: int) -> int:
    """Compute output dimension after conv with given total padding."""
    return (input_size + pad_total - kernel) // stride + 1


class DSCNN(nn.Module):
    """Depthwise Separable CNN for keyword spotting.

    Builds the network dynamically from MODEL_SIZE_INFO, computing padding
    and spatial dimensions at each layer to match Rusci's implementation.

    Args:
        model_size: "L" (276ch, 5 DS blocks) or "S" (64ch, 4 DS blocks).
        feature_mode: "CONV", "RELU", or "NORM". Metadata for caller;
            all modes return raw output (L2-norm applied externally).
        input_shape: (H, W) of input feature map. Default (49, 10) for MFCC.
    """

    def __init__(
        self,
        model_size: str = "L",
        feature_mode: str = "NORM",
        input_shape: tuple[int, int] = (49, 10),
    ):
        super().__init__()

        if model_size not in MODEL_SIZE_INFO:
            raise ValueError(f"model_size must be 'L' or 'S', got '{model_size}'")

        self.model_size = model_size
        self.feature_mode = feature_mode

        info = MODEL_SIZE_INFO[model_size]
        num_layers = info[0]
        channels = info[1]
        self.embedding_dim = channels

        h, w = input_shape

        layers: list[nn.Module] = []

        # --- Initial Conv ---
        init_kh, init_kw = info[2], info[3]
        init_sh, init_sw = info[4], info[5]

        pad = _compute_padding(h, w, init_kh, init_kw, init_sh, init_sw)
        layers.append(nn.ZeroPad2d(pad))

        h = _conv_output_size(h, init_kh, init_sh, pad[2] + pad[3])
        w = _conv_output_size(w, init_kw, init_sw, pad[0] + pad[1])

        layers.append(nn.Conv2d(1, channels, kernel_size=(init_kh, init_kw),
                                stride=(init_sh, init_sw), padding=0))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))

        # --- DS Blocks ---
        num_ds_blocks = num_layers - 1
        for i in range(num_ds_blocks):
            offset = 6 + i * 5
            block_ch = info[offset]
            kh, kw = info[offset + 1], info[offset + 2]
            sh, sw = info[offset + 3], info[offset + 4]
            is_last = (i == num_ds_blocks - 1)

            # Depthwise conv
            pad = _compute_padding(h, w, kh, kw, sh, sw)
            layers.append(nn.ZeroPad2d(pad))

            h = _conv_output_size(h, kh, sh, pad[2] + pad[3])
            w = _conv_output_size(w, kw, sw, pad[0] + pad[1])

            layers.append(nn.Conv2d(block_ch, block_ch,
                                    kernel_size=(kh, kw), stride=(sh, sw),
                                    padding=0, groups=block_ch))
            layers.append(nn.BatchNorm2d(block_ch))
            layers.append(nn.ReLU(inplace=True))

            # Pointwise conv
            pw_pad = _compute_padding(h, w, 1, 1, 1, 1)
            if any(p > 0 for p in pw_pad):
                layers.append(nn.ZeroPad2d(pw_pad))
            layers.append(nn.Conv2d(block_ch, block_ch,
                                    kernel_size=1, stride=1, padding=0))

            if is_last:
                layers.append(nn.LayerNorm([block_ch, h, w],
                                           elementwise_affine=False))
            else:
                layers.append(nn.BatchNorm2d(block_ch))
                layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self._final_h = h
        self._final_w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, H, W) MFCC features. Default (B, 1, 49, 10).

        Returns:
            (B, embedding_dim) raw embedding. L2-norm NOT applied.
        """
        x = self.features(x)
        x = self.avg_pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)
        return x
