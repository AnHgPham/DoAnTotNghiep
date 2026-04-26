"""Trainable Per-Channel Energy Normalization (PCEN) frontend.

Replaces static log compression with learned automatic gain control and
root compression, improving noise robustness and cross-domain transfer.

Reference: Wang et al., "Trainable Frontend for Robust and Far-Field KWS" (2016)
"""

import torch
import torch.nn as nn


class PCEN(nn.Module):
    """Trainable PCEN layer.

    Applies causal IIR smoothing followed by adaptive gain control and
    stabilized root compression. All parameters are learnable.

    Args:
        n_channels: Number of frequency channels (e.g. 40 for mel).
        alpha_init: Initial AGC strength.
        delta_init: Initial bias for root compression.
        r_init: Initial root compression exponent.
        s_init: Initial smoothing coefficient.
        eps: Small constant for numerical stability.
        per_channel: If True, learn separate params per channel.
    """

    def __init__(
        self,
        n_channels: int = 40,
        alpha_init: float = 0.96,
        delta_init: float = 2.0,
        r_init: float = 0.5,
        s_init: float = 0.04,
        eps: float = 1e-6,
        per_channel: bool = False,
    ):
        super().__init__()
        self.eps = eps
        shape = (n_channels,) if per_channel else (1,)

        self.log_alpha = nn.Parameter(torch.full(shape, self._inv_sigmoid(alpha_init)))
        self.log_delta = nn.Parameter(torch.full(shape, self._inv_softplus(delta_init)))
        self.log_r = nn.Parameter(torch.full(shape, self._inv_sigmoid(r_init)))
        self.log_s = nn.Parameter(torch.full(shape, self._inv_sigmoid(s_init)))

    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        return -torch.tensor(1.0 / x - 1.0).clamp(min=1e-6).log().item()

    @staticmethod
    def _inv_softplus(x: float) -> float:
        return torch.tensor(x).expm1().clamp(min=1e-6).log().item()

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to mel/spectrogram energies.

        Args:
            E: (B, C, T) or (B, 1, C, T) energy tensor. C = n_channels.

        Returns:
            PCEN-normalized tensor (same shape).
        """
        squeeze_4d = False
        if E.dim() == 4:
            B, _, C, T = E.shape
            E = E.squeeze(1)
            squeeze_4d = True

        alpha = torch.sigmoid(self.log_alpha).unsqueeze(-1)  # (..., 1)
        delta = torch.nn.functional.softplus(self.log_delta).unsqueeze(-1)
        r = torch.sigmoid(self.log_r).unsqueeze(-1)
        s = torch.sigmoid(self.log_s).unsqueeze(-1)

        # Causal IIR smoother: M(t) = (1-s)*M(t-1) + s*E(t)
        M = torch.zeros_like(E[:, :, :1])  # (B, C, 1)
        M_list = []
        for t in range(E.shape[-1]):
            M = (1.0 - s) * M + s * E[:, :, t:t+1]
            M_list.append(M)
        M_smooth = torch.cat(M_list, dim=-1)  # (B, C, T)

        pcen = (E / (self.eps + M_smooth).pow(alpha) + delta).pow(r) - delta.pow(r)

        if squeeze_4d:
            pcen = pcen.unsqueeze(1)
        return pcen
