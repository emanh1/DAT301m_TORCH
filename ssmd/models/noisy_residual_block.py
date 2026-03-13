"""
Noisy Residual Block (NRB) — Section 3.2 of SSMD paper.

Adds channel-wise, attention-gated Gaussian noise to intermediate feature maps.
The noise magnitude per channel is learned from the features themselves via
global average pooling → 1×1 conv → scaled sigmoid, so shallow layers can
receive wild noise while deeper layers stay more stable.

Architecture (Fig. 3):
  X^l  ──► AvgPool ──► 1×1 Conv ──► γ·sigmoid(·) ──┐
  X^l  ──────────────────── Gaussian noise X^n ──── ⊗ ──► ⊕ ──► X^q
                                                              ▲
                                                             X^l  (residual)
"""

import torch
import torch.nn as nn


class NoisyResidualBlock(nn.Module):
    """
    Wraps any residual block and adds learned, channel-wise Gaussian noise.

    Args:
        in_channels (int): Number of feature-map channels C.
        gamma (float): Scale factor for the sigmoid gate (default 0.9 from paper).
        mu (float): Mean of the Gaussian noise distribution.
        sigma (float): Std of the Gaussian noise distribution.
    """

    def __init__(self, in_channels: int, gamma: float = 0.9,
                 mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma

        # 1×1 conv after channel-wise average pooling  (C×1×1 → C×1×1)
        self.channel_conv = nn.Conv2d(in_channels, in_channels,
                                      kernel_size=1, bias=True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map  [B, C, H, W]
        Returns:
            Noisy output          [B, C, H, W]
        """
        B, C, H, W = x.shape

        # --- channel-wise average pooling → [B, C, 1, 1]
        x_o = x.mean(dim=[2, 3], keepdim=True)           # AvgPool

        # --- 1×1 conv → [B, C, 1, 1]
        x_p = self.channel_conv(x_o)

        # --- scaled sigmoid gate → [B, C, 1, 1]
        gate = torch.sigmoid(self.gamma * x_p)            # Eq. 7 sigmoid part

        # --- sample Gaussian noise → [B, C, H, W]
        x_n = torch.randn_like(x) * self.sigma + self.mu

        # --- channel-wise multiply noise by gate, then add residual  (Eq. 7)
        #     X^q = (X^n ⊗ sigmoid(γ X^p)) ⊕ X^l
        x_q = (x_n * gate) + x                            # gate broadcasts H×W

        return x_q
