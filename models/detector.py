"""
RetinaNet detector backbone with optional Noisy Residual Blocks (NRB).

The paper builds SSMD on top of RetinaNet (Section 3.1).  This module:
  • Uses a ResNet50/101 + FPN backbone from torchvision.
  • Replaces every residual block in the chosen layers with an NRB-wrapped
    version (feature-space perturbation, Section 3.2).
  • Exposes a forward() that returns flat (cls_logits, reg_deltas) tensors
    suitable for the consistency-cost and adversarial-perturbation modules.

Anchor encoding follows Eq. 3:
    p_x = (x - x_a) / w_a,  p_y = (y - y_a) / h_a
    p_w = log(w / w_a),      p_h = log(h / h_a)
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops import sigmoid_focal_loss

from .noisy_residual_block import NoisyResidualBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_nrb(module: nn.Module, gamma: float = 0.9) -> nn.Module:
    """
    Recursively walk a ResNet and wrap every BasicBlock / Bottleneck with a
    NoisyResidualBlock that operates on the block's *output* channels.

    We inject noise after the block's final BN so it acts as a post-residual
    perturbation (matching Fig. 3 in the paper).
    """
    from torchvision.models.resnet import BasicBlock, Bottleneck

    for name, child in list(module.named_children()):
        if isinstance(child, (BasicBlock, Bottleneck)):
            out_channels = (child.conv3.out_channels
                            if isinstance(child, Bottleneck)
                            else child.conv2.out_channels)
            setattr(module, name,
                    _NRBWrappedBlock(child, out_channels, gamma))
        else:
            _inject_nrb(child, gamma)
    return module


class _NRBWrappedBlock(nn.Module):
    """Wraps an existing residual block and appends a NoisyResidualBlock."""

    def __init__(self, block: nn.Module, out_channels: int, gamma: float):
        super().__init__()
        self.block = block
        self.nrb = NoisyResidualBlock(out_channels, gamma=gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nrb(self.block(x))


# ---------------------------------------------------------------------------
# Main Detector
# ---------------------------------------------------------------------------

class SSMDDetector(nn.Module):
    """
    RetinaNet with optional NRB injection.

    Args:
        num_classes   : Number of *foreground* classes (background added internally).
        use_nrb       : Whether to inject Noisy Residual Blocks (student branch).
        nrb_gamma     : γ scale factor for NRB sigmoid gate (paper default 0.9).
        pretrained    : Load ImageNet-pretrained backbone weights.
        score_thresh  : Inference confidence threshold.
    """

    def __init__(
        self,
        num_classes: int = 1,
        use_nrb: bool = False,
        nrb_gamma: float = 0.9,
        pretrained: bool = True,
        score_thresh: float = 0.05,
    ):
        super().__init__()

        # torchvision RetinaNet already includes backbone + FPN + Head
        self.model = retinanet_resnet50_fpn(
            pretrained=pretrained,
            num_classes=num_classes + 1,   # +1 for background
            score_thresh=score_thresh,
        )

        if use_nrb:
            _inject_nrb(self.model.backbone.body, gamma=nrb_gamma)

        self.num_classes = num_classes

    # ------------------------------------------------------------------
    def forward_train(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning *flat* logits and regression deltas for
        the consistency loss.

        Returns:
            cls_logits : [total_anchors, num_classes+1]
            reg_deltas : [total_anchors, 4]
        """
        # Extract image-list features
        image_list, _ = self.model.transform(images, targets)
        features = self.model.backbone(image_list.tensors)

        # FPN outputs ordered dict  {'0':..,'1':..,'2':..,'3':..,'pool':..}
        feature_maps = list(features.values())

        # Head produces per-level outputs
        head_outputs = self.model.head(features)

        cls_logits = head_outputs['cls_logits']   # list[Tensor[B,A*K,H,W]]
        bbox_reg   = head_outputs['bbox_regression']

        # Flatten all levels → [B*total_anchors, K] and [B*total_anchors, 4]
        cls_flat = self._flatten_head_output(cls_logits,
                                             self.num_classes + 1)
        reg_flat = self._flatten_head_output(bbox_reg, 4)

        return cls_flat, reg_flat

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_head_output(
        level_outputs: List[torch.Tensor],
        channels: int,
    ) -> torch.Tensor:
        """
        Flatten spatial + batch dimensions.

        level_outputs: list of [B, A*C, H, W] per FPN level
        Returns:       [B * sum(A*H*W), C]
        """
        flat = []
        for lvl in level_outputs:
            B, AC, H, W = lvl.shape
            A = AC // channels
            # [B, A*C, H, W] → [B, H, W, A, C] → [B*H*W*A, C]
            lvl = lvl.view(B, A, channels, H, W)
            lvl = lvl.permute(0, 3, 4, 1, 2).contiguous()
            lvl = lvl.view(-1, channels)
            flat.append(lvl)
        return torch.cat(flat, dim=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
    ):
        """
        Standard torchvision-style forward.
        • Training (targets given)  → dict of losses
        • Inference (no targets)   → list of detection dicts
        """
        return self.model(images, targets)
