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
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights

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
# Gradient Checkpointing Block Wrapper
# ---------------------------------------------------------------------------

class _CheckpointedBlock(nn.Module):
    """Wraps any residual block to recompute activations during backward.

    Uses torch.utils.checkpoint so the backbone's OrderedDict interface
    (required by FPN) is completely untouched — only individual blocks
    inside each layer are wrapped.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self.block, x, use_reentrant=False)


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
        use_grad_ckpt: bool = True,
    ):
        super().__init__()

        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            trainable_layers=3,
        )

        self.model = RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            score_thresh=score_thresh,
        )

        if use_nrb:
            _inject_nrb(self.model.backbone.body, gamma=nrb_gamma)

        # Gradient checkpointing: wrap each residual block so activations are
        # recomputed during backward instead of stored. This saves ~40% VRAM
        # at the cost of ~20% extra compute. We wrap individual blocks (not
        # the whole layer) so the OrderedDict backbone interface is untouched.
        if use_grad_ckpt:
            self._apply_grad_ckpt(self.model.backbone.body)

        self.num_classes = num_classes

    @staticmethod
    def _apply_grad_ckpt(body: nn.Module) -> None:
        """Wrap every ResNet BasicBlock/Bottleneck with gradient checkpointing."""
        from torchvision.models.resnet import BasicBlock, Bottleneck
        from torch.utils.checkpoint import checkpoint

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(body, layer_name, None)
            if layer is None:
                continue
            for i, block in enumerate(layer):
                if isinstance(block, (BasicBlock, Bottleneck, _NRBWrappedBlock)):
                    # Replace block with a CheckpointedBlock wrapper
                    layer[i] = _CheckpointedBlock(block)

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

        # FPN returns an OrderedDict; head expects a plain list of tensors
        feature_maps = list(features.values())

        # Head produces per-level outputs
        head_outputs = self.model.head(feature_maps)

        cls_logits = head_outputs['cls_logits']    # [B, total_anchors, num_classes]
        bbox_reg   = head_outputs['bbox_regression']  # [B, total_anchors, 4]

        # Already flat from newer torchvision — just merge batch dim
        # cls_logits / bbox_reg are either:
        #   new API: list of [B, H*W*A, C]  per level  OR  single [B, total, C]
        #   old API: list of [B, A*C, H, W] per level
        cls_flat = self._flatten_head_output(cls_logits, self.num_classes)
        reg_flat = self._flatten_head_output(bbox_reg, 4)

        return cls_flat, reg_flat

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_head_output(
        level_outputs,
        channels: int,
    ) -> torch.Tensor:
        """
        Accepts whatever shape torchvision's head returns and produces
        a flat [total_anchors, C] tensor.

        Handles:
          • list of [B, A*C, H, W]  — old torchvision spatial layout
          • list of [B, N, C]        — new torchvision flat layout
          • single [B, N, C] tensor  — concatenated already
        """
        # Normalise to a list
        if isinstance(level_outputs, torch.Tensor):
            level_outputs = [level_outputs]

        flat = []
        for lvl in level_outputs:
            if lvl.dim() == 3:
                # New API: [B, N, C] → [B*N, C]
                B, N, C = lvl.shape
                flat.append(lvl.reshape(B * N, C))
            elif lvl.dim() == 4:
                # Old API: [B, A*C, H, W] → [B*H*W*A, C]
                B, AC, H, W = lvl.shape
                A = AC // channels
                lvl = lvl.view(B, A, channels, H, W)
                lvl = lvl.permute(0, 3, 4, 1, 2).contiguous()
                flat.append(lvl.view(-1, channels))
            else:
                raise ValueError(f"Unexpected head output shape: {lvl.shape}")
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