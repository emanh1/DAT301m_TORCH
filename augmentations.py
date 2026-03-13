"""
Heterogeneous Perturbation Pool — Section 3 / Algorithm 1.

Implements the image-space augmentations used to build the
student and teacher input pairs:

  Student branch: random rotation  →  cutout
  Teacher branch: horizontal flip  →  cutout  →  adversarial perturbation

Also provides a Cutout transform compatible with torchvision pipelines.
"""

from __future__ import annotations
import random
import math
from typing import Tuple, List, Optional

import torch
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Cutout  (Section 3 & 4.5.7)
# ---------------------------------------------------------------------------

class Cutout:
    """
    Randomly mask n_masks square regions of side length mask_size.

    Paper default: n=5, s=70  (448×448 input).

    Args:
        n_masks   : Number of rectangular masks.
        mask_size : Side length of each square mask (pixels).
        fill      : Pixel fill value (default 0).
    """

    def __init__(self, n_masks: int = 5, mask_size: int = 70,
                 fill: float = 0.0):
        self.n_masks = n_masks
        self.mask_size = mask_size
        self.fill = fill

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Float tensor [C, H, W]
        Returns:
            img with masked regions zeroed out.
        """
        _, H, W = img.shape
        img = img.clone()
        for _ in range(self.n_masks):
            # Random top-left corner
            y = random.randint(0, H - 1)
            x = random.randint(0, W - 1)
            y1 = max(0, y - self.mask_size // 2)
            y2 = min(H, y + self.mask_size // 2)
            x1 = max(0, x - self.mask_size // 2)
            x2 = min(W, x + self.mask_size // 2)
            img[:, y1:y2, x1:x2] = self.fill
        return img


# ---------------------------------------------------------------------------
# Student augmentation pipeline
# ---------------------------------------------------------------------------

def student_augment(
    img: torch.Tensor,
    max_rotation_deg: float = 10.0,
    cutout_n: int = 5,
    cutout_s: int = 70,
) -> torch.Tensor:
    """
    Augmentation for the student branch (Algorithm 1, line 1):
        Xs = cutout( Rot.(X) )

    Args:
        img            : Float tensor [C, H, W]
        max_rotation_deg: Maximum rotation angle (±degrees). Paper uses 10°.
        cutout_n       : Number of cutout masks.
        cutout_s       : Cutout mask side length.
    Returns:
        Augmented image [C, H, W]
    """
    # Random rotation  ±max_rotation_deg
    angle = random.uniform(-max_rotation_deg, max_rotation_deg)
    img = TF.rotate(img, angle)

    # Cutout
    img = Cutout(n_masks=cutout_n, mask_size=cutout_s)(img)
    return img


# ---------------------------------------------------------------------------
# Teacher augmentation pipeline  (before adversarial perturbation)
# ---------------------------------------------------------------------------

def teacher_augment_base(
    student_img: torch.Tensor,
    cutout_n: int = 5,
    cutout_s: int = 70,
) -> torch.Tensor:
    """
    Deterministic base augmentation for the teacher branch (Algorithm 1, line 2):
        Xt_base = cutout( Rot.( Flip(X) ) )

    Note: Adversarial perturbation is added separately by
          instance_adversarial_perturbation() from adversarial_perturbation.py.

    Args:
        student_img: The *already rotation-augmented* student image [C, H, W].
        cutout_n   : Number of cutout masks.
        cutout_s   : Cutout mask side length.
    Returns:
        Teacher base image before adversarial noise [C, H, W].
    """
    # Horizontal flip  (teacher sees flipped version of the student's rotation)
    img = TF.hflip(student_img)

    # Cutout
    img = Cutout(n_masks=cutout_n, mask_size=cutout_s)(img)
    return img


# ---------------------------------------------------------------------------
# Batch-level helpers
# ---------------------------------------------------------------------------

def batch_student_augment(
    imgs: List[torch.Tensor],
    max_rotation_deg: float = 10.0,
    cutout_n: int = 5,
    cutout_s: int = 70,
) -> List[torch.Tensor]:
    return [student_augment(img, max_rotation_deg, cutout_n, cutout_s)
            for img in imgs]


def batch_teacher_base_augment(
    student_imgs: List[torch.Tensor],
    cutout_n: int = 5,
    cutout_s: int = 70,
) -> List[torch.Tensor]:
    return [teacher_augment_base(img, cutout_n, cutout_s)
            for img in student_imgs]
