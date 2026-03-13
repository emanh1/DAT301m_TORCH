"""
Exponential Moving Average (EMA) teacher update — Eq. 2 of the SSMD paper.

    θ_t^n = α · θ_t^{n-1} + (1 - α) · θ_s^n

The teacher network is *never* updated by gradient descent; it only receives
EMA updates from the student.  This produces smoother, more stable predictions
which serve as better consistency targets.

Paper ablation (Table 6): α = 0.99 is optimal.
"""

import copy
import torch
import torch.nn as nn


class EMATeacher:
    """
    Maintains an EMA copy of a student network (the teacher).

    Usage::

        student = SSMDDetector(use_nrb=True)
        teacher = EMATeacher(student, alpha=0.99)

        # inside the training loop:
        teacher.update(student)        # after each optimiser step
        pred = teacher.model(images)   # use teacher for consistency targets

    Args:
        student_model : The student nn.Module to shadow.
        alpha         : EMA decay factor (paper default 0.99, Eq. 2).
    """

    def __init__(self, student_model: nn.Module, alpha: float = 0.99):
        self.alpha = alpha
        # Deep-copy student to get identical initial weights  (θ_t^0 = θ_s^0)
        self.model = copy.deepcopy(student_model)
        self.model.eval()

        # Freeze teacher: no gradients, never updated by an optimiser
        for param in self.model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, student_model: nn.Module) -> None:
        """
        EMA update step (Eq. 2):  θ_t ← α·θ_t + (1-α)·θ_s

        Call this *after* every student optimiser step.

        Args:
            student_model: The student model whose current weights are θ_s^n.
        """
        for t_param, s_param in zip(self.model.parameters(),
                                    student_model.parameters()):
            t_param.data.mul_(self.alpha).add_(s_param.data,
                                               alpha=1.0 - self.alpha)

        # Also keep BN running stats in sync (not shown in paper but important)
        for t_buf, s_buf in zip(self.model.buffers(),
                                student_model.buffers()):
            t_buf.data.copy_(s_buf.data)

    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """Delegate inference calls directly to the teacher model."""
        return self.model(*args, **kwargs)
