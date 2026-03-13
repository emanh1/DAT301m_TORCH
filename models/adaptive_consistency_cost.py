"""
Adaptive Consistency Cost — Section 3.1 of SSMD paper.

The key idea: instead of treating every spatial proposal equally, weight the
KL-divergence (classification) and MSE (localisation) losses by how much the
proposal looks like a *foreground* object.

Dynamic instance weight (Eq. 4):
    W(p_s^c, p_t^c) = [ (1 - p_s^c[0])^2 + (1 - p_t^c[0])^2 ] / 2

where index 0 is the background class.  Proposals where both networks are
confident about background → weight ≈ 0 (ignored).
Foreground proposals → weight ≈ 1 (heavily regularised).

Total consistency cost (Algorithm 1, lines 11-12):
    L_cons = W ⊗ ( KL(p_s^c ‖ p_t^c) + MSE(p_s^{xywh}, p_t^{xywh}) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConsistencyCost(nn.Module):
    """
    Computes the adaptive consistency loss between student and teacher outputs.

    Args:
        lambda_cls (float): Weight on the KL classification term.
        lambda_reg (float): Weight on the MSE localisation term.
        eps (float): Small value for numerical stability in KL.
    """

    def __init__(self, lambda_cls: float = 1.0,
                 lambda_reg: float = 1.0,
                 eps: float = 1e-8):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.eps = eps

    # ------------------------------------------------------------------
    @staticmethod
    def _instance_weight(p_cls_s: torch.Tensor,
                         p_cls_t: torch.Tensor) -> torch.Tensor:
        """
        Adaptive weight W from Eq. 4.

        Args:
            p_cls_s: Student class probs after softmax  [N, num_classes]
            p_cls_t: Teacher class probs after softmax  [N, num_classes]
        Returns:
            weight: [N, 1]  in range [0, 1]
        """
        bg_s = p_cls_s[:, 0]          # P(background) from student
        bg_t = p_cls_t[:, 0]          # P(background) from teacher
        w = ((1.0 - bg_s) ** 2 + (1.0 - bg_t) ** 2) / 2.0
        return w.unsqueeze(1)          # [N, 1]

    # ------------------------------------------------------------------
    def _kl_loss(self, p_s: torch.Tensor,
                 p_t: torch.Tensor) -> torch.Tensor:
        """
        KL divergence  KL(p_s ‖ p_t) per proposal  →  [N]

        Uses p_t as the target distribution (teacher as pseudo-label).
        """
        p_s = p_s.clamp(min=self.eps)
        p_t = p_t.clamp(min=self.eps)
        # KL(p_s || p_t) = sum p_s * log(p_s / p_t)
        kl = (p_s * (p_s.log() - p_t.log())).sum(dim=-1)   # [N]
        return kl

    # ------------------------------------------------------------------
    @staticmethod
    def _mse_box(reg_s: torch.Tensor,
                 reg_t: torch.Tensor) -> torch.Tensor:
        """
        MSE over all four box deltas (Eq. 5) per proposal  →  [N]

        reg_s / reg_t: [N, 4]  (px, py, pw, ph)
        """
        return F.mse_loss(reg_s, reg_t, reduction='none').sum(dim=-1)  # [N]

    # ------------------------------------------------------------------
    def forward(self,
                cls_s: torch.Tensor, cls_t: torch.Tensor,
                reg_s: torch.Tensor, reg_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_s: Student class logits  [N, num_classes]  (raw, pre-softmax)
            cls_t: Teacher class logits  [N, num_classes]
            reg_s: Student box deltas    [N, 4]
            reg_t: Teacher box deltas    [N, 4]
        Returns:
            Scalar adaptive consistency loss.
        """
        p_s = F.softmax(cls_s, dim=-1)    # [N, C]
        p_t = F.softmax(cls_t, dim=-1)    # [N, C]

        # --- adaptive weight (Eq. 4)
        W = self._instance_weight(p_s, p_t)   # [N, 1]

        # --- KL classification consistency
        kl = self._kl_loss(p_s, p_t)          # [N]

        # --- MSE localisation consistency  (Eq. 5)
        mse = self._mse_box(reg_s, reg_t)     # [N]

        # --- combine  (Algorithm 1, line 11)
        per_proposal = W.squeeze(1) * (self.lambda_cls * kl +
                                       self.lambda_reg * mse)
        return per_proposal.mean()
