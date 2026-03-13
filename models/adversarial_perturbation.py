"""
Instance-level Adversarial Perturbation — Section 3.3 of SSMD paper.

Goal: perturb the *input image* to maximise the consistency loss between
student and teacher, focusing perturbation on HIGH-CONFIDENCE foreground
proposals (Eq. 9).

Unlike standard virtual adversarial training (VAT) that treats all pixels
equally, SSMD uses an indicator function:
    1[ Σ p_s^c > τ ]
to zero-out gradient contributions from low-confidence regions, so only
foreground proposals drive the adversarial direction.

Two-pass procedure (Algorithm 1, lines 1-2 + Eq. 9):
  1. Forward student + teacher on the *current* inputs to get consistency loss.
  2. Back-prop through the consistency loss w.r.t. the noise variable r_adv.
  3. Normalise the gradient  →  r_adv = ε · g / ‖g‖₂
  4. Add scaled r_adv to the original image:  Adv(X) = X + ξ · r_adv
"""

import torch
import torch.nn.functional as F


def instance_adversarial_perturbation(
    images: torch.Tensor,
    student_net,
    teacher_net,
    consistency_loss_fn,
    xi: float = 5e-7,
    eps: float = 2.0,
    tau: float = 0.95,
) -> torch.Tensor:
    """
    Generate instance-level adversarial perturbation for a batch of images.

    Args:
        images   : Input images [B, C, H, W]  (teacher branch input)
        student_net: Callable  x → (cls_logits [N,K], reg_deltas [N,4])
        teacher_net: Callable  x → (cls_logits [N,K], reg_deltas [N,4])
        consistency_loss_fn: AdaptiveConsistencyCost instance
        xi       : Scale factor for the initial Gaussian seed (Eq. 8, ξ)
        eps      : Magnitude of the final perturbation  ε  (Eq. 9)
        tau      : Confidence threshold for foreground mask  τ

    Returns:
        perturbed_images: [B, C, H, W]  — Adv.(X) from Eq. 8
    """
    # --- Step 1: initialise r_adv from a normalised Gaussian
    r_adv = torch.randn_like(images)
    r_adv = r_adv / (r_adv.norm(p=2) + 1e-12)   # unit-norm seed
    r_adv = r_adv.detach().requires_grad_(True)

    # --- Step 2: perturbed forward pass  (teacher sees Adv.(X))
    adv_images = images + xi * r_adv

    cls_s, reg_s = student_net(images)      # student on clean images
    cls_t, reg_t = teacher_net(adv_images)  # teacher on perturbed images

    # --- Step 3: build foreground indicator mask  1[ Σ p_s^c > τ ]
    p_s = F.softmax(cls_s, dim=-1)          # [N, K]
    fg_prob = 1.0 - p_s[:, 0]              # P(foreground) = 1 - P(bg)
    mask = (fg_prob > tau).float()          # [N]  indicator

    # Mask the student logits/regs so only foreground drives the gradient
    cls_s_masked = cls_s * mask.unsqueeze(1)
    reg_s_masked = reg_s * mask.unsqueeze(1)

    # --- Step 4: consistency loss for gradient computation
    loss = consistency_loss_fn(cls_s_masked, cls_t,
                               reg_s_masked, reg_t)

    # --- Step 5: compute gradient w.r.t. r_adv  (Eq. 9)
    grad = torch.autograd.grad(loss, r_adv,
                               create_graph=False,
                               retain_graph=False)[0]

    # --- Step 6: normalise and scale  →  r_adv = ε · g / ‖g‖₂
    r_adv_final = eps * grad / (grad.norm(p=2) + 1e-12)

    # --- Step 7: add to original image  Adv.(X) = X + ξ · r_adv
    perturbed = (images + xi * r_adv_final).detach()
    return perturbed
