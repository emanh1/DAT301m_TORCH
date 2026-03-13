"""
Instance-level Adversarial Perturbation — Section 3.3 of SSMD paper.

Memory-efficient rewrite:
  - Student runs under torch.no_grad() to get the foreground mask only.
    Its graph does NOT need to flow back to r_adv.
  - Only the teacher forward pass (which takes adv_images as input) needs
    a gradient graph, and only w.r.t. r_adv — not the full model weights.
  - If every proposal is background (mask all-zero), we fall back to a
    plain random perturbation so training continues without crashing.
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
    Args:
        images             : [B, C, H, W] teacher-branch input (on GPU)
        student_net        : callable x -> (cls_logits [N,K], reg [N,4])
        teacher_net        : callable x -> (cls_logits [N,K], reg [N,4])
        consistency_loss_fn: AdaptiveConsistencyCost instance
        xi                 : seed perturbation scale xi  (Eq. 8)
        eps                : final perturbation magnitude eps  (Eq. 9)
        tau                : foreground confidence threshold tau

    Returns:
        perturbed_images: [B, C, H, W] detached
    """

    # ------------------------------------------------------------------ #
    # Step 1 — student forward (no grad needed, just for the fg mask)
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        cls_s, reg_s = student_net(images)
        p_s     = F.softmax(cls_s, dim=-1)
        fg_prob = 1.0 - p_s[:, 0]
        mask    = (fg_prob > tau).float()            # [N]

    # If all proposals are background, skip adversarial step entirely.
    if mask.sum() == 0:
        noise = torch.randn_like(images)
        noise = noise / (noise.norm(p=2) + 1e-12)
        return (images + xi * eps * noise).detach()

    # ------------------------------------------------------------------ #
    # Step 2 — initialise r_adv seed, gradient attached
    # ------------------------------------------------------------------ #
    r_adv = torch.randn_like(images)
    r_adv = (r_adv / (r_adv.norm(p=2) + 1e-12)).detach().requires_grad_(True)

    # ------------------------------------------------------------------ #
    # Step 3 — teacher forward on perturbed image (graph flows to r_adv)
    #           images.detach() ensures grad ONLY flows through r_adv
    # ------------------------------------------------------------------ #
    adv_images = images.detach() + xi * r_adv
    cls_t, reg_t = teacher_net(adv_images)

    cls_t_masked = cls_t * mask.unsqueeze(1)
    reg_t_masked = reg_t * mask.unsqueeze(1)

    loss = consistency_loss_fn(
        cls_s.detach() * mask.unsqueeze(1),
        cls_t_masked,
        reg_s.detach() * mask.unsqueeze(1),
        reg_t_masked,
    )

    # ------------------------------------------------------------------ #
    # Step 4 — gradient w.r.t. r_adv only
    # ------------------------------------------------------------------ #
    grad = torch.autograd.grad(
        loss, r_adv,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )[0]

    if grad is None or grad.abs().max() < 1e-12:
        noise = torch.randn_like(images)
        noise = noise / (noise.norm(p=2) + 1e-12)
        return (images + xi * eps * noise).detach()

    # ------------------------------------------------------------------ #
    # Step 5 — normalise and apply  Adv(X) = X + eps * g/||g||
    # ------------------------------------------------------------------ #
    r_adv_final = eps * grad / (grad.norm(p=2) + 1e-12)
    return (images.detach() + r_adv_final).detach()