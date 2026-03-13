"""
SSMD Trainer — full implementation of Algorithm 1.

Memory-efficient sequencing:
  - Each forward pass is isolated; intermediate tensors are deleted and
    torch.cuda.empty_cache() is called between phases.
  - Adversarial perturbation uses no_grad for the student pass (done in
    adversarial_perturbation.py) and only builds a graph for r_adv.
  - Consistency teacher passes run under no_grad.
  - Gradient checkpointing is enabled on the backbone if use_grad_ckpt=True.
"""

from __future__ import annotations
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models.detector import SSMDDetector
from .models.adaptive_consistency_cost import AdaptiveConsistencyCost
from .models.adversarial_perturbation import instance_adversarial_perturbation
from .utils.ema_teacher import EMATeacher
from .utils.augmentations import batch_student_augment, batch_teacher_base_augment
from .utils.lambda_schedule import ConsistencyScheduler


def _free(*tensors):
    """Delete tensors and clear CUDA cache."""
    for t in tensors:
        del t
    torch.cuda.empty_cache()


class SSMDTrainer:

    def __init__(
        self,
        num_classes: int = 1,
        device: torch.device = torch.device("cpu"),
        ema_alpha: float = 0.99,
        lr: float = 1e-5,
        total_epochs: int = 100,
        steps_per_epoch: int = 100,
        xi: float = 5e-7,
        adv_eps: float = 2.0,
        tau: float = 0.95,
        cutout_n: int = 5,
        cutout_s: int = 70,
        max_rot_deg: float = 10.0,
        nrb_gamma: float = 0.9,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        use_grad_ckpt: bool = True,
    ):
        self.device = device
        self.xi = xi
        self.adv_eps = adv_eps
        self.tau = tau
        self.cutout_n = cutout_n
        self.cutout_s = cutout_s
        self.max_rot_deg = max_rot_deg

        self.student = SSMDDetector(
            num_classes=num_classes,
            use_nrb=True,
            nrb_gamma=nrb_gamma,
            pretrained=True,
            use_grad_ckpt=use_grad_ckpt,
        ).to(device)

        self.teacher = EMATeacher(self.student, alpha=ema_alpha)
        self.teacher.model.to(device)

        self.consistency_loss = AdaptiveConsistencyCost(
            lambda_cls=lambda_cls,
            lambda_reg=lambda_reg,
        )

        self.optimizer = optim.Adam(self.student.parameters(), lr=lr)

        total_iters = total_epochs * steps_per_epoch
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(0.75 * total_iters)],
            gamma=0.1,
        )
        self.cons_scheduler = ConsistencyScheduler(total_iters)

    # ------------------------------------------------------------------
    def _move_targets(self, targets):
        cleaned = []
        for t in targets:
            t = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
            # Ensure boxes tensor is float32 and labels is int64
            if "boxes" in t:
                t["boxes"] = t["boxes"].float()
            if "labels" in t:
                t["labels"] = t["labels"].long()
            # If no valid boxes remain, give a dummy box so RetinaNet doesn't crash
            if t.get("boxes") is not None and t["boxes"].numel() == 0:
                t["boxes"]  = torch.zeros((0, 4), dtype=torch.float32,
                                          device=self.device)
                t["labels"] = torch.zeros((0,),   dtype=torch.long,
                                          device=self.device)
            cleaned.append(t)
        return cleaned

    # ------------------------------------------------------------------
    def train_step(
        self,
        labeled_imgs: List[torch.Tensor],
        labeled_targets: List[dict],
        unlabeled_imgs: List[torch.Tensor],
    ) -> Dict[str, float]:

        self.student.train()
        self.teacher.model.eval()

        # ── Augment (CPU) ───────────────────────────────────────────────
        lab_s  = batch_student_augment(labeled_imgs,   self.max_rot_deg,
                                       self.cutout_n,  self.cutout_s)
        ulab_s = batch_student_augment(unlabeled_imgs, self.max_rot_deg,
                                       self.cutout_n,  self.cutout_s)
        lab_t_base  = batch_teacher_base_augment(lab_s,  self.cutout_n,
                                                 self.cutout_s)
        ulab_t_base = batch_teacher_base_augment(ulab_s, self.cutout_n,
                                                 self.cutout_s)

        # Move to device
        lab_s   = [x.to(self.device) for x in lab_s]
        ulab_s  = [x.to(self.device) for x in ulab_s]
        lab_t   = [x.to(self.device) for x in lab_t_base]
        ulab_t  = [x.to(self.device) for x in ulab_t_base]
        targets = self._move_targets(labeled_targets)
        del lab_t_base, ulab_t_base

        # ── Adversarial perturbation (no full graph kept) ────────────────
        def _student_fn(x):
            # Called inside no_grad in adversarial_perturbation.py
            return self.student.forward_train(list(x))

        def _teacher_fn(x):
            # Teacher graph only needed w.r.t. r_adv, not model weights
            return self.teacher.model.forward_train(list(x))

        lab_t_stack = torch.stack(lab_t)
        lab_t_adv = instance_adversarial_perturbation(
            lab_t_stack, _student_fn, _teacher_fn,
            self.consistency_loss,
            xi=self.xi, eps=self.adv_eps, tau=self.tau,
        )
        _free(lab_t_stack)

        ulab_t_stack = torch.stack(ulab_t)
        ulab_t_adv = instance_adversarial_perturbation(
            ulab_t_stack, _student_fn, _teacher_fn,
            self.consistency_loss,
            xi=self.xi, eps=self.adv_eps, tau=self.tau,
        )
        _free(ulab_t_stack)

        lab_t_list  = list(lab_t_adv)
        ulab_t_list = list(ulab_t_adv)

        # ── Phase A: supervised loss ─────────────────────────────────────
        self.optimizer.zero_grad(set_to_none=True)

        sup_loss_dict = self.student(lab_s, targets)
        loss_sup = sum(sup_loss_dict.values())

        # ── Phase B: consistency — student forward ────────────────────────
        cls_s_lab,  reg_s_lab  = self.student.forward_train(lab_s)
        cls_s_ulab, reg_s_ulab = self.student.forward_train(ulab_s)

        # ── Phase C: consistency — teacher forward (no grad) ─────────────
        with torch.no_grad():
            cls_t_lab,  reg_t_lab  = self.teacher.model.forward_train(lab_t_list)
            cls_t_ulab, reg_t_ulab = self.teacher.model.forward_train(ulab_t_list)

        # ── Phase D: consistency losses ───────────────────────────────────
        loss_cons = (
            self.consistency_loss(cls_s_lab,  cls_t_lab,  reg_s_lab,  reg_t_lab) +
            self.consistency_loss(cls_s_ulab, cls_t_ulab, reg_s_ulab, reg_t_ulab)
        )
        _free(cls_s_lab, reg_s_lab, cls_t_lab, reg_t_lab,
              cls_s_ulab, reg_s_ulab, cls_t_ulab, reg_t_ulab)

        # ── Phase E: combine + backward ───────────────────────────────────
        lam = self.cons_scheduler.advance()
        loss_total = loss_sup + lam * loss_cons

        loss_total.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # ── EMA teacher update ────────────────────────────────────────────
        self.teacher.update(self.student)

        return {
            "loss_sup":   float(loss_sup),
            "loss_cons":  float(loss_cons),
            "loss_total": float(loss_total),
            "lambda":     lam,
        }

    # ------------------------------------------------------------------
    def evaluate(self, data_loader):
        self.student.eval()
        results = []
        with torch.no_grad():
            for imgs, targets in data_loader:
                imgs = [img.to(self.device) for img in imgs]
                preds = self.student(imgs)
                results.append(preds)
        return results

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "student":   self.student.state_dict(),
            "teacher":   self.teacher.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "cons_step": self.cons_scheduler.step,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["student"])
        self.teacher.model.load_state_dict(ckpt["teacher"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler"])
        self.cons_scheduler.step = ckpt["cons_step"]
        print(f"Checkpoint loaded ← {path}")