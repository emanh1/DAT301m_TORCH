"""
SSMD Trainer — full implementation of Algorithm 1.

Ties together:
  • SSMDDetector (student)         — models/detector.py
  • EMATeacher   (teacher)         — utils/ema_teacher.py
  • AdaptiveConsistencyCost        — models/adaptive_consistency_cost.py
  • instance_adversarial_perturbation — models/adversarial_perturbation.py
  • Augmentation pipeline           — utils/augmentations.py
  • ConsistencyScheduler (λ)        — utils/lambda_schedule.py

Training loop sketch
--------------------
For each iteration n:
  1.  Augment labeled batch   → X_s  (student),  X_t_base (teacher)
  2.  Augment unlabeled batch → X̃_s (student),  X̃_t_base (teacher)
  3.  Compute adversarial perturbation on teacher inputs.
  4.  Forward student + teacher.
  5.  Supervised loss on labeled images.
  6.  Adaptive consistency cost on labeled + unlabeled images.
  7.  Total loss = L_sup + λ · L_cons.
  8.  Backward + optimiser step (student only).
  9.  EMA update of teacher.
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


class SSMDTrainer:
    """
    Full SSMD training orchestrator.

    Args:
        num_classes    : Foreground classes.
        device         : torch.device.
        ema_alpha      : EMA decay for teacher (paper: 0.99).
        lr             : Initial learning rate (paper: 1e-5 for DSB).
        total_epochs   : Total training epochs.
        steps_per_epoch: Batches per epoch (used to compute total iterations N).
        xi             : Adversarial perturbation seed scale ξ.
        adv_eps        : Adversarial perturbation magnitude ε.
        tau            : Foreground confidence threshold τ.
        cutout_n       : Number of cutout masks.
        cutout_s       : Cutout mask side length.
        max_rot_deg    : Max random rotation degrees (paper: 10).
        nrb_gamma      : NRB sigmoid gate scale γ (paper: 0.9).
        lambda_cls     : Weight on KL term in consistency cost.
        lambda_reg     : Weight on MSE term in consistency cost.
    """

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
    ):
        self.device = device
        self.xi = xi
        self.adv_eps = adv_eps
        self.tau = tau
        self.cutout_n = cutout_n
        self.cutout_s = cutout_s
        self.max_rot_deg = max_rot_deg

        # --- Student network (with NRB)
        self.student = SSMDDetector(
            num_classes=num_classes,
            use_nrb=True,
            nrb_gamma=nrb_gamma,
            pretrained=True,
        ).to(device)

        # --- Teacher network (EMA of student, no NRB needed at inference)
        self.teacher = EMATeacher(self.student, alpha=ema_alpha)
        self.teacher.model.to(device)

        # --- Loss modules
        self.consistency_loss = AdaptiveConsistencyCost(
            lambda_cls=lambda_cls,
            lambda_reg=lambda_reg,
        )

        # --- Optimiser  (Adam, paper Sec. 4.2)
        self.optimizer = optim.Adam(self.student.parameters(), lr=lr)

        # --- LR scheduler: divide by 10 at 75% of training
        total_iters = total_epochs * steps_per_epoch
        milestones = [int(0.75 * total_iters)]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1
        )

        # --- Consistency weight schedule  (Eq. 10)
        self.cons_scheduler = ConsistencyScheduler(total_iters)

    # ------------------------------------------------------------------
    def _move_targets(self, targets: List[dict]) -> List[dict]:
        return [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()} for t in targets]

    # ------------------------------------------------------------------
    def train_step(
        self,
        labeled_imgs: List[torch.Tensor],
        labeled_targets: List[dict],
        unlabeled_imgs: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        One training iteration (Algorithm 1).

        Args:
            labeled_imgs    : List of float tensors [C,H,W] (labeled batch X).
            labeled_targets : List of dicts with 'boxes'[N,4] and 'labels'[N].
            unlabeled_imgs  : List of float tensors [C,H,W] (unlabeled batch X̃).

        Returns:
            dict with 'loss_sup', 'loss_cons', 'loss_total', 'lambda'.
        """
        self.student.train()
        self.teacher.model.eval()

        # ---------- Step 1-2: augment both branches ----------
        # Student branch
        lab_s  = batch_student_augment(labeled_imgs,   self.max_rot_deg,
                                       self.cutout_n,  self.cutout_s)
        ulab_s = batch_student_augment(unlabeled_imgs, self.max_rot_deg,
                                       self.cutout_n,  self.cutout_s)

        # Teacher branch base (before adversarial perturbation)
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

        # ---------- Step 3: adversarial perturbation on teacher inputs ----------
        # Build batch tensors for the adversarial module
        lab_t_stack  = torch.stack(lab_t)
        ulab_t_stack = torch.stack(ulab_t)

        def _student_fn(x):
            return self.student.forward_train(list(x))

        def _teacher_fn(x):
            with torch.no_grad():
                cls_t, reg_t = self.teacher.model.forward_train(list(x))
            return cls_t, reg_t

        lab_t_adv = instance_adversarial_perturbation(
            lab_t_stack, _student_fn, _teacher_fn,
            self.consistency_loss,
            xi=self.xi, eps=self.adv_eps, tau=self.tau,
        )
        ulab_t_adv = instance_adversarial_perturbation(
            ulab_t_stack, _student_fn, _teacher_fn,
            self.consistency_loss,
            xi=self.xi, eps=self.adv_eps, tau=self.tau,
        )
        lab_t_list  = list(lab_t_adv)
        ulab_t_list = list(ulab_t_adv)

        # ---------- Step 4: forward passes ----------
        # Supervised loss (labeled data, student)
        sup_loss_dict = self.student(lab_s, targets)
        loss_sup = sum(sup_loss_dict.values())

        # Consistency: student on student-augmented images
        cls_s_lab,  reg_s_lab  = self.student.forward_train(lab_s)
        cls_s_ulab, reg_s_ulab = self.student.forward_train(ulab_s)

        # Consistency: teacher on adversarially perturbed images
        with torch.no_grad():
            cls_t_lab,  reg_t_lab  = self.teacher.model.forward_train(lab_t_list)
            cls_t_ulab, reg_t_ulab = self.teacher.model.forward_train(ulab_t_list)

        # ---------- Step 5-6: consistency losses ----------
        loss_cons_lab  = self.consistency_loss(cls_s_lab,  cls_t_lab,
                                               reg_s_lab,  reg_t_lab)
        loss_cons_ulab = self.consistency_loss(cls_s_ulab, cls_t_ulab,
                                               reg_s_ulab, reg_t_ulab)
        loss_cons = loss_cons_lab + loss_cons_ulab

        # ---------- Step 7: total loss with λ schedule ----------
        lam = self.cons_scheduler.advance()
        loss_total = loss_sup + lam * loss_cons

        # ---------- Step 8: backward + optimiser ----------
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # ---------- Step 9: EMA teacher update ----------
        self.teacher.update(self.student)

        return {
            "loss_sup":   float(loss_sup),
            "loss_cons":  float(loss_cons),
            "loss_total": float(loss_total),
            "lambda":     lam,
        }

    # ------------------------------------------------------------------
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Inference with the student network.

        Note: at inference only the student is used (Section 3.1, paper).
        """
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
            "student": self.student.state_dict(),
            "teacher": self.teacher.model.state_dict(),
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
