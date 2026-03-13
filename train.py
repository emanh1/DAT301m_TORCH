"""
train.py — SSMD training entry point.

Usage
-----
# DSB 2018 nuclei detection:
python train.py --dataset dsb --data_dir /path/to/dsb

# DeepLesion CT lesion detection:
python train.py --dataset deeplesion --data_dir /path/to/deeplesion

# Resume from checkpoint:
python train.py --dataset dsb --data_dir /path/to/dsb --resume ckpts/epoch_10.pt

# Use only 10% labeled data:
python train.py --dataset dsb --data_dir /path/to/dsb --labeled_fraction 0.1
"""

import argparse
import itertools
import os
import time
import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SSMD semi-supervised detector")

    # Data
    p.add_argument("--dataset",          choices=["dsb", "deeplesion"], required=True)
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--labeled_fraction", type=float, default=0.2,
                   help="Fraction of training data that is labeled (default 0.2)")
    p.add_argument("--num_workers",      type=int,   default=4)

    # Model
    p.add_argument("--num_classes",      type=int,   default=1,
                   help="Number of foreground classes")
    p.add_argument("--nrb_gamma",        type=float, default=0.9)

    # Training
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=1e-5)
    p.add_argument("--ema_alpha",        type=float, default=0.99)
    p.add_argument("--resume",           type=str,   default=None,
                   help="Path to checkpoint to resume from")

    # Consistency / perturbation
    p.add_argument("--xi",               type=float, default=5e-7)
    p.add_argument("--adv_eps",          type=float, default=2.0)
    p.add_argument("--tau",              type=float, default=0.95)
    p.add_argument("--cutout_n",         type=int,   default=5)
    p.add_argument("--cutout_s",         type=int,   default=70)
    p.add_argument("--max_rot_deg",      type=float, default=10.0)

    # Memory
    p.add_argument("--image_size",       type=int,   default=None,
                   help="Override default image size (dsb=448, dl=512). "
                        "Reduce (e.g. 256) to save VRAM.")
    p.add_argument("--no_grad_ckpt",     action="store_true",
                   help="Disable gradient checkpointing (faster but uses more VRAM)")

    # I/O
    p.add_argument("--ckpt_dir",         type=str,   default="ckpts")
    p.add_argument("--save_every",       type=int,   default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--log_every",        type=int,   default=10,
                   help="Print metrics every N iterations")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_target_size(args) -> int:
    if args.image_size is not None:
        return args.image_size
    return 448 if args.dataset == "dsb" else 512


def build_loaders(args):
    from ssmd.data import make_loaders_dsb, make_loaders_deeplesion
    ts = get_target_size(args)
    kw = dict(
        data_dir=args.data_dir,
        labeled_fraction=args.labeled_fraction,
        target_size=ts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.dataset == "dsb":
        return make_loaders_dsb(**kw)
    else:
        return make_loaders_deeplesion(**kw)


def compute_val_loss(trainer, val_loader, device):
    """Run one pass over val set with student in eval mode and return avg loss."""
    trainer.student.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = trainer.student(imgs, targets)
            total += sum(v.item() for v in loss_dict.values())
            count += 1
    trainer.student.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data
    print("Building data loaders...")
    labeled_loader, unlabeled_loader, val_loader = build_loaders(args)

    steps_per_epoch = len(labeled_loader)

    # --- Trainer
    from ssmd.trainer import SSMDTrainer
    trainer = SSMDTrainer(
        num_classes=args.num_classes,
        device=device,
        ema_alpha=args.ema_alpha,
        lr=args.lr,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        xi=args.xi,
        adv_eps=args.adv_eps,
        tau=args.tau,
        cutout_n=args.cutout_n,
        cutout_s=args.cutout_s,
        max_rot_deg=args.max_rot_deg,
        nrb_gamma=args.nrb_gamma,
        use_grad_ckpt=not args.no_grad_ckpt,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Unlabeled iterator cycles indefinitely — it's typically larger than labeled
    ulab_iter = itertools.cycle(unlabeled_loader)

    best_val_loss = float("inf")
    print(f"\nStarting training: {args.epochs} epochs, "
          f"{steps_per_epoch} steps/epoch\n")

    # --- Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running = {"loss_sup": 0.0, "loss_cons": 0.0, "loss_total": 0.0}

        for step, (lab_imgs, lab_targets) in enumerate(labeled_loader, 1):
            ulab_imgs, _ = next(ulab_iter)   # unlabeled targets ignored

            metrics = trainer.train_step(
                labeled_imgs=lab_imgs,
                labeled_targets=lab_targets,
                unlabeled_imgs=ulab_imgs,
            )

            for k in running:
                running[k] += metrics[k]

            if step % args.log_every == 0:
                avg = {k: v / step for k, v in running.items()}
                print(
                    f"  Epoch {epoch:03d} | Step {step:04d}/{steps_per_epoch} | "
                    f"sup={avg['loss_sup']:.4f}  "
                    f"cons={avg['loss_cons']:.4f}  "
                    f"total={avg['loss_total']:.4f}  "
                    f"λ={metrics['lambda']:.4f}"
                )

        # --- End-of-epoch summary
        avg = {k: v / steps_per_epoch for k, v in running.items()}
        elapsed = time.time() - epoch_start
        print(
            f"\nEpoch {epoch:03d} done ({elapsed:.1f}s) | "
            f"sup={avg['loss_sup']:.4f}  "
            f"cons={avg['loss_cons']:.4f}  "
            f"total={avg['loss_total']:.4f}"
        )

        # --- Validation
        val_loss = compute_val_loss(trainer, val_loader, device)
        print(f"  Val loss: {val_loss:.4f}", end="")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.ckpt_dir, "best.pt")
            trainer.save_checkpoint(best_path)
            print(f"  ← best, saved to {best_path}", end="")
        print()

        # --- Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pt")
            trainer.save_checkpoint(ckpt_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()