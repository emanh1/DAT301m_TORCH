"""
DataLoader utilities for SSMD training.

Provides:
  - collate_fn        : handles variable-length boxes across a batch
  - make_loaders_dsb  : returns (labeled_loader, unlabeled_loader, val_loader)
  - make_loaders_dl   : same for DeepLesion
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .dsb_dataset import DSBDataset
from .deeplesion_dataset import DeepLesionDataset


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Stack images into a tensor; keep targets as a list of dicts.
    This matches the format torchvision detection models expect.

    Returns:
        images  : list[FloatTensor[3,H,W]]
        targets : list[dict]  each with 'boxes' [N,4] and 'labels' [N]
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


# ---------------------------------------------------------------------------
# DSB 2018 loaders
# ---------------------------------------------------------------------------

def make_loaders_dsb(
    data_dir: str,
    labeled_fraction: float = 0.2,
    target_size: int = 448,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (labeled_loader, unlabeled_loader, val_loader) for DSB 2018.

    The unlabeled loader has no targets (targets will be empty boxes tensors,
    which the trainer ignores — only the images are used for consistency loss).
    """
    common = dict(
        data_dir=data_dir,
        labeled_fraction=labeled_fraction,
        target_size=target_size,
        seed=seed,
    )

    labeled_ds   = DSBDataset(split="labeled",   **common)
    unlabeled_ds = DSBDataset(split="unlabeled", **common)
    val_ds       = DSBDataset(split="val",        **common)

    loader_kw = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    labeled_loader   = DataLoader(labeled_ds,   shuffle=True,  **loader_kw)
    unlabeled_loader = DataLoader(unlabeled_ds, shuffle=True,  **loader_kw)
    val_loader       = DataLoader(val_ds,       shuffle=False, **loader_kw)

    print(f"[DSB]  labeled={len(labeled_ds)}  "
          f"unlabeled={len(unlabeled_ds)}  val={len(val_ds)}")
    return labeled_loader, unlabeled_loader, val_loader


# ---------------------------------------------------------------------------
# DeepLesion loaders
# ---------------------------------------------------------------------------

def make_loaders_deeplesion(
    data_dir: str,
    labeled_fraction: float = 0.2,
    target_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (labeled_loader, unlabeled_loader, val_loader) for DeepLesion.
    """
    common = dict(
        data_dir=data_dir,
        labeled_fraction=labeled_fraction,
        target_size=target_size,
        mean=mean,
        std=std,
        seed=seed,
    )

    labeled_ds   = DeepLesionDataset(split="labeled",   **common)
    unlabeled_ds = DeepLesionDataset(split="unlabeled", **common)
    val_ds       = DeepLesionDataset(split="val",        **common)

    loader_kw = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    labeled_loader   = DataLoader(labeled_ds,   shuffle=True,  **loader_kw)
    unlabeled_loader = DataLoader(unlabeled_ds, shuffle=True,  **loader_kw)
    val_loader       = DataLoader(val_ds,       shuffle=False, **loader_kw)

    print(f"[DeepLesion]  labeled={len(labeled_ds)}  "
          f"unlabeled={len(unlabeled_ds)}  val={len(val_ds)}")
    return labeled_loader, unlabeled_loader, val_loader