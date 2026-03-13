"""
DSB 2018 Nuclei Detection — PyTorch Dataset.

Converted from the TensorFlow tf.data pipeline.

Expected layout:
    data_dir/train/<image_id>/images/*.png
    data_dir/train/<image_id>/masks/*.png

Each sample returns:
    image  : FloatTensor [3, H, W]  in [0, 1]
    target : dict with
               'boxes'  : FloatTensor [N, 4]  (x1, y1, x2, y2)
               'labels' : LongTensor  [N]      (all 1 = nucleus)
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _masks_to_boxes(mask_dir: str) -> np.ndarray:
    """Convert per-nucleus PNG masks → bounding boxes [G, 4] float32."""
    boxes = []
    if not os.path.isdir(mask_dir):
        return np.zeros((0, 4), dtype=np.float32)

    for fname in sorted(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, fname)
        try:
            mask = np.array(Image.open(mask_path).convert("L"))
        except Exception:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        boxes.append([float(xs.min()), float(ys.min()),
                      float(xs.max()), float(ys.max())])

    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


class DSBDataset(Dataset):
    """
    Args:
        data_dir         : Root dir containing train/<id>/images + masks.
        split            : 'labeled' | 'unlabeled' | 'val' | 'train'
        labeled_fraction : Fraction of train IDs used as labeled.
        target_size      : Resize target (paper: 448).
        seed             : RNG seed for deterministic split.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "labeled",
        labeled_fraction: float = 0.2,
        target_size: int = 448,
        seed: int = 42,
    ):
        self.target_size = target_size
        self.split = split

        train_dir = os.path.join(data_dir, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"DSB train dir not found: {train_dir}")

        image_ids = sorted(os.listdir(train_dir))
        rng = np.random.default_rng(seed)
        rng.shuffle(image_ids)

        n_total   = len(image_ids)
        n_val     = max(1, int(0.1 * n_total))
        n_train   = n_total - n_val
        n_labeled = max(1, int(labeled_fraction * n_train))

        split_map = {
            "labeled":   image_ids[:n_labeled],
            "unlabeled": image_ids[n_labeled:n_train],
            "val":       image_ids[n_train:],
            "train":     image_ids[:n_train],
        }
        ids = split_map[split]

        self.samples: list[tuple[str, str]] = []
        for img_id in ids:
            img_folder = os.path.join(train_dir, img_id, "images")
            msk_folder = os.path.join(train_dir, img_id, "masks")
            if not os.path.isdir(img_folder):
                continue
            pngs = sorted(f for f in os.listdir(img_folder) if f.endswith(".png"))
            if not pngs:
                continue
            self.samples.append((os.path.join(img_folder, pngs[0]), msk_folder))

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {data_dir}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path, mask_dir = self.samples[idx]
        ts = self.target_size

        # --- load image
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            img = np.zeros((ts, ts, 3), dtype=np.uint8)

        orig_h, orig_w = img.shape[:2]
        img = np.array(Image.fromarray(img).resize((ts, ts)))
        image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3,H,W]

        # --- load boxes
        boxes = _masks_to_boxes(mask_dir)
        if len(boxes):
            scale_x = ts / orig_w
            scale_y = ts / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        target = {
            "boxes":  torch.from_numpy(boxes),
            "labels": torch.ones(len(boxes), dtype=torch.long),
        }
        return image, target