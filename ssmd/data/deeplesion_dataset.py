"""
DeepLesion CT Lesion Detection — PyTorch Dataset.

Converted from the TensorFlow tf.data pipeline.

Expected layout:
    data_dir/DL_info.csv
    data_dir/Images_png/<patient_study_slice>/<slice>.png

Each PNG is a 16-bit grayscale image stored as HU + 32768.

Each sample returns:
    image  : FloatTensor [3, H, W]  normalised to [-1, 1] then standardised
    target : dict with
               'boxes'  : FloatTensor [1, 4]  (x1, y1, x2, y2)
               'labels' : LongTensor  [1]      (1 = lesion)
"""

from __future__ import annotations
import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# HU clipping range (paper Sec. 4.2)
HU_MIN, HU_MAX = -1100.0, 1100.0


def _hu_to_float(arr: np.ndarray) -> np.ndarray:
    """Clip HU and map to [-1, 1]."""
    arr = np.clip(arr.astype(np.float32), HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)   # [0, 1]
    return arr * 2.0 - 1.0                       # [-1, 1]


def _parse_dl_csv(csv_path: str) -> list[tuple[str, list[float]]]:
    """Parse DL_info.csv → list of (relative_path, [x1,y1,x2,y2])."""
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            fn_idx = header.index("File_name")
        except ValueError:
            fn_idx = 0
        try:
            bb_idx = header.index("Bounding_boxes")
        except ValueError:
            bb_idx = 6

        for row in reader:
            if len(row) <= max(fn_idx, bb_idx):
                continue
            fname  = row[fn_idx].strip()
            bb_str = row[bb_idx].strip()
            try:
                coords = [float(v) for v in bb_str.replace(",", " ").split()]
                if len(coords) < 4:
                    continue
                records.append((fname, coords[:4]))
            except ValueError:
                continue
    return records


class DeepLesionDataset(Dataset):
    """
    Args:
        data_dir         : Root directory with DL_info.csv and Images_png/.
        split            : 'labeled' | 'unlabeled' | 'val' | 'train'
        labeled_fraction : Fraction of train samples used as labeled.
        target_size      : Resize target (paper: 512).
        mean             : Dataset mean for final standardisation.
        std              : Dataset std for final standardisation.
        seed             : RNG seed for deterministic split.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "labeled",
        labeled_fraction: float = 0.2,
        target_size: int = 512,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 42,
    ):
        self.images_root = os.path.join(data_dir, "Images_png")
        self.target_size = target_size
        self.mean = mean
        self.std  = std

        csv_path = os.path.join(data_dir, "DL_info.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"DL_info.csv not found at {csv_path}")

        records = _parse_dl_csv(csv_path)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(records))
        rng.shuffle(idx)
        records = [records[i] for i in idx]

        n_total   = len(records)
        n_val     = max(1, int(0.1 * n_total))
        n_train   = n_total - n_val
        n_labeled = max(1, int(labeled_fraction * n_train))

        split_map = {
            "labeled":   records[:n_labeled],
            "unlabeled": records[n_labeled:n_train],
            "val":       records[n_train:],
            "train":     records[:n_train],
        }
        self.records = split_map[split]

        if not self.records:
            raise ValueError(f"No records for split '{split}'")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        fname, bbox = self.records[idx]
        ts = self.target_size
        img_path = os.path.join(self.images_root, fname)

        # --- load 16-bit PNG
        try:
            pil_img = Image.open(img_path)
            arr = np.array(pil_img)
        except Exception:
            arr = np.zeros((ts, ts), dtype=np.uint16)

        orig_h, orig_w = arr.shape[:2]

        # Undo DeepLesion HU offset
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) - 32768.0
        else:
            arr = arr.astype(np.float32)

        # HU normalise → [-1, 1]
        arr = _hu_to_float(arr)

        # Dataset-level standardisation
        arr = (arr - self.mean) / (self.std + 1e-8)

        # Resize (PIL expects uint8, so we do it via numpy + zoom)
        from PIL import Image as PILImage
        arr_pil = PILImage.fromarray(arr.squeeze()).resize((ts, ts),
                                                           PILImage.BILINEAR)
        arr = np.array(arr_pil, dtype=np.float32)

        # Replicate to 3 channels
        img_3ch = np.stack([arr, arr, arr], axis=0)   # [3, H, W]
        image = torch.from_numpy(img_3ch)

        # Scale bbox
        x1, y1, x2, y2 = bbox
        scale_x = ts / orig_w
        scale_y = ts / orig_h
        boxes = torch.tensor(
            [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]],
            dtype=torch.float32
        )
        target = {
            "boxes":  boxes,
            "labels": torch.ones(1, dtype=torch.long),
        }
        return image, target