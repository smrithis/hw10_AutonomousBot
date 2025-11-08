# dataset.py  (image-only; no history)
"""
Minimal PyTorch dataset for TurtleBot3 steering (image → ω or ω[5]).

Records look like:
{
  "image_path": "processed/run_YYYYMMDD/images/bag_1_<ts>.png",
  "label":   {"omega": ...}
  "split":   "train" | "val" | "test",
  ... (timestamps/audit fields allowed but ignored)
}
"""

from typing import Optional, Dict, Tuple, List
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class TopCrop(torch.nn.Module):
    def __init__(self, frac: float):
        super().__init__()
        self.frac = float(max(0.0, min(1.0, frac)))
    def forward(self, im: Image.Image):
        if self.frac <= 0:
            return im
        w, h = im.size
        cut = int(h * self.frac)
        return im.crop((0, cut, w, h))

def make_img_tf_keep_ar(short_side: int = 256, top_crop_frac: float = 0.0):
    """
    Keep aspect ratio: resize the *shorter* side to `short_side` (e.g., 640x480 -> 341x256).
    Input is already PIL.Image; do NOT use ToPILImage here.
    """
    ops: List[torch.nn.Module] = []
    if top_crop_frac > 0:
        ops.append(TopCrop(top_crop_frac))
    ops += [
        T.Resize(short_side),  # preserves aspect ratio (short side -> short_side)
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(ops)

class SteeringDataset(Dataset):
    def __init__(self,
                 index_json: str,
                 split: str = "train",
                 root: Optional[str] = None,
                 hist_len: int = 0,              # kept only for API compatibility; ignored
                 standardize: bool = True,
                 stats: Optional[Dict[str, float]] = None,
                 img_tf: Optional[T.Compose] = None,
                 omega_sign: float = 1.0):
        """
        Args:
            index_json: path to index_split.json (or index.json if you set split yourself)
            split: 'train' | 'val' | 'test'
            root: optional prefix to join with image_path if paths are relative
            hist_len: ignored (history is not used); kept for API compatibility
            standardize: if True, returns standardized target using mu/sigma
            stats: optional {'mu': float, 'sigma': float}; if None, computed from TRAIN labels in file
            img_tf: torchvision transform; default keeps AR and normalizes
            omega_sign: multiply LABEL by this value (use -1.0 to flip sign)
        """
        super().__init__()
        self.all_entries: List[Dict] = json.load(open(index_json, "r"))
        self.entries: List[Dict] = [e for e in self.all_entries if e.get("split", "train") == split]
        if len(self.entries) == 0:
            raise ValueError(f"No entries with split='{split}' found in {index_json}")

        self.root = root
        self.standardize = standardize
        self.img_tf = img_tf or make_img_tf_keep_ar()
        self.omega_sign = float(omega_sign)

        # stats (mu, sigma) computed across TRAIN labels AFTER sign is applied
        if stats is None:
            # Allow either scalar label["omega"] or vector label["omega_seq"]
            def _extract_label_value(e):
                lab = e["label"]
                if "omega" in lab:
                    return float(lab["omega"])
                elif "omega_seq" in lab:
                    # choose how to compute stats for sequences: use the first step by default
                    return float(lab["omega_seq"][0])
                else:
                    raise KeyError("label must contain 'omega' or 'omega_seq'")
            train_labels = [self.omega_sign * _extract_label_value(e)
                            for e in self.all_entries if e.get("split","train")=="train"]
            if len(train_labels) == 0:
                raise ValueError("No train labels found to compute (mu, sigma). Provide stats=... or set split fields.")
            mu = float(np.mean(train_labels))
            sigma = float(np.std(train_labels) + 1e-8)
            self.stats = {"mu": mu, "sigma": sigma}
        else:
            self.stats = stats
        self.mu, self.sigma = float(self.stats["mu"]), float(self.stats["sigma"])

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, path: str) -> torch.Tensor:
        im = Image.open(path).convert("RGB")
        return self.img_tf(im)  # tensor CxHxW

    def __getitem__(self, i: int):
        rec = self.entries[i]

        # image path (allow relative paths)
        img_path = rec["image_path"]
        if self.root is not None and not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)

        x = self._load_image(img_path)

        # ---- NO HISTORY ----
        h = torch.zeros(0, dtype=torch.float32)   # placeholder (shape [0])

        # ---- Label (supports scalar or 5-step sequence) ----
        lab = rec["label"]
        if "omega" in lab:
            y_raw = self.omega_sign * float(lab["omega"])
        elif "omega_seq" in lab:
            y_raw = self.omega_sign * torch.tensor(lab["omega_seq"], dtype=torch.float32)
        else:
            raise KeyError("label must contain 'omega' or 'omega_seq'")

        # Standardize
        if isinstance(y_raw, torch.Tensor):
            y_std = (y_raw - self.mu) / self.sigma if self.standardize else y_raw
        else:
            y_std = (y_raw - self.mu) / self.sigma if self.standardize else y_raw
            y_raw = torch.tensor(y_raw, dtype=torch.float32)

        # lightweight meta for debugging
        meta = {
            "image_path": rec["image_path"],
            "stamp_img_bag": rec.get("stamp_img_bag"),
            "stamp_img_header": rec.get("stamp_img_header"),
        }

        return x, h, torch.as_tensor(y_std, dtype=torch.float32), torch.as_tensor(y_raw, dtype=torch.float32), meta


def make_loaders(index_json: str,
                 root: Optional[str] = None,
                 bs: int = 64,
                 num_workers: int = 4,
                 hist_len: int = 0,                 # accepted but ignored
                 omega_sign: float = 1.0,
                 short_side: int = 256,
                 top_crop_frac: float = 0.0) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, float]]:
    """
    Builds train/val/test loaders from a single index_split.json, sharing (mu,sigma).
    `omega_sign` is applied consistently across all splits.
    `short_side` and `top_crop_frac` control the deterministic preprocessing for all splits
    (you can still override train_dl.dataset.img_tf in your training script for augmentations).
    """
    base_tf = make_img_tf_keep_ar(short_side=short_side, top_crop_frac=top_crop_frac)

    # Build train first to compute stats with the chosen sign
    train_ds = SteeringDataset(index_json, split="train", root=root, hist_len=0,
                               standardize=True, stats=None, img_tf=base_tf, omega_sign=omega_sign)
    stats = train_ds.stats

    # Optional splits
    has_val  = any(e.get("split") == "val"  for e in train_ds.all_entries)
    has_test = any(e.get("split") == "test" for e in train_ds.all_entries)

    val_ds  = SteeringDataset(index_json, split="val",  root=root, hist_len=0,
                              standardize=True, stats=stats, img_tf=base_tf, omega_sign=omega_sign) if has_val else None
    test_ds = SteeringDataset(index_json, split="test", root=root, hist_len=0,
                              standardize=True, stats=stats, img_tf=base_tf, omega_sign=omega_sign) if has_test else None

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True) if val_ds else None
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None

    return train_dl, val_dl, test_dl, stats


# Quick standalone check
if __name__ == "__main__":
    # Example:
    # python dataset.py --check data/processed/run_YYYYMMDD/index_split.json
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        idx_path = sys.argv[2]
        train_dl, val_dl, test_dl, stats = make_loaders(idx_path, root=os.path.dirname(idx_path), bs=4, omega_sign=1.0)
        print("mu, sigma:", stats)
        xb, hb, yb_std, yb_raw, meta = next(iter(train_dl))
        print("Batch shapes:", xb.shape, hb.shape, yb_std.shape, yb_raw.shape)
        print("First meta:", meta["image_path"][0], meta["stamp_img_bag"][0])
    else:
        print("Usage: python dataset.py --check <path_to_index_split.json>")

