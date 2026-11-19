"""Dataset utilities for the 4-class Sentinel-2 land-cover tiles.

Expected folder layout (what your DATASET.zip already has):
    DATASET/{Train,Validation,Test}/{images,labels}/<same_name>.tif

Label values in the files:  1 = built-up, 2 = water, 3 = trees, 4 = barren.
Internally they are remapped to 0..3; anything else becomes IGNORE_INDEX (255).
"""
import glob
import os
import warnings

import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from torch.utils.data import Dataset

# Label rasters have no georeference -> silence the (harmless) warning.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

CLASS_NAMES = ["built-up", "water", "trees", "barren"]
NUM_CLASSES = 4
IGNORE_INDEX = 255
TILE = 256  # all tiles are padded up to TILE x TILE (edge tiles are smaller)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_pairs(root, split):
    img_dir = os.path.join(root, split, "images")
    lab_dir = os.path.join(root, split, "labels")
    pairs = []
    for ip in sorted(glob.glob(os.path.join(img_dir, "*.tif"))):
        lp = os.path.join(lab_dir, os.path.basename(ip))
        if not os.path.exists(lp):
            raise FileNotFoundError(f"No label for image {ip}")
        pairs.append((ip, lp))
    if not pairs:
        raise FileNotFoundError(f"No .tif images found in {img_dir}")
    return pairs


def read_pair(ip, lp):
    with rasterio.open(ip) as s:
        img = s.read()  # (C, H, W) uint8
    with rasterio.open(lp) as s:
        lab = s.read(1)  # (H, W) uint8
    if img.shape[1:] != lab.shape:
        raise ValueError(f"Size mismatch {ip}: image {img.shape[1:]} vs label {lab.shape}")
    out = np.full(lab.shape, IGNORE_INDEX, dtype=np.uint8)
    for v in range(1, NUM_CLASSES + 1):
        out[lab == v] = v - 1
    return img, out


def pad_to(img, lab, size=TILE):
    """Pad (C,H,W) image with 0 and (H,W) label with IGNORE_INDEX at bottom/right."""
    _, h, w = img.shape
    if h > size or w > size:
        raise ValueError(f"Tile {h}x{w} is larger than {size}; crop it first.")
    ph, pw = size - h, size - w
    if ph or pw:
        img = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="constant", constant_values=0)
        lab = np.pad(lab, ((0, ph), (0, pw)), mode="constant", constant_values=IGNORE_INDEX)
    return img, lab


class LandCoverDataset(Dataset):
    def __init__(self, root, split, augment=False, limit=None):
        self.pairs = list_pairs(root, split)
        if limit:
            self.pairs = self.pairs[:limit]
        # Whole split fits in RAM (a few hundred 256x256x3 uint8 tiles) -> cache it.
        self.data = [read_pair(ip, lp) for ip, lp in self.pairs]
        self.augment = augment
        self.mean = None
        self.std = None

    # ---- statistics -------------------------------------------------------
    def compute_stats(self):
        """Per-band mean/std (0..1 scale) over all cached images."""
        s = np.zeros(3, np.float64)
        s2 = np.zeros(3, np.float64)
        n = 0
        for img, _ in self.data:
            x = img.astype(np.float64) / 255.0
            s += x.sum(axis=(1, 2))
            s2 += (x ** 2).sum(axis=(1, 2))
            n += x.shape[1] * x.shape[2]
        mean = s / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
        return mean.tolist(), std.tolist()

    def class_counts(self):
        counts = np.zeros(NUM_CLASSES, np.int64)
        for _, lab in self.data:
            counts += np.bincount(lab[lab != IGNORE_INDEX], minlength=NUM_CLASSES)
        return counts

    def set_norm(self, mean, std):
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.std = np.asarray(std, np.float32)[:, None, None]

    # ---- augmentation -----------------------------------------------------
    @staticmethod
    def _photometric(x):
        # mild brightness / contrast jitter (scenes differ in haze and exposure)
        if np.random.rand() < 0.5:
            x = x * np.random.uniform(0.85, 1.15)
        if np.random.rand() < 0.5:
            m = x.mean()
            x = (x - m) * np.random.uniform(0.85, 1.15) + m
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def _geometric(x, y):
        if np.random.rand() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        if np.random.rand() < 0.5:
            x, y = x[:, ::-1, :], y[::-1, :]
        k = np.random.randint(4)
        if k:
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(0, 1))
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    # ---- torch API --------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        assert self.mean is not None, "call set_norm(mean, std) first"
        img, lab = self.data[i]
        x = img.astype(np.float32) / 255.0
        if self.augment:
            x = self._photometric(x)
            x, lab = self._geometric(x, lab)
        x = (x - self.mean) / self.std  # zero padding below == "mean colour"
        x, lab = pad_to(x, lab, TILE)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(lab.astype(np.int64))
