#!/usr/bin/env python
"""Sanity-check the dataset before training:  python check_data.py --data_root DATASET"""
import argparse
import collections

import numpy as np
import rasterio

from src.dataset import CLASS_NAMES, list_pairs

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
a = ap.parse_args()

for split in ("Train", "Validation", "Test"):
    pairs = list_pairs(a.data_root, split)
    sizes, vals = collections.Counter(), np.zeros(256, np.int64)
    bands, dtypes, mean, std = set(), set(), [], []
    for ip, lp in pairs:
        with rasterio.open(ip) as s:
            x = s.read()
            bands.add(s.count)
            dtypes.add(s.dtypes[0])
        with rasterio.open(lp) as s:
            y = s.read(1)
        assert x.shape[1:] == y.shape, f"size mismatch: {ip}"
        sizes[y.shape] += 1
        u, c = np.unique(y, return_counts=True)
        vals[u] += c
        mean.append(x.mean((1, 2)))
        std.append(x.std((1, 2)))
    tot = vals.sum()
    print(f"\n{split}: {len(pairs)} tiles | bands {bands} | dtype {dtypes} | tile sizes {dict(sizes)}")
    for v in range(256):
        if vals[v]:
            name = CLASS_NAMES[v - 1] if 1 <= v <= 4 else "UNEXPECTED VALUE"
            print(f"   label {v} ({name}): {100 * vals[v] / tot:.2f}%")
    print("   image mean per band:", np.mean(mean, 0).round(1), "| std:", np.mean(std, 0).round(1))
