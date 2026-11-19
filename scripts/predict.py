#!/usr/bin/env python
"""Apply the trained U-Net to new RGB tiles / mosaics (sliding window).

Writes one GeoTIFF per input (1 = built-up, 2 = water, 3 = trees, 4 = barren) and area_ha.csv.
    python predict.py --checkpoint runs/<name>/best.pt --input new_tiles/ --out preds/ --crs EPSG:32643
"""
import argparse
import csv
import glob
import os
import warnings

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.crs import CRS

from src.dataset import CLASS_NAMES, NUM_CLASSES
from src.model import build_model


@torch.no_grad()
def infer(model, x, tta):
    flips = [(), (3,), (2,), (2, 3)] if tta else [()]
    p = 0
    for d in flips:
        xi = x.flip(d) if d else x
        pi = model(xi).float().softmax(1)
        p = p + (pi.flip(d) if d else pi)
    return p / len(flips)


def predict_probs(model, x, device, tile=256, overlap=64, tta=False):
    """x: (C,H,W) normalised tensor -> (classes,H,W) mean softmax probabilities."""
    _, h, w = x.shape
    ph, pw = max(tile - h, 0), max(tile - w, 0)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    _, hp, wp = x.shape
    stride = tile - overlap

    def starts(size):
        s = list(range(0, size - tile + 1, stride))
        if s[-1] != size - tile:
            s.append(size - tile)
        return s

    acc, cnt = torch.zeros(NUM_CLASSES, hp, wp), torch.zeros(1, hp, wp)
    for y in starts(hp):
        for xx in starts(wp):
            patch = x[:, y:y + tile, xx:xx + tile].unsqueeze(0).to(device)
            acc[:, y:y + tile, xx:xx + tile] += infer(model, patch, tta)[0].cpu()
            cnt[:, y:y + tile, xx:xx + tile] += 1
    return (acc / cnt)[:, :h, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="a .tif file or a folder of .tif files")
    ap.add_argument("--out", default="preds")
    ap.add_argument("--crs", default=None, help="override the CRS tag, e.g. EPSG:32643 (your files say 4326)")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation (4x slower)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    ck = torch.load(a.checkpoint, map_location="cpu")
    c = ck["args"]
    model = build_model(c.get("model", "unet"), c.get("encoder", "resnet34"), False,  # weights come from the checkpoint
                        c.get("width", 32), c.get("levels", 4), c["dropout"], NUM_CLASSES)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    mean = torch.tensor(ck["mean"]).view(3, 1, 1).float()
    std = torch.tensor(ck["std"]).view(3, 1, 1).float()

    files = [a.input] if os.path.isfile(a.input) else sorted(glob.glob(os.path.join(a.input, "*.tif")))
    os.makedirs(a.out, exist_ok=True)
    rows, totals = [], np.zeros(NUM_CLASSES)
    for f in files:
        with rasterio.open(f) as src:
            img, prof = src.read(), src.profile.copy()
            px_area = abs(src.transform.a * src.transform.e)  # m^2 when the grid is in metres
        x = (torch.from_numpy(img).float() / 255.0 - mean) / std
        pred = predict_probs(model, x, device, a.tile, a.overlap, a.tta).argmax(0).numpy().astype(np.uint8) + 1
        prof.update(count=1, dtype="uint8", nodata=0, compress="lzw")
        for k in ("photometric", "interleave"):
            prof.pop(k, None)
        if a.crs:
            prof["crs"] = CRS.from_string(a.crs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(os.path.join(a.out, os.path.basename(f)), "w", **prof) as dst:
                dst.write(pred, 1)
        ha = np.bincount(pred.ravel(), minlength=NUM_CLASSES + 1)[1:] * px_area / 1e4
        totals += ha
        rows.append([os.path.basename(f)] + [round(v, 2) for v in ha])
        if px_area < 0.01:
            print("WARNING: pixel size looks like degrees; areas are not in hectares")
        print(os.path.basename(f), {n: round(float(v), 1) for n, v in zip(CLASS_NAMES, ha)})
    with open(os.path.join(a.out, "area_ha.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file"] + [f"{n}_ha" for n in CLASS_NAMES])
        w.writerows(rows)
        w.writerow(["TOTAL"] + [round(v, 2) for v in totals])
    print("TOTAL (ha):", {n: round(float(v), 1) for n, v in zip(CLASS_NAMES, totals)})


if __name__ == "__main__":
    main()
