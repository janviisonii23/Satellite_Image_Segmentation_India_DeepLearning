#!/usr/bin/env python
"""Draw what the attention gates do (works for attn_unet and attn_resnet_unet checkpoints).

Each row = one tile:  RGB | true label | prediction | attention at the finest level ... coarsest level
(bright = the gate lets the skip features through, dark = suppressed).
Also prints the mean attention on pixels of each TRUE class.

    python attention_maps.py --checkpoint runs/<name>/best.pt --data_root DATASET --split Test --n 6
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from src.dataset import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, LandCoverDataset
from src.model import AttentionGate, build_model

PALETTE = {0: (200, 200, 200), 1: (30, 90, 255), 2: (20, 160, 40), 3: (230, 180, 60)}  # built-up, water, trees, barren


def color_labels(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for k, c in PALETTE.items():
        out[lab == k] = c
    return out


def heat(a):  # 0..1 -> black / red / yellow / white
    a = np.clip(a, 0, 1)
    return (np.stack([np.clip(3 * a, 0, 1), np.clip(3 * a - 1, 0, 1), np.clip(3 * a - 2, 0, 1)], -1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="Test", choices=["Train", "Validation", "Test"])
    ap.add_argument("--n", type=int, default=6, help="number of tiles (evenly spaced over the split)")
    ap.add_argument("--out", default="attention_viz")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu")
    c = ck["args"]
    model = build_model(c.get("model", "unet"), c.get("encoder", "resnet34"), False,
                        c.get("width", 32), c.get("levels", 4), c["dropout"], NUM_CLASSES)
    model.load_state_dict(ck["model_state"])
    model.to(a.device).eval()
    gates = [m for m in model.modules() if isinstance(m, AttentionGate)]
    if not gates:
        raise SystemExit("this checkpoint has no attention gates (use an attn_unet / attn_resnet_unet run)")
    for g in gates:
        g.store = True

    ds = LandCoverDataset(a.data_root, a.split)
    ds.set_norm(ck["mean"], ck["std"])
    mean = np.array(ck["mean"], np.float32)[:, None, None]
    std = np.array(ck["std"], np.float32)[:, None, None]
    idx = np.unique(np.linspace(0, len(ds) - 1, a.n).round().astype(int))

    rows = []
    sums = None  # [level][class] -> (sum of attention, pixel count)
    with torch.no_grad():
        for i in idx:
            x, y = ds[i]
            pred = model(x.unsqueeze(0).to(a.device)).argmax(1)[0].cpu().numpy()
            maps = sorted([g.last[0, 0].cpu().numpy() for g in gates], key=lambda m: -m.shape[0])
            maps = [np.asarray(Image.fromarray(m).resize((256, 256), Image.BILINEAR)) for m in maps]
            rgb = (np.clip(x.numpy() * std + mean, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
            lab = y.numpy()
            pred = np.where(lab == IGNORE_INDEX, IGNORE_INDEX, pred)
            rows.append(np.concatenate([rgb, color_labels(lab), color_labels(pred)] + [heat(m) for m in maps], axis=1))
            if sums is None:
                sums = np.zeros((len(maps), NUM_CLASSES, 2))
            for l, m in enumerate(maps):
                for k in range(NUM_CLASSES):
                    sel = lab == k
                    sums[l, k] += (m[sel].sum(), sel.sum())

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"attention_{a.split.lower()}.png")
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    print("saved", path, f"({len(idx)} tiles; columns: RGB | label | prediction | attention fine -> coarse)")
    print("\nmean attention on pixels of each TRUE class (1.0 = fully open, 0 = suppressed):")
    print(f"{'level':<14}" + "".join(f"{n:>10}" for n in CLASS_NAMES))
    for l in range(sums.shape[0]):
        name = "finest" if l == 0 else ("coarsest" if l == sums.shape[0] - 1 else f"level {l + 1}")
        print(f"{name:<14}" + "".join(f"{sums[l, k, 0] / max(sums[l, k, 1], 1):10.3f}" for k in range(NUM_CLASSES)))


if __name__ == "__main__":
    main()
