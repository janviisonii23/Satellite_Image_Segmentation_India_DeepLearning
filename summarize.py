#!/usr/bin/env python
"""Compare runs: mean +- std over seeds for every configuration.
    python summarize.py --runs runs            # test split
    python summarize.py --runs runs --split val
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--split", default="test", choices=["val", "test"])
a = ap.parse_args()

groups = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(a.runs, "*", "metrics.json"))):
    m = json.load(open(p))
    c = m["args"]
    mdl = c.get("model", "unet")
    if mdl in ("unet", "attn_unet"):
        key = f'{mdl} (scratch) w{c.get("width", 32)} l{c.get("levels", 4)}'
    else:
        key = f'{mdl} {c["encoder"]} ({"ImageNet" if c.get("pretrained") else "scratch"})'
    groups[key].append(m[a.split])

if not groups:
    raise SystemExit(f"no metrics.json found under {a.runs}/")
classes = list(next(iter(groups.values()))[0]["per_class"].keys())
head = ["model", "runs", "mIoU"] + [f"IoU {c}" for c in classes] + ["barren prec", "barren rec"]
print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
ms = lambda v: f"{np.mean(v):.3f}±{np.std(v):.3f}"
for k, rs in groups.items():
    row = [k, str(len(rs)), ms([r["mIoU"] for r in rs])]
    row += [ms([r["per_class"][c]["iou"] for r in rs]) for c in classes]
    row += [ms([r["per_class"]["barren"]["precision"] for r in rs]), ms([r["per_class"]["barren"]["recall"] for r in rs])]
    print("| " + " | ".join(row) + " |")
