#!/usr/bin/env python
"""Train the normal (classic) U-Net on the 4-class Sentinel-2 RGB tiles.

    python train.py --data_root DATASET
"""
import argparse
import contextlib
import csv
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from src.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, LandCoverDataset
from src.model import build_model, param_groups
from src.utils import CEDiceLoss, ConfusionMeter, class_weights_from_counts, format_metrics, set_seed


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="folder containing Train/ Validation/ Test/")
    p.add_argument("--model", default="unet",
                   choices=["unet", "attn_unet", "resnet_unet", "attn_resnet_unet"],
                   help="unet = classic (scratch) | attn_unet = classic + attention gates | "
                        "resnet_unet = ResNet encoder | attn_resnet_unet = ResNet encoder + attention gates")
    p.add_argument("--encoder", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--no_pretrained", action="store_true", help="ResNet models: random init instead of ImageNet")
    p.add_argument("--width", type=int, default=32, help="classic / attention U-Net: channels of the first block")
    p.add_argument("--levels", type=int, default=4, help="classic / attention U-Net: number of down-sampling steps")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="decoder lr (and everything for from-scratch models)")
    p.add_argument("--lr_enc", type=float, default=1e-4, help="pretrained ResNet encoder lr")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=3, help="epochs of linear learning-rate warm-up")
    p.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    p.add_argument("--patience", type=int, default=20, help="stop if val mIoU has not improved for N epochs")
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--class_weights", default="auto", choices=["auto", "none"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--limit", type=int, default=None, help="debug: use only N tiles per split")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    a.pretrained = a.model.endswith("resnet_unet") and not a.no_pretrained
    if a.name is None:
        a.name = (f"{a.model}_w{a.width}_l{a.levels}_s{a.seed}" if a.model in ("unet", "attn_unet") else
                  f"{a.model}_{a.encoder}_{'pre' if a.pretrained else 'scratch'}_s{a.seed}")
    return a


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, amp):
    # mixed precision only on CUDA; plain no-op context everywhere else
    return torch.autocast(device_type="cuda") if amp else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval()
    meter, total, n = ConfusionMeter(NUM_CLASSES), 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device, amp):
            logits = model(x)
        total += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        meter.update(logits.argmax(1), y)
    return total / max(n, 1), meter


def main():
    a = get_args()
    set_seed(a.seed)
    device = pick_device(a.device)
    amp = device.type == "cuda"
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    print(f"run: {a.name} | device: {device} | mixed precision: {amp}")

    # ---- data -------------------------------------------------------------
    train_ds = LandCoverDataset(a.data_root, "Train", augment=True, limit=a.limit)
    val_ds = LandCoverDataset(a.data_root, "Validation", augment=False, limit=a.limit)
    test_ds = LandCoverDataset(a.data_root, "Test", augment=False, limit=a.limit)
    if a.pretrained:  # must match what the ImageNet encoder was trained with
        mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
    else:  # from the TRAIN tiles only
        mean, std = train_ds.compute_stats()
    for ds in (train_ds, val_ds, test_ds):
        ds.set_norm(mean, std)
    counts = train_ds.class_counts()
    cw = class_weights_from_counts(counts) if a.class_weights == "auto" else None
    print("tiles train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    print("train class share:", {n: f"{100 * c / counts.sum():.1f}%" for n, c in zip(CLASS_NAMES, counts)})
    print("class weights:", None if cw is None else dict(zip(CLASS_NAMES, cw.round(2).tolist())))

    kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, drop_last=len(train_ds) >= a.batch_size, **kw)
    val_dl = DataLoader(val_ds, a.batch_size, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, a.batch_size, shuffle=False, **kw)

    # ---- model / optimiser -----------------------------------------------
    model = build_model(a.model, a.encoder, a.pretrained, a.width, a.levels, a.dropout, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f} M")
    criterion = CEDiceLoss(cw, a.ce_weight, a.dice_weight).to(device)
    if a.pretrained:
        groups = param_groups(model, a.lr_enc, a.lr, a.wd)  # small lr for the pretrained encoder
    else:
        groups = [{"params": list(model.parameters()), "lr": a.lr, "weight_decay": a.wd}]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(  # warm-up, then cosine decay
        opt, lambda e: min(1.0, (e + 1) / max(a.warmup, 1)) * max(0.01, 0.5 * (1 + math.cos(math.pi * e / a.epochs))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ---- training loop ----------------------------------------------------
    hist_path = os.path.join(out, "history.csv")
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mIoU"] +
                               [f"val_IoU_{c}" for c in CLASS_NAMES] + ["lr", "sec"])
    best, best_ep, bad = -1.0, 0, 0
    ckpt_path = os.path.join(out, "best.pt")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device, amp):
                logits = model(x)
            loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        vloss, meter = evaluate(model, val_dl, criterion, device, amp)
        s = meter.summary()
        ious = [s["per_class"][c]["iou"] for c in CLASS_NAMES]
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, tot / max(n, 1), vloss, s["mIoU"]] + ious +
                                   [opt.param_groups[-1]["lr"], round(time.time() - t0, 1)])
        print(f"ep {ep:3d} | train {tot / max(n, 1):.4f} | val {vloss:.4f} | val mIoU {s['mIoU']:.4f} | "
              + " ".join(f"{c[:5]} {i:.2f}" for c, i in zip(CLASS_NAMES, ious)) + f" | {time.time() - t0:.0f}s")
        if s["mIoU"] > best:
            best, best_ep, bad = s["mIoU"], ep, 0
            torch.save(dict(model_state=model.state_dict(), args=vars(a), mean=mean, std=std,
                            epoch=ep, val_mIoU=best), ckpt_path)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep} (best epoch {best_ep}, val mIoU {best:.4f})")
                break

    # ---- final evaluation with the best checkpoint --------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    results = dict(args=vars(a), best_epoch=best_ep, n_params=n_params, mean=mean, std=std,
                   class_weights=None if cw is None else cw.tolist())
    for split, dl in (("val", val_dl), ("test", test_dl)):
        _, meter = evaluate(model, dl, criterion, device, amp)
        results[split] = meter.summary()
        print(format_metrics(results[split], f"{split} (best epoch {best_ep})"))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved to", out)


if __name__ == "__main__":
    main()
