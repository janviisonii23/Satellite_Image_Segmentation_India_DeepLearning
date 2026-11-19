import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights_from_counts(counts):
    """Inverse-square-root frequency, normalised to mean 1 (softer than 1/freq)."""
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    return (w / w.mean()).astype(np.float32)


def dice_loss(logits, target, ignore_index=IGNORE_INDEX, eps=1e-6):
    n = logits.shape[1]
    prob = logits.float().softmax(1)
    valid = target != ignore_index
    t = target.clone()
    t[~valid] = 0
    onehot = F.one_hot(t, n).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    prob, onehot = prob * v, onehot * v
    inter = (prob * onehot).sum((0, 2, 3))
    card = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (card + eps)).mean()


class CEDiceLoss(nn.Module):
    """Class-weighted cross-entropy + Dice, both ignoring padded pixels (label 255)."""

    def __init__(self, class_weights=None, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.register_buffer("w", None if class_weights is None else torch.as_tensor(class_weights))
        self.ce_weight, self.dice_weight = ce_weight, dice_weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits.float(), target, weight=self.w, ignore_index=IGNORE_INDEX)
        return self.ce_weight * ce + self.dice_weight * dice_loss(logits, target)


class ConfusionMeter:
    def __init__(self, n=NUM_CLASSES):
        self.n = n
        self.mat = torch.zeros(n, n, dtype=torch.int64)

    def update(self, pred, target):
        valid = target != IGNORE_INDEX
        idx = target[valid] * self.n + pred[valid]
        self.mat += torch.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n).cpu()

    def summary(self):
        m = self.mat.double()
        tp = m.diag()
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        rec = tp / (tp + fn).clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
        per = {CLASS_NAMES[c]: dict(iou=iou[c].item(), precision=prec[c].item(), recall=rec[c].item(),
                                    f1=f1[c].item(), support_px=int(m.sum(1)[c].item()))
               for c in range(self.n)}
        return dict(per_class=per, mIoU=iou.mean().item(), mF1=f1.mean().item(),
                    overall_acc=(tp.sum() / m.sum().clamp(min=1)).item(), confusion=self.mat.tolist())


def format_metrics(s, title=""):
    lines = [f"== {title} ==" if title else "",
             f"{'class':<10}{'IoU':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'pixels':>12}"]
    for name, d in s["per_class"].items():
        lines.append(f"{name:<10}{d['iou']:8.3f}{d['precision']:8.3f}{d['recall']:8.3f}"
                     f"{d['f1']:8.3f}{d['support_px']:12d}")
    lines.append(f"mIoU {s['mIoU']:.3f} | mF1 {s['mF1']:.3f} | overall acc {s['overall_acc']:.3f}")
    return "\n".join(l for l in lines if l)
