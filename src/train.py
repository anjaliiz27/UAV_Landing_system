"""Train the 4-enc/4-dec CNN. CrossEntropyLoss + Adam; best checkpoint by validation loss.

Run from repo root:
    python -m src.train --epochs 60
"""
import argparse, os, random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.segmentation_pipeline import SegNet
from src.preprocessing import PREP_DIR, NUM_CLASSES
from src.dataset import DroneSeg, split_stems, SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoints/best_model.pth"


@torch.no_grad()
def confusion(pred, target, n):
    return torch.bincount(n * target + pred, minlength=n * n).reshape(n, n)


def report(conf):
    tp = conf.diag().float()
    iou = tp / (conf.sum(0) + conf.sum(1) - conf.diag()).float().clamp(min=1)
    present = conf.sum(1) > 0
    return (tp.sum() / conf.sum()).item(), iou[present].mean().item(), iou


def class_weights(stems):
    """Sqrt-inverse-frequency weights from TRAIN masks only."""
    cnt = np.zeros(NUM_CLASSES)
    for s in stems:
        m = cv2.imread(f"{PREP_DIR}/masks/{s}.png", cv2.IMREAD_UNCHANGED)
        cnt += np.bincount(m.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    w = 1.0 / np.sqrt(np.maximum(cnt, 1) / cnt.sum())
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main(epochs, batch_size, lr):
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    tr_stems, va_stems = split_stems()
    print(f"train {len(tr_stems)} | val {len(va_stems)} | device {DEVICE}")
    tr = DataLoader(DroneSeg(tr_stems, augment=True), batch_size, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(DroneSeg(va_stems), batch_size, shuffle=False, num_workers=2)

    model = SegNet(NUM_CLASSES).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr_stems).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(DEVICE, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item()
        tl /= len(tr)

        model.eval(); vl = 0.0
        conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                vl += crit(out, y).item()
                conf += confusion(out.argmax(1).flatten(), y.flatten(), NUM_CLASSES)
        vl /= len(va); sched.step(vl)
        acc, miou, _ = report(conf)
        tag = ""
        if vl < best:
            best = vl; torch.save(model.state_dict(), CKPT); tag = "  <- saved best checkpoint"
        print(f"ep {ep:3d} | train {tl:.4f} | val {vl:.4f} | pixAcc {acc:.3f} | mIoU {miou:.3f}{tag}")

    # final per-class IoU of the best checkpoint
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for x, y in va:
            conf += confusion(model(x.to(DEVICE)).argmax(1).flatten(), y.to(DEVICE).flatten(), NUM_CLASSES)
    acc, miou, iou = report(conf)
    print(f"\nBEST CHECKPOINT: pixAcc {acc:.3f} | mIoU {miou:.3f}")
    for c, v in enumerate(iou.tolist()):
        print(f"  class {c:2d} IoU {v:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    main(a.epochs, a.batch_size, a.lr)
