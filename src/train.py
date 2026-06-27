# -*- coding: utf-8 -*-
"""
Training Script -- Turf Segmentation
=====================================
Fine-tunes SegFormer-B0 (or LightUNet) on aerial turf imagery.

Usage:
    python src/train.py [--epochs 30] [--batch_size 2] [--lr 6e-5] [--model segformer|unet]
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataloaders
from model import build_model, CombinedLoss


# --- CONFIG -----------------------------------------------------------------
DEFAULT_EPOCHS     = 30
DEFAULT_BATCH      = 2
DEFAULT_LR         = 6e-5
DEFAULT_PATCHES    = "data/patches"
DEFAULT_WEIGHTS    = "weights/best_model.pth"
DEFAULT_LOG        = "weights/training_log.json"
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------


def compute_metrics(logits, targets):
    """Compute IoU and Dice score for turf class."""
    preds = logits.argmax(dim=1)  # (B, H, W)

    # Flatten
    p = preds.view(-1).cpu().numpy()
    t = targets.view(-1).cpu().numpy()

    # Turf class = 1
    tp = ((p == 1) & (t == 1)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()

    iou  = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    return float(iou), float(dice)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0
    total_dice = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        iou, dice = compute_metrics(logits.detach(), masks.detach())
        total_loss += loss.item()
        total_iou  += iou
        total_dice += dice
        n_batches  += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.3f}")

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    total_dice = 0.0
    n_batches  = 0

    for images, masks in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = criterion(logits, masks)

        iou, dice = compute_metrics(logits, masks)
        total_loss += loss.item()
        total_iou  += iou
        total_dice += dice
        n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches


def train(args):
    print("=" * 60)
    print("  TURF DETECTION -- TRAINING")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  LR      : {args.lr}")
    print("=" * 60)

    # Build model
    use_segformer = (args.model == "segformer")
    model, model_type = build_model(use_segformer=use_segformer)
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # Data
    train_loader, val_loader = get_dataloaders(
        args.patches_dir,
        batch_size=args.batch_size,
        num_workers=0
    )

    # Optimizer + Scheduler + Loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedLoss(dice_weight=0.5)

    # Training loop
    best_val_iou = 0.0
    start_epoch  = 1
    log          = []

    # --- Resume from checkpoint if requested ---
    if args.resume and Path(args.weights).exists():
        print(f"\n[RESUME] Loading checkpoint: {args.weights}")
        ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_val_iou = ckpt.get("val_iou", 0.0)
        start_epoch  = ckpt.get("epoch", 0) + 1
        print(f"  Resuming from epoch {start_epoch} | Best Val IoU so far: {best_val_iou:.4f}")
        # Load existing log
        if Path(args.log).exists():
            with open(args.log) as f:
                log = json.load(f)
    # ------------------------------------------

    weights_dir = Path(args.weights).parent
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n[Epoch {epoch:02d}/{args.epochs}]")

        tr_loss, tr_iou, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_iou, vl_dice = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Train -- loss: {tr_loss:.4f} | IoU: {tr_iou:.4f} | Dice: {tr_dice:.4f}")
        print(f"  Val   -- loss: {vl_loss:.4f} | IoU: {vl_iou:.4f} | Dice: {vl_dice:.4f}  [{elapsed:.1f}s]")

        entry = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_iou": round(tr_iou, 4), "train_dice": round(tr_dice, 4),
            "val_loss":   round(vl_loss, 4), "val_iou":   round(vl_iou, 4), "val_dice":   round(vl_dice, 4),
        }
        log.append(entry)

        # Save best model
        if vl_iou > best_val_iou:
            best_val_iou = vl_iou
            torch.save({
                "epoch":      epoch,
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "val_iou":    vl_iou,
                "val_dice":   vl_dice,
            }, args.weights)
            print(f"  * Best model saved (IoU={vl_iou:.4f})")

        # Save log
        with open(args.log, "w") as f:
            json.dump(log, f, indent=2)

    # Always save final checkpoint too
    final_path = str(args.weights).replace(".pth", "_final.pth")
    torch.save({
        "epoch":      args.epochs,
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "val_iou":    vl_iou,
        "val_dice":   vl_dice,
    }, final_path)

    print(f"\n{'='*60}")
    print(f"[OK] Training complete!")
    print(f"   Best Val IoU : {best_val_iou:.4f}")
    print(f"   Weights saved: {args.weights}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train turf segmentation model")
    parser.add_argument("--model",       default="segformer", choices=["segformer", "unet"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--patches_dir", default=DEFAULT_PATCHES)
    parser.add_argument("--weights",     default=DEFAULT_WEIGHTS)
    parser.add_argument("--log",         default=DEFAULT_LOG)
    parser.add_argument("--resume",      action="store_true", help="Resume training from existing checkpoint")
    args = parser.parse_args()
    train(args)
