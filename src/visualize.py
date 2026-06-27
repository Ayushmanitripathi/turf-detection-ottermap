# -*- coding: utf-8 -*-
"""
Visualization Utilities
========================
Creates visual overlays of turf predictions on aerial imagery.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


TURF_COLOR = (50, 205, 50)    # Lime green overlay
OVERLAY_ALPHA = 0.45


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color=TURF_COLOR, alpha=OVERLAY_ALPHA) -> np.ndarray:
    """
    Overlay a binary mask on a BGR/RGB image with a colored fill + contour.

    Args:
        image: (H, W, 3) uint8 RGB array
        mask:  (H, W) binary uint8 (0 or 1)
        color: RGB tuple for the overlay
        alpha: Blending strength

    Returns:
        overlay: (H, W, 3) uint8 RGB image with colored turf regions
    """
    overlay = image.copy().astype(np.float32)
    colored = np.zeros_like(image, dtype=np.float32)
    colored[mask == 1] = color

    # Blend
    blend = overlay * (1 - alpha) + colored * alpha
    result = np.clip(blend, 0, 255).astype(np.uint8)

    # Draw contours for clarity
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def save_comparison_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_path: str,
    title: str = "Turf Segmentation",
):
    """
    Save a 3-panel figure: Original | Ground Truth | Prediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(image)
    axes[0].set_title("Original Aerial Image", fontsize=11)
    axes[0].axis("off")

    if gt_mask is not None:
        gt_overlay = overlay_mask_on_image(image, gt_mask, color=(50, 205, 50))
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth (Turf)", fontsize=11)
    else:
        axes[1].imshow(np.zeros_like(image))
        axes[1].set_title("Ground Truth (N/A)", fontsize=11)
    axes[1].axis("off")

    pred_overlay = overlay_mask_on_image(image, pred_mask, color=(255, 165, 0))  # Orange for prediction
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction (Turf)", fontsize=11)
    axes[2].axis("off")

    # Legend
    gt_patch   = mpatches.Patch(color=(50/255, 205/255, 50/255),   label="Ground Truth Turf")
    pred_patch = mpatches.Patch(color=(255/255, 165/255, 0/255), label="Predicted Turf")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center", ncol=2, fontsize=10, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved overlay: {out_path}")


def save_prediction_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    out_path: str,
    title: str = "Turf Prediction",
):
    """Save a single prediction overlay (no ground truth)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].imshow(image)
    axes[0].set_title("Original Aerial Image", fontsize=11)
    axes[0].axis("off")

    pred_overlay = overlay_mask_on_image(image, pred_mask, color=(255, 165, 0))
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Predicted Turf (orange)", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved overlay: {out_path}")


def plot_training_curves(log_path: str, out_path: str):
    """Plot training + validation loss/IoU curves from the JSON log."""
    import json
    with open(log_path) as f:
        log = json.load(f)

    epochs     = [e["epoch"]      for e in log]
    tr_loss    = [e["train_loss"] for e in log]
    vl_loss    = [e["val_loss"]   for e in log]
    tr_iou     = [e["train_iou"]  for e in log]
    vl_iou     = [e["val_iou"]    for e in log]
    tr_dice    = [e["train_dice"] for e in log]
    vl_dice    = [e["val_dice"]   for e in log]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves -- Turf Segmentation", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, tr_loss, label="Train", color="#2196F3")
    axes[0].plot(epochs, vl_loss, label="Val",   color="#FF5722")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_iou, label="Train", color="#2196F3")
    axes[1].plot(epochs, vl_iou, label="Val",   color="#FF5722")
    axes[1].set_title("IoU"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, tr_dice, label="Train", color="#2196F3")
    axes[2].plot(epochs, vl_dice, label="Val",   color="#FF5722")
    axes[2].set_title("Dice Score"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Training curves saved: {out_path}")
