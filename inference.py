# -*- coding: utf-8 -*-
"""
Inference Script -- Turf Detection
===================================
Run trained model on any aerial GeoTIFF image.

Usage:
    python inference.py --image input_image.tif
    python inference.py --image input_image.tif --output results/ --weights weights/best_model.pth
    python inference.py --input ./images/          (batch mode)

Outputs:
    - Prediction GeoJSON in --output dir
    - Visual overlay PNG in --output dir
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import TurfSegFormer, LightUNet
from gis_output import save_gis_outputs_from_tiff, stitch_mask_predictions
from visualize import save_prediction_overlay


# --- CONFIG -----------------------------------------------------------------
PATCH_SIZE   = 512
STRIDE       = 256
THRESHOLD    = 0.7      # Conservative threshold to reduce rooftop false positives
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# -----------------------------------------------------------------------------


def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """Normalize TIFF bands to uint8 (2-98 percentile stretch)."""
    if img_array.dtype == np.uint8:
        return img_array
    p2, p98 = np.percentile(img_array, (2, 98))
    img_array = np.clip(img_array, p2, p98)
    img_array = ((img_array - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
    return img_array


def load_tiff_as_rgb(tiff_path: str):
    """
    Read a GeoTIFF and return:
        image: (H, W, 3) uint8 RGB numpy array
        transform: rasterio Affine transform
        crs: rasterio CRS
    """
    with rasterio.open(tiff_path) as src:
        data      = src.read()
        transform = src.transform
        crs       = src.crs

    bands = min(data.shape[0], 3)
    rgb   = data[:bands]
    rgb_norm = np.stack([normalize_image(rgb[b]) for b in range(bands)], axis=0)
    if bands < 3:
        rgb_norm = np.repeat(rgb_norm, 3, axis=0)

    image = rgb_norm.transpose(1, 2, 0)  # (H, W, 3)
    return image, transform, crs


def preprocess_patch(img_patch: np.ndarray) -> torch.Tensor:
    """Convert numpy RGB patch to normalized model input tensor."""
    pil = Image.fromarray(img_patch).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(pil)
    tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def load_model(weights_path: str) -> tuple:
    """Load model from checkpoint. Auto-detects architecture."""
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model_type = checkpoint.get("model_type", "segformer")

    if model_type == "segformer":
        model = TurfSegFormer(num_classes=2, pretrained=False)
    else:
        model = LightUNet(in_channels=3, num_classes=2)

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(DEVICE)
    model.eval()

    val_iou  = checkpoint.get("val_iou", "N/A")
    val_dice = checkpoint.get("val_dice", "N/A")
    epoch    = checkpoint.get("epoch", "N/A")
    print(f"[OK] Model loaded: {model_type} | Epoch {epoch} | Val IoU={val_iou:.4f} | Dice={val_dice:.4f}")
    return model, model_type


@torch.no_grad()
def run_inference(model, image: np.ndarray, patch_size: int = PATCH_SIZE, stride: int = STRIDE, threshold: float = THRESHOLD) -> np.ndarray:
    """
    Tile the image, run model on each patch, stitch back.

    Returns:
        pred_mask: (H, W) binary uint8 numpy array (1=turf, 0=background)
    """
    H, W = image.shape[:2]
    patch_preds = []

    # Generate tile coordinates
    coords = []
    for y in range(0, max(H - patch_size + 1, 1), stride):
        for x in range(0, max(W - patch_size + 1, 1), stride):
            y = min(y, H - patch_size)
            x = min(x, W - patch_size)
            coords.append((y, x))

    # Deduplicate
    coords = list(set(coords))

    total = len(coords)
    print(f"  Running inference on {total} patches...")

    for i, (y, x) in enumerate(coords):
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  Patch {i+1}/{total}", end="\r")

        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        img_patch = image[y:y_end, x:x_end]

        # Pad if needed
        ph, pw = img_patch.shape[:2]
        if ph < patch_size or pw < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            padded[:ph, :pw] = img_patch
            img_patch = padded

        tensor = preprocess_patch(img_patch).to(DEVICE)
        logits = model(tensor)                        # (1, 2, H, W)
        probs  = torch.softmax(logits, dim=1)[0, 1]  # (H, W) turf probability
        probs_np = probs.cpu().numpy()

        patch_preds.append((y, x, probs_np))

    print()  # newline after progress

    # Stitch
    vote_map  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    for y, x, pred in patch_preds:
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        ph = y_end - y
        pw = x_end - x
        vote_map[y:y_end, x:x_end]  += pred[:ph, :pw]
        count_map[y:y_end, x:x_end] += 1.0

    count_map  = np.maximum(count_map, 1.0)
    prob_map   = vote_map / count_map
    pred_mask  = (prob_map >= threshold).astype(np.uint8)

    turf_pct = pred_mask.mean() * 100
    print(f"  Turf coverage: {turf_pct:.1f}%")
    return pred_mask


def refine_turf_mask(
    image: np.ndarray,
    pred_mask: np.ndarray,
    min_component_pixels: int = 1200,
) -> np.ndarray:
    """
    Reduce common false positives such as gray/tan roofs and roads.

    The learned model provides the main segmentation signal. This post-process
    keeps only regions with a vegetation-like color response and removes small
    connected components. It is intentionally conservative because rooftop false
    positives are more harmful for GIS delivery than missing some dry turf.
    """
    rgb = image.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    saturation = (maxc - minc) / (maxc + 1e-6)
    exg = 2 * g - r - b

    vegetation_like = (
        (g > 45)
        & (g >= r - 8)
        & (g >= b + 6)
        & (exg > 4)
        & (saturation > 0.08)
    )

    refined = ((pred_mask > 0) & vegetation_like).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    cleaned = np.zeros_like(refined)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_pixels:
            cleaned[labels == label] = 1

    before = pred_mask.mean() * 100
    after = cleaned.mean() * 100
    print(f"  Post-process coverage: {before:.1f}% -> {after:.1f}%")
    return cleaned


def process_single_image(
    tiff_path: str,
    weights: str,
    out_dir: str,
    threshold: float = THRESHOLD,
    postprocess: bool = True,
    min_component_pixels: int = 400,
):
    """Full inference pipeline for one image."""
    tiff_path = Path(tiff_path)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = tiff_path.stem

    print(f"\n{'='*55}")
    print(f"  Processing: {tiff_path.name}")
    print(f"{'='*55}")

    # Load image
    print("  Loading TIFF...")
    image, transform, crs = load_tiff_as_rgb(str(tiff_path))
    H, W = image.shape[:2]
    print(f"  Image size: {W}×{H} px")

    # Load model
    model, _ = load_model(weights)

    # Inference
    t0 = time.time()
    pred_mask = run_inference(model, image, threshold=threshold)
    if postprocess:
        pred_mask = refine_turf_mask(
            image=image,
            pred_mask=pred_mask,
            min_component_pixels=min_component_pixels,
        )
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.1f}s")

    # GIS Output
    geojson_path = out_dir / f"{stem}_turf_prediction.geojson"
    save_gis_outputs_from_tiff(
        tiff_path=str(tiff_path),
        pred_mask=pred_mask,
        out_path=str(geojson_path),
    )

    # Visual Overlay
    overlay_path = out_dir / f"{stem}_overlay.png"
    # Resize image for display if very large
    display_image = image
    display_mask  = pred_mask
    if H > 3000 or W > 3000:
        scale = 2000 / max(H, W)
        nH, nW = int(H * scale), int(W * scale)
        display_image = np.array(Image.fromarray(image).resize((nW, nH), Image.BILINEAR))
        display_mask  = np.array(Image.fromarray(pred_mask * 255).resize((nW, nH), Image.NEAREST)) // 255

    save_prediction_overlay(
        image=display_image,
        pred_mask=display_mask,
        out_path=str(overlay_path),
        title=f"Turf Detection -- {stem}",
    )

    print("[DONE]")
    print(f"  GeoJSON output : {geojson_path}")
    print(f"  Overlay image  : {overlay_path}")
    return pred_mask


def main():
    parser = argparse.ArgumentParser(
        description="Turf Detection Inference -- Ottermap Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --image input_image.tif
  python inference.py --image my_field.tif --output results/my_field/ --weights weights/best_model.pth
  python inference.py --input ./images/   (batch -- processes all .tif/.tiff files)
        """
    )
    parser.add_argument("--image",   type=str, default=None, help="Path to a single GeoTIFF image")
    parser.add_argument("--input",   type=str, default=None, help="Directory of GeoTIFF images (batch mode)")
    parser.add_argument("--output",  type=str, default="results/inference/", help="Output directory")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth", help="Model weights .pth file")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Turf probability threshold [0-1]")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable color/area post-processing")
    parser.add_argument("--min-component-pixels", type=int, default=1200, help="Minimum connected turf region size")
    args = parser.parse_args()

    if not args.image and not args.input:
        parser.error("Provide either --image or --input")

    if not Path(args.weights).exists():
        print(f"[ERROR] Weights not found: {args.weights}")
        print("  Run training first: python src/train.py")
        sys.exit(1)

    thr = args.threshold
    postprocess = not args.no_postprocess

    if args.image:
        # Single image
        process_single_image(
            args.image,
            args.weights,
            args.output,
            threshold=thr,
            postprocess=postprocess,
            min_component_pixels=args.min_component_pixels,
        )

    elif args.input:
        # Batch mode
        input_dir = Path(args.input)
        tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
        if not tiff_files:
            print(f"[ERROR] No .tif/.tiff files found in {input_dir}")
            sys.exit(1)
        print(f"Found {len(tiff_files)} images to process")
        for tiff_path in tiff_files:
            out_sub = Path(args.output) / tiff_path.stem
            process_single_image(
                str(tiff_path),
                args.weights,
                str(out_sub),
                threshold=thr,
                postprocess=postprocess,
                min_component_pixels=args.min_component_pixels,
            )

    print(f"\n[DONE] All outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
