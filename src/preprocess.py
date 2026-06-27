# -*- coding: utf-8 -*-
"""
Preprocessing Pipeline
======================
1. Reads each GeoTIFF + its GeoJSON annotation
2. Rasterizes GeoJSON polygons -> binary mask (1=turf, 0=background)
3. Tiles image + mask into 512×512 overlapping patches (stride=256)
4. Splits into train/val sets and saves as PNG files
"""

import os
import json
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape, box
from PIL import Image
import cv2
from pathlib import Path
import random
import shutil

# --- CONFIG -----------------------------------------------------------------
RAW_DIR        = Path("data/raw")
SHAPEFILE_DIR  = Path("../feature_layers (1)/feature_layers/ShapeFile")
PATCHES_DIR    = Path("data/patches")
PATCH_SIZE     = 512
STRIDE         = 256          # Overlap for more training samples
VAL_SPLIT      = 0.2          # 20% validation
RANDOM_SEED    = 42
MIN_TURF_PCT   = 0.01         # Skip patches with <1% turf (mostly background)
# -----------------------------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def normalize_image(img_array):
    """Normalize image to 0-255 uint8 (handles various TIFF formats)."""
    if img_array.dtype == np.uint8:
        return img_array
    # Percentile stretch for better contrast
    p2, p98 = np.percentile(img_array, (2, 98))
    img_array = np.clip(img_array, p2, p98)
    img_array = ((img_array - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
    return img_array


def rasterize_geojson(tiff_path, geojson_path, shapefile_path=None):
    """
    Read a GeoTIFF and its GeoJSON annotation.

    Since the TIFFs in this dataset have NO embedded georeference (CRS=None,
    pixel-space transform), we use the ShapeFile footprint GeoJSON to compute
    the correct affine transform: it maps pixel (col, row) to WGS84 (lon, lat).

    Args:
        tiff_path:      Path to the .tiff file
        geojson_path:   Path to the annotation .geojson (turf polygons)
        shapefile_path: Path to the ShapeFile footprint .geojson (TIFF extent in WGS84)

    Returns:
        image (H, W, 3) uint8 numpy array
        mask  (H, W)    uint8 binary mask (1=turf, 0=background)
        transform:      Affine transform (pixel -> WGS84)
        crs:            EPSG:4326
    """
    from rasterio.transform import from_bounds as rasterio_from_bounds
    from rasterio.crs import CRS

    print(f"  Reading TIFF: {tiff_path.name}")
    with rasterio.open(tiff_path) as src:
        data   = src.read()          # (bands, H, W)
        height = src.height
        width  = src.width

    # Build RGB image
    bands    = min(data.shape[0], 3)
    rgb      = data[:bands]
    rgb_norm = np.stack([normalize_image(rgb[b]) for b in range(bands)], axis=0)
    if bands < 3:
        rgb_norm = np.repeat(rgb_norm, 3, axis=0)
    image = rgb_norm.transpose(1, 2, 0)   # (H, W, 3)

    # -- Compute proper affine transform from ShapeFile footprint --------------
    # The ShapeFile GeoJSON is a single polygon defining the TIFF's geographic
    # bounding box in WGS84. We use its bounds to build from_bounds() transform.
    if shapefile_path is not None and Path(shapefile_path).exists():
        print(f"  Reading footprint: {Path(shapefile_path).name}")
        fp_gdf   = gpd.read_file(shapefile_path)
        if fp_gdf.crs is None:
            fp_gdf = fp_gdf.set_crs("EPSG:4326")
        west, south, east, north = fp_gdf.total_bounds  # (minx, miny, maxx, maxy)
        print(f"  Geographic bounds: W={west:.5f} S={south:.5f} E={east:.5f} N={north:.5f}")
    else:
        # Fallback: derive bounds from annotation polygon extents
        print(f"  No shapefile found -- inferring bounds from GeoJSON annotations")
        ann_gdf  = gpd.read_file(geojson_path)
        if ann_gdf.crs is None:
            ann_gdf = ann_gdf.set_crs("EPSG:4326")
        west, south, east, north = ann_gdf.total_bounds

    # from_bounds(west, south, east, north, width, height)
    transform = rasterio_from_bounds(west, south, east, north, width, height)
    crs       = CRS.from_epsg(4326)
    # -------------------------------------------------------------------------

    # Load annotation polygons
    print(f"  Reading GeoJSON: {geojson_path.name}")
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Create binary mask by rasterizing polygons using the computed transform
    print(f"  Rasterizing {len(gdf)} polygons...")
    from rasterio.features import rasterize as rio_rasterize
    valid_geoms = [geom for geom in gdf.geometry if geom is not None and geom.is_valid]
    print(f"  Valid geometries: {len(valid_geoms)}")

    if len(valid_geoms) == 0:
        print("  WARNING: No valid geometries!")
        mask = np.zeros((height, width), dtype=np.uint8)
    else:
        shapes_iter = ((geom.__geo_interface__, 1) for geom in valid_geoms)
        mask = rio_rasterize(
            shapes=shapes_iter,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

    turf_pct = mask.mean() * 100
    print(f"  Image: {width}x{height}px  |  Mask turf coverage: {turf_pct:.2f}%")
    return image, mask, transform, crs


def extract_patches(image, mask, patch_size=512, stride=256):
    """Slide a window over the image and extract patches."""
    H, W = image.shape[:2]
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch  = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            patches.append((img_patch, mask_patch, y, x))

    # Also grab the bottom-right corner if not covered
    if H % stride != 0:
        y = H - patch_size
        for x in range(0, W - patch_size + 1, stride):
            img_patch  = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            patches.append((img_patch, mask_patch, y, x))
    if W % stride != 0:
        x = W - patch_size
        for y in range(0, H - patch_size + 1, stride):
            img_patch  = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            patches.append((img_patch, mask_patch, y, x))

    return patches


def save_patches(patches, split, source_idx, out_dir, min_turf_pct=0.01):
    """Save patches as PNG files under out_dir/{split}/images/ and /masks/."""
    img_dir  = out_dir / split / "images"
    mask_dir = out_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, (img_patch, mask_patch, y, x) in enumerate(patches):
        # Filter near-empty patches
        if mask_patch.mean() < min_turf_pct and random.random() > 0.3:
            continue  # keep 30% of background-only patches for balance

        fname = f"src{source_idx}_y{y}_x{x}.png"
        Image.fromarray(img_patch).save(img_dir / fname)
        Image.fromarray(mask_patch * 255).save(mask_dir / fname)
        saved += 1
    return saved


def run_preprocessing():
    print("=" * 60)
    print("  TURF DETECTION -- PREPROCESSING PIPELINE")
    print("=" * 60)

    out_dir = PATCHES_DIR
    # Clean and recreate
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    all_patches_per_source = []

    for idx in [1, 2, 3]:
        tiff_path    = RAW_DIR / f"{idx}.tiff"
        geojson_path = RAW_DIR / f"{idx}.geojson"

        if not tiff_path.exists() or not geojson_path.exists():
            print(f"[!] Skipping source {idx} -- files not found")
            continue

        shapefile_path = SHAPEFILE_DIR / f"{idx}.geojson"
        print(f"\n[Source {idx}]")
        image, mask, transform, crs = rasterize_geojson(
            tiff_path, geojson_path,
            shapefile_path=shapefile_path if shapefile_path.exists() else None
        )
        patches = extract_patches(image, mask, PATCH_SIZE, STRIDE)
        print(f"  Extracted {len(patches)} raw patches")
        all_patches_per_source.append((idx, patches))

    # Gather all patches, shuffle, split
    all_patches = []
    for source_idx, patches in all_patches_per_source:
        for p in patches:
            all_patches.append((source_idx, p))

    random.shuffle(all_patches)
    n_val   = int(len(all_patches) * VAL_SPLIT)
    val_set   = all_patches[:n_val]
    train_set = all_patches[n_val:]

    print(f"\n[Split] Total: {len(all_patches)} | Train: {len(train_set)} | Val: {len(val_set)}")

    # Save
    train_saved = 0
    for source_idx, patch in train_set:
        img_patch, mask_patch, y, x = patch
        fname = f"src{source_idx}_y{y}_x{x}.png"
        img_dir  = out_dir / "train" / "images"
        mask_dir = out_dir / "train" / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        if mask_patch.mean() < MIN_TURF_PCT and random.random() > 0.3:
            continue
        Image.fromarray(img_patch).save(img_dir / fname)
        Image.fromarray(mask_patch * 255).save(mask_dir / fname)
        train_saved += 1

    val_saved = 0
    for source_idx, patch in val_set:
        img_patch, mask_patch, y, x = patch
        fname = f"src{source_idx}_y{y}_x{x}.png"
        img_dir  = out_dir / "val" / "images"
        mask_dir = out_dir / "val" / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(img_patch).save(img_dir / fname)
        Image.fromarray(mask_patch * 255).save(mask_dir / fname)
        val_saved += 1

    print(f"\nSaved {train_saved} training patches")
    print(f"Saved {val_saved} validation patches")
    print(f"Patches saved to: {out_dir.resolve()}")
    print("=" * 60)
    print("Preprocessing complete!")


if __name__ == "__main__":
    run_preprocessing()
