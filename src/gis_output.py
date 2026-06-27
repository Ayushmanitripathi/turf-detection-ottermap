# -*- coding: utf-8 -*-
"""
GIS Output Generation
======================
Converts binary prediction masks -> georeferenced GeoJSON polygons.

Given a prediction mask (numpy array) and the source TIFF metadata,
this module:
1. Polygonizes contiguous turf regions (rasterio.features.shapes)
2. Filters small spurious detections
3. Projects to WGS84 (EPSG:4326)
4. Writes output as a GeoJSON FeatureCollection
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional

import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.warp import transform_geom
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")


# Minimum area in pixels to keep a polygon (filters noise)
MIN_POLYGON_PIXELS = 100


def mask_to_geojson(
    mask: np.ndarray,
    transform,
    src_crs,
    output_path: str,
    source_name: str = "turf_prediction",
    min_pixels: int = MIN_POLYGON_PIXELS,
) -> dict:
    """
    Convert a binary prediction mask to a GeoJSON FeatureCollection.

    Args:
        mask:        (H, W) uint8 numpy array, 1 = turf, 0 = background
        transform:   Affine transform from the source rasterio dataset
        src_crs:     CRS of the source TIFF
        output_path: Where to save the .geojson file
        source_name: Name tag for the FeatureCollection
        min_pixels:  Minimum polygon area in pixels to include

    Returns:
        GeoJSON dict
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    # Extract polygon shapes from the mask
    features = []
    for geom, value in rasterio.features.shapes(mask_uint8, transform=transform):
        if value != 1:
            continue  # skip background
        poly = shape(geom)
        if poly.area < min_pixels * abs(transform.a * transform.e):
            continue  # too small

        # Reproject to WGS84 if source CRS is not already geographic
        if src_crs is not None:
            try:
                from rasterio.warp import transform_geom as rasterio_transform
                geom_wgs84 = rasterio_transform(
                    src_crs, "EPSG:4326", geom
                )
                poly = shape(geom_wgs84)
            except Exception:
                pass  # keep original projection

        features.append({
            "type": "Feature",
            "properties": {
                "class": "turf",
                "area_m2": round(poly.area, 2),
                "source": source_name,
            },
            "geometry": mapping(poly),
        })

    geojson = {
        "type": "FeatureCollection",
        "name": "Turf_Predictions",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"  [OK] GeoJSON saved: {output_path} ({len(features)} polygons)")
    return geojson


def stitch_mask_predictions(patch_preds: list, full_shape: tuple, patch_size: int, stride: int) -> np.ndarray:
    """
    Stitch overlapping patch predictions back into a full-resolution mask.
    Uses voting (sum) for overlapping regions.

    Args:
        patch_preds: List of (y, x, pred_patch) tuples
        full_shape:  (H, W) of the full image
        patch_size:  Size of each square patch
        stride:      Stride used during tiling

    Returns:
        full_mask: (H, W) float32 -- sum of votes (normalize externally)
    """
    H, W = full_shape
    vote_map  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y, x, pred in patch_preds:
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        ph = y_end - y
        pw = x_end - x
        vote_map[y:y_end, x:x_end]  += pred[:ph, :pw]
        count_map[y:y_end, x:x_end] += 1.0

    count_map = np.maximum(count_map, 1.0)
    avg_mask = vote_map / count_map
    return avg_mask


def save_gis_outputs_from_tiff(
    tiff_path: str,
    pred_mask: np.ndarray,
    out_path: str,
    min_pixels: int = MIN_POLYGON_PIXELS,
):
    """
    Convenience wrapper: read CRS/transform from source TIFF, then save GeoJSON.

    Args:
        tiff_path: Path to the original GeoTIFF (for CRS + transform)
        pred_mask: (H, W) binary prediction mask
        out_path:  Output GeoJSON path
        min_pixels: Minimum polygon area filter
    """
    with rasterio.open(tiff_path) as src:
        transform = src.transform
        crs       = src.crs

    source_name = Path(tiff_path).stem
    return mask_to_geojson(
        mask=pred_mask,
        transform=transform,
        src_crs=crs,
        output_path=out_path,
        source_name=source_name,
        min_pixels=min_pixels,
    )
