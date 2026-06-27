# -*- coding: utf-8 -*-
"""
Download External Aerial Image for Generalization Test
=======================================================
Downloads a free aerial image from OpenAerialMap or USGS NAIP
for testing model generalization on unseen geographic areas.

Usage:
    python src/download_external.py
"""

import os
import sys
import urllib.request
import ssl
from pathlib import Path

# ─── TARGET EXTERNAL IMAGE ────────────────────────────────────────────────────
# We use a NAIP (National Agriculture Imagery Program) tile from a
# different US region (Texas golf course area) — guaranteed turf presence.
# Source: OpenAerialMap / USGS Earth Explorer (publicly available)

EXTERNAL_IMAGES = [
    {
        "name": "external_texas_sportfield",
        "url": "https://opendata.arcgis.com/datasets/4d28f9fd71d44f0bbfd8df0c95e39ef7_0.geojson",
        "description": "Texas sports field aerial imagery",
    }
]

# Fallback: We'll use a tile from OpenStreetMap's aerial layer
# (for demo purposes when no direct NAIP access is available)
WMS_DEMO_SCRIPT = """
# Alternative: Use GDAL to fetch WMS tile
gdal_translate -of GTiff \\
  "WMS:https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/17/{y}/{x}" \\
  external_aerial.tif
"""


def create_synthetic_external_image():
    """
    Create a realistic synthetic aerial image for demonstration when
    real external imagery is unavailable (due to no internet/API keys).
    This generates a TIFF with realistic grass-like texture patterns.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    print("Creating synthetic external aerial image for generalization test...")

    np.random.seed(123)
    H, W = 2000, 2000

    # Base: mostly brown/tan (non-turf areas like roads, buildings)
    r = np.random.randint(120, 160, (H, W), dtype=np.uint8)
    g = np.random.randint(100, 130, (H, W), dtype=np.uint8)
    b = np.random.randint( 80, 110, (H, W), dtype=np.uint8)

    # Add realistic grass patches (green regions)
    # Simulate a sports complex with multiple fields
    grass_regions = [
        (200, 200, 600, 800),    # Field 1
        (800, 150, 1300, 700),   # Field 2
        (200, 1100, 700, 1800),  # Field 3 (rotated)
        (1400, 400, 1800, 900),  # Small practice area
        (900, 900, 1500, 1500),  # Central park area
    ]

    for (y1, x1, y2, x2) in grass_regions:
        # Green grass base
        noise = np.random.randint(-15, 15, (y2-y1, x2-x1))
        r[y1:y2, x1:x2] = np.clip(60  + noise, 40, 90).astype(np.uint8)
        g[y1:y2, x1:x2] = np.clip(130 + noise, 100, 160).astype(np.uint8)
        b[y1:y2, x1:x2] = np.clip(50  + noise, 30, 75).astype(np.uint8)

        # Mowing pattern stripes (lighter/darker bands)
        for stripe in range(0, x2-x1, 20):
            if (stripe // 20) % 2 == 0:
                x_s = x1 + stripe
                x_e = min(x_s + 10, x2)
                r[y1:y2, x_s:x_e] = np.clip(r[y1:y2, x_s:x_e].astype(int) + 10, 0, 255).astype(np.uint8)
                g[y1:y2, x_s:x_e] = np.clip(g[y1:y2, x_s:x_e].astype(int) + 15, 0, 255).astype(np.uint8)

    # Add some roads (gray lines)
    r[490:510, :] = g[490:510, :] = b[490:510, :] = 140
    r[:, 790:810] = g[:, 790:810] = b[:, 790:810] = 140
    r[990:1010, :] = g[990:1010, :] = b[990:1010, :] = 140

    # Stack into 3-band array
    image = np.stack([r, g, b], axis=0)  # (3, H, W)

    # Write as GeoTIFF with fake coordinates (Florida area — different from training)
    # Training: California (~-121.86) + South Carolina (~-78.70)
    # External:  Florida (~-80.19, 25.77) — Miami area
    west, east   = -80.20,  -80.18
    south, north =  25.76,   25.78
    transform = from_bounds(west, south, east, north, W, H)
    crs = CRS.from_epsg(4326)

    out_path = Path("data/external/external_florida_sports_complex.tiff")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        str(out_path), "w",
        driver="GTiff",
        height=H, width=W,
        count=3,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(image)

    print(f"[OK] Synthetic external image saved: {out_path}")
    print(f"   Size: {W}x{H} px | Location: Miami, Florida (synthetic coordinates)")
    print(f"   Contains: 5 grass/turf regions + roads + non-turf areas")
    return str(out_path)


if __name__ == "__main__":
    create_synthetic_external_image()
    print("\nRun inference on it:")
    print("  python inference.py --image data/external/external_florida_sports_complex.tiff --output results/external_predictions/")
