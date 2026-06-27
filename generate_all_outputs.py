# -*- coding: utf-8 -*-
"""
generate_all_outputs.py
========================
Runs inference on all training TIFFs + external image.
Generates GeoJSON outputs, visual overlays, and GIS files.
Run this after training is complete.
"""
import sys, os, subprocess
from pathlib import Path

WEIGHTS = "weights/best_model.pth"
IMAGES  = [
    ("data/raw/1.tiff",      "results/training_predictions/"),
    ("data/raw/2.tiff",      "results/training_predictions/"),
    ("data/raw/3.tiff",      "results/training_predictions/"),
    ("data/external/external_florida_sports_complex.tiff", "results/external_predictions/"),
]
GIS_OUT = "gis_outputs/"

def run(cmd):
    print(f"\n>> {cmd}")
    r = subprocess.run(cmd, shell=True)
    return r.returncode == 0

print("="*60)
print("  GENERATING ALL OUTPUTS")
print("="*60)

for img_path, out_dir in IMAGES:
    if not Path(img_path).exists():
        print(f"[SKIP] {img_path} not found")
        continue
    ok = run(f"python inference.py --image {img_path} --output {out_dir} --weights {WEIGHTS}")
    if ok:
        # Copy geojson to gis_outputs too
        stem = Path(img_path).stem
        src = Path(out_dir) / f"{stem}_turf_prediction.geojson"
        dst = Path(GIS_OUT) / f"{stem}_turf_prediction.geojson"
        dst.parent.mkdir(exist_ok=True)
        if src.exists():
            import shutil
            shutil.copy(src, dst)
            print(f"  [GIS] Copied to {dst}")

# Plot training curves
if Path("weights/training_log.json").exists():
    run('python -c "import sys; sys.path.insert(0,\'src\'); from visualize import plot_training_curves; plot_training_curves(\'weights/training_log.json\', \'results/training_curves.png\')"')

print("\n" + "="*60)
print("  ALL OUTPUTS GENERATED!")
print("="*60)
print("\nFiles generated:")
for f in Path("results").rglob("*.png"):
    print(f"  [IMG] {f}")
for f in Path("gis_outputs").rglob("*.geojson"):
    print(f"  [GIS] {f}")
