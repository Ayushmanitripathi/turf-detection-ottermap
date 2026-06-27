# -*- coding: utf-8 -*-
"""
run_all.py -- Full Pipeline Runner
====================================
Runs the complete turf detection pipeline end-to-end:
  1. Preprocess (rasterize + tile)
  2. Train model
  3. Generate external test image
  4. Run inference on all training images + external image
  5. Save results + GIS outputs

Usage:
    python run_all.py
    python run_all.py --skip_preprocess   (if patches already generated)
    python run_all.py --skip_train        (if weights already exist)
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  STEP: {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        sys.exit(1)
    print(f"[DONE] {desc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train",      action="store_true")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--model",      default="unet", choices=["unet", "segformer"])
    args = parser.parse_args()

    print("=" * 60)
    print("  TURF DETECTION -- FULL PIPELINE")
    print("=" * 60)

    # Step 1: Preprocess
    if not args.skip_preprocess:
        run("python src/preprocess.py", "Data Preprocessing")
    else:
        print("\n[SKIP] Preprocessing (--skip_preprocess)")

    # Step 2: Train
    if not args.skip_train:
        run(
            f"python src/train.py --model {args.model} --epochs {args.epochs} --batch_size {args.batch_size}",
            f"Training ({args.model}, {args.epochs} epochs)"
        )
    else:
        print("\n[SKIP] Training (--skip_train)")

    # Step 3: Generate external image
    run("python src/download_external.py", "Generate External Test Image")

    # Step 4: Inference on training images
    for i in [1, 2, 3]:
        run(
            f"python inference.py --image data/raw/{i}.tiff --output results/training_predictions/",
            f"Inference on Training Image {i}"
        )

    # Step 5: Inference on external image
    run(
        "python inference.py --image data/external/external_florida_sports_complex.tiff --output results/external_predictions/",
        "Inference on External Image (Generalization Test)"
    )

    # Step 6: Plot training curves
    if Path("weights/training_log.json").exists():
        run(
            "python -c \"import sys; sys.path.insert(0,'src'); from visualize import plot_training_curves; plot_training_curves('weights/training_log.json', 'results/training_curves.png')\"",
            "Plot Training Curves"
        )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    print("  results/training_predictions/   -- overlays on training data")
    print("  results/external_predictions/   -- generalization test results")
    print("  gis_outputs/                    -- GeoJSON predictions")
    print("  weights/best_model.pth          -- trained model")
    print("  results/training_curves.png     -- loss/IoU curves")


if __name__ == "__main__":
    main()
