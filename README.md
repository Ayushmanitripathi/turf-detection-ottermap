# Turf Detection - Ottermap ML Assessment

End-to-end semantic segmentation pipeline for detecting turf/grass regions in aerial GeoTIFF imagery and exporting GIS-compatible polygon predictions.

## Current Deliverable Status

- Raw training imagery and annotations are prepared under `data/raw/`.
- Patch preprocessing is complete:
  - Train: 595 image/mask patches
  - Validation: 177 image/mask patches
- A CPU-friendly U-Net checkpoint is provided at `weights/best_model.pth`.
- Current checkpoint metrics:
  - Epochs completed: 4
  - Best validation IoU: 0.5563
  - Best validation Dice: 0.6780
- Training-image predictions are available in `results/training_predictions/`.
- Validation patch comparisons are available in `results/validation_predictions/`.
- External generalization prediction is available in `results/external_predictions/`.
- GIS-ready GeoJSON samples are available in `gis_outputs/`.
- Technical summary PDF is generated at `technical_summary.pdf`.

## Project Structure

```text
turf-detection/
  inference.py                  Main inference entry point
  run_all.py                    End-to-end pipeline runner
  check_metrics.py              Prints current training metrics
  requirements.txt              Python dependencies
  README.md
  technical_summary.pdf
  data/
    raw/                        Provided GeoTIFF + GeoJSON files
    patches/                    Generated train/validation patches
    external/                   External generalization test imagery
  gis_outputs/                  Sample polygonized GeoJSON outputs
  results/
    training_predictions/       Full-image predictions and overlays
    validation_predictions/     Validation patch comparison overlays
    external_predictions/       External imagery prediction and overlay
    training_curves.png
  src/
    preprocess.py               Rasterize annotations and tile patches
    dataset.py                  PyTorch dataset and augmentation
    model.py                    SegFormer wrapper and LightUNet model
    train.py                    Training workflow
    predict_validation.py       Validation overlay generator
    gis_output.py               Mask-to-GeoJSON polygonization
    visualize.py                Overlay and curve plotting
    download_external.py        External test image generator
  weights/
    best_model.pth              Best trained checkpoint
    training_log.json           Epoch metrics
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Reproduce Preprocessing

The provided raw files are expected at:

```text
data/raw/1.tiff
data/raw/1.geojson
data/raw/2.tiff
data/raw/2.geojson
data/raw/3.tiff
data/raw/3.geojson
```

Run:

```bash
python src/preprocess.py
```

This rasterizes turf polygons and generates 512 x 512 train/validation patches with 50% overlap.

## Train

CPU-friendly U-Net training:

```bash
python src/train.py --model unet --epochs 30 --batch_size 4
```

SegFormer can also be used when Hugging Face downloads and compute resources are available:

```bash
python src/train.py --model segformer --epochs 30 --batch_size 2
```

The submitted checkpoint is `weights/best_model.pth`.

Check metrics:

```bash
python check_metrics.py
```

## Inference on New Imagery

Single GeoTIFF:

```bash
python inference.py --image input_image.tif --output results/inference/
```

Directory of GeoTIFF files:

```bash
python inference.py --input ./images/ --output results/batch/
```

Each run writes:

- `<image>_turf_prediction.geojson`
- `<image>_overlay.png`

## Generate Included Outputs

Training imagery:

```bash
python inference.py --image data/raw/1.tiff --output results/training_predictions/
python inference.py --image data/raw/2.tiff --output results/training_predictions/
python inference.py --image data/raw/3.tiff --output results/training_predictions/
```

Validation examples:

```bash
python src/predict_validation.py --limit 12
```

External generalization test:

```bash
python src/download_external.py
python inference.py --image data/external/external_florida_sports_complex.tiff --output results/external_predictions/
```

Training curves:

```bash
python -c "import sys; sys.path.insert(0,'src'); from visualize import plot_training_curves; plot_training_curves('weights/training_log.json', 'results/training_curves.png')"
```

## GIS Output

Prediction masks are polygonized with `rasterio.features.shapes` and saved as GeoJSON FeatureCollections. Sample outputs are in `gis_outputs/` and can be opened in QGIS, ArcGIS, geojson.io, or other GIS tools.

## Notes and Limitations

- The included external image is a synthetic georeferenced aerial-style sports complex because network access may not be available in the evaluation environment.
- The current checkpoint was trained on CPU for 4 epochs. More epochs, stronger validation splits, and real external NAIP/orthophoto imagery would likely improve generalization.
- The model tends to over-predict some green or low-texture regions; area filters and additional negative examples would reduce false positives.
