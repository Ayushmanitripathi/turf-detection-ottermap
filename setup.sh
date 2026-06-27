#!/bin/bash
# ============================================================
#  Turf Detection -- One-Click Setup for Linux/Mac
# ============================================================
set -e

echo "[SETUP] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[DONE] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. python src/preprocess.py       -- generate training patches"
echo "  2. python src/train.py            -- train the model"
echo "  3. python inference.py --image data/raw/1.tiff"
echo ""
echo "Or run everything at once:"
echo "  python run_all.py"
