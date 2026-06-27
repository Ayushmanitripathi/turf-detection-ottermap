@echo off
REM ============================================================
REM  Turf Detection -- One-Click Setup for Windows
REM ============================================================
echo [SETUP] Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)
echo.
echo [DONE] Setup complete!
echo.
echo Next steps:
echo   1. python src/preprocess.py       -- generate training patches
echo   2. python src/train.py            -- train the model
echo   3. python inference.py --image data/raw/1.tiff
echo.
echo Or run everything at once:
echo   python run_all.py
pause
