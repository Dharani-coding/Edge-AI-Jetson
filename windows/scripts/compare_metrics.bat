@echo off
setlocal

rem Score the Jetson COCO runs and compare them with the Windows ONNX baseline.
rem Run from anywhere: paths are based on this script's folder (%~dp0).

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%" || exit /b 1

rem 1. Score each Jetson run (writes metrics.json next to its predictions.json)
uv run --project windows --with pycocotools python scripts\coco_metrics.py --pred results\accuracy\jetson_fp16\predictions.json || exit /b 1
uv run --project windows --with pycocotools python scripts\coco_metrics.py --pred results\accuracy\jetson_fp32\predictions.json || exit /b 1

rem 2. Compare both against the Windows ONNX baseline (also writes results\accuracy\comparison.csv)
uv run --project windows python scripts\compare_metrics.py || exit /b 1
