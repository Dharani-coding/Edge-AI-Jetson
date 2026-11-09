@echo off
setlocal

rem ONNX sanity check on the test images. Run from anywhere.
rem Paths are based on this script's folder (%~dp0), not the current directory.

for %%I in ("%~dp0..") do set "WINDOWS_DIR=%%~fI"
for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"

set "MODEL=%WINDOWS_DIR%\yolov8n-seg.onnx"
set "IMAGES=%PROJECT_ROOT%\data\test_images"

uv run --project "%WINDOWS_DIR%" yolo predict model="%MODEL%" source="%IMAGES%\test_img1.jpeg" imgsz="[320,416]" || exit /b 1
uv run --project "%WINDOWS_DIR%" yolo predict model="%MODEL%" source="%IMAGES%\test_img2.jpeg" imgsz="[320,416]" || exit /b 1
