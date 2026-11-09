#!/usr/bin/env bash
# Clean-build the C++ app, then run the FP16 and FP32 engines on data/test_images.
#
#   ./test_engine_predict.sh
#
# Output (annotated images + <name>.txt detections + run log):
#   data/results/jetson_fp16/
#   data/results/jetson_fp32/
#

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)       # .../Edge-AI-Jetson/jetson
REPO_DIR=$(dirname "$SCRIPT_DIR")               # .../Edge-AI-Jetson
SOURCE_DIR="$SCRIPT_DIR/cpp"
BUILD_DIR="/mnt/usb-drive/build/edge-ai-inference"
MODELS_DIR="/mnt/usb-drive/models"
IMAGES_DIR="$REPO_DIR/data/test_images"
RESULTS_DIR="$REPO_DIR/data/results"

# Engine names follow build_engines.sh: <model>_<HxW>_<precision>_trt<version>.engine
MODEL="yolov8n-seg"
SIZE="320x416"
TRT_VERSION=$(dpkg-query -W -f='${Version}' libnvinfer8 | cut -d- -f1)   # e.g. 8.2.1

[[ -d $IMAGES_DIR ]] || { echo "Test images not found: $IMAGES_DIR (upload data/test_images first)"; exit 1; }

# 1. Clean build: start from an empty build directory
echo "==> Clean build in $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$SOURCE_DIR"
cmake --build . -- -j4

# 2. Inference with each engine on the test images
for PRECISION in fp16 fp32; do
    ENGINE="$MODELS_DIR/${MODEL}_${SIZE}_${PRECISION}_trt${TRT_VERSION}.engine"
    OUT_DIR="$RESULTS_DIR/jetson_${PRECISION}"

    if [[ ! -f $ENGINE ]]; then
        echo "==> Skipping $PRECISION: $ENGINE not found (run build_engines.sh)"
        continue
    fi

    echo
    echo "==> $PRECISION: $ENGINE"
    mkdir -p "$OUT_DIR"
    ./edge_ai_inference --engine "$ENGINE" --images "$IMAGES_DIR" --output "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/run.log"
done

echo
echo "==> Done. Results in $RESULTS_DIR/jetson_fp16 and $RESULTS_DIR/jetson_fp32"
