#!/usr/bin/env bash
# Clean-build the C++ app, then annotate a video with the FP16 engine (--video mode).
#
#   ./run_video_demo.sh [video.mp4] [frames-to-save]
#   ./run_video_demo.sh                                   # data/videos/demo_video.mp4, 10 frames
#
# Output:
#   data/results/video/<name>_annotated.mp4
#   data/results/video/<name>_frames/frame_NNNNNN.jpg
#   data/results/video/run.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)       # .../Edge-AI-Jetson/jetson
REPO_DIR=$(dirname "$SCRIPT_DIR")               # .../Edge-AI-Jetson
BUILD_DIR="/mnt/usb-drive/build/edge-ai-inference"
ENGINE="/mnt/usb-drive/models/yolov8n-seg_320x416_fp16_trt8.2.1.engine"
VIDEO="${1:-$REPO_DIR/data/videos/demo_video.mp4}"
SAVE_FRAMES="${2:-10}"
OUT_DIR="$REPO_DIR/data/results/video"

[[ -f $VIDEO ]]  || { echo "Video not found: $VIDEO"; exit 1; }
[[ -f $ENGINE ]] || { echo "Engine not found: $ENGINE (run build_engines.sh)"; exit 1; }

# 1. Clean build
bash "$SCRIPT_DIR/clean_and_build.sh" "$BUILD_DIR"
cd "$BUILD_DIR"

# 2. Annotate the video
echo
echo "==> Annotating $VIDEO"
mkdir -p "$OUT_DIR"
./edge_ai_inference --engine "$ENGINE" \
    --video "$VIDEO" \
    --output "$OUT_DIR" \
    --save-frames "$SAVE_FRAMES" \
    2>&1 | tee "$OUT_DIR/run.log"

echo
echo "==> Done. Results in $OUT_DIR"
