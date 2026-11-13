#!/usr/bin/env bash
# GPU kernel profile of the C++ app with nvprof (FP16, MAXN), looping the demo video.
# nvprof is used because nsys 2021.5 records no GPU kernels on the Nano (Maxwell, sm_53).
#
#   ./nvprof_profile.sh
#
# Output in results/profiling/:
#   nvprof_fp16_maxn.log   GPU summary only: one row per kernel / memcpy
#                          (time %, total, calls, avg, min, max)
#
# Left out of the log:
#   - the app's own output (shown on screen only)
#   - nvprof's per-NVTX-range sections (~220 TensorRT layer ranges); per-layer
#     GPU time is in trtexec_profile.sh, and the kernel totals are in the summary
#   - long C++ template signatures in kernel names (cut to NAME_WIDTH characters)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)       # .../Edge-AI-Jetson/jetson/profiling
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)       # .../Edge-AI-Jetson
RESULTS="$REPO_DIR/results/profiling"
LOG="$RESULTS/nvprof_fp16_maxn.log"
RAW="$(mktemp)"                                 # full nvprof output, deleted at the end
NVPROF="/usr/local/cuda/bin/nvprof"
APP="/mnt/usb-drive/build/edge-ai-inference/edge_ai_inference"
ENGINE="/mnt/usb-drive/models/yolov8n-seg_320x416_fp16_trt8.2.1.engine"
VIDEO="$REPO_DIR/data/videos/demo_video.mp4"
NAME_WIDTH=160                                  # max characters per log line

[[ -x $APP ]] || { echo "App not built: $APP (run: bash ../scripts/clean_and_build.sh)"; exit 1; }
mkdir -p "$RESULTS"
trap 'sudo rm -f "$RAW"' EXIT

echo "==> MAXN"
sudo nvpmodel -m 0
sudo jetson_clocks

echo
echo "==> nvprof (app output below is not logged)"
# --unified-memory-profiling off: required on Jetson, nvprof fails without it.
sudo "$NVPROF" \
  --unified-memory-profiling off \
  --print-gpu-summary \
  --log-file "$RAW" \
  "$APP" \
    --engine "$ENGINE" \
    --video  "$VIDEO" \
    --loop-seconds 30 --stats 0

# Keep the GPU summary, stop at the per-NVTX-range sections, shorten long kernel names.
sudo rm -f "$LOG"                               # an older log may be owned by root
sudo cat "$RAW" \
  | awk -v w="$NAME_WIDTH" '/NVTX result:/ { exit }
                            { print (length($0) > w ? substr($0, 1, w) "..." : $0) }' \
  > "$LOG"

echo
cat "$LOG"
echo
echo "Log: $LOG"
