#!/usr/bin/env bash
# trtexec profile of the FP32 and FP16 engines in MAXN (10 W), then in 5 W.
# All output goes to one log: results/profiling/trtexec_profile_<date>_<time>.log
#
#   ./trtexec_profile.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)       # .../Edge-AI-Jetson/jetson/profiling
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)       # .../Edge-AI-Jetson
RESULTS="$REPO_DIR/results/profiling"
LOG="$RESULTS/trtexec_profile_$(date +%Y%m%d_%H%M%S).log"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
FP32_ENGINE="/mnt/usb-drive/models/yolov8n-seg_320x416_fp32_trt8.2.1.engine"
FP16_ENGINE="/mnt/usb-drive/models/yolov8n-seg_320x416_fp16_trt8.2.1.engine"

mkdir -p "$RESULTS"
exec > >(tee "$LOG") 2>&1                       # everything below goes to the screen and the log

profile() {   # label, engine
    echo
    echo "==================== $1 ===================="
    "$TRTEXEC" --loadEngine="$2" --warmUp=2000 --duration=60 --percentile=99 --useSpinWait
}

echo "==> MAXN (10 W)"
sudo nvpmodel -m 0
sudo jetson_clocks
sudo nvpmodel -q
profile "FP32 / MAXN" "$FP32_ENGINE"
profile "FP16 / MAXN" "$FP16_ENGINE"

echo
echo "==> 5 W"
sudo nvpmodel -m 1
sudo jetson_clocks
sudo nvpmodel -q
profile "FP32 / 5W" "$FP32_ENGINE"
profile "FP16 / 5W" "$FP16_ENGINE"

echo
echo "==> Back to MAXN"
sudo nvpmodel -m 0
sudo jetson_clocks
echo "Log: $LOG"
