#!/usr/bin/env bash
# Nsight Systems profile of the C++ app (FP16, MAXN), looping the demo video,
# with the app's NVTX stage labels (frame > capture, pre, trt, post, draw, stream).
#
#   bash ../scripts/clean_and_build.sh      # once, so the app is built with NVTX
#   ./nsight_systems_profile.sh
#
# Output in results/profiling/:
#   nsys_fp16_maxn.nsys-rep             open in the Nsight Systems GUI (timeline)
#   nsys_profile_<date>_<time>.log      nsys stats tables only
#
# Why no CUDA trace (--trace=cuda):
#   On this Jetson Nano (Tegra X1, JetPack 4.6, nsys 2021.5.4) GPU kernels and
#   memcpys are NOT recorded: gpukernsum / gpumemtimesum report "does not contain
#   CUDA kernel data". CUDA tracing still hooks every CUDA call (~350 us per
#   kernel launch, ~185 launches per frame) and slowed the app from 13.7 to ~3.4 FPS,
#   so stage times were meaningless. Without it, NVTX ranges give real per-stage
#   latency (avg, median, min, max) with little overhead.
#   Inference details come from trtexec instead (profiling/trtexec_profile.sh).
#
# Also off to keep overhead low:
#   --sample=none        no CPU sampling (it inflated every CPU stage ~2-6x)
#   --cpuctxsw=none      no thread context-switch tracing
#   --backtrace=none     no call stacks on OS-runtime calls
# For a CPU hotspot run instead (which functions take time), use
#   --trace=nvtx --sample=cpu  (stage times are then inflated, shares still valid).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)       # .../Edge-AI-Jetson/jetson/profiling
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)       # .../Edge-AI-Jetson
RESULTS="$REPO_DIR/results/profiling"
REPORT="$RESULTS/nsys_fp16_maxn"
LOG="$RESULTS/nsys_profile_$(date +%Y%m%d_%H%M%S).log"
APP="/mnt/usb-drive/build/edge-ai-inference/edge_ai_inference"
ENGINE="/mnt/usb-drive/models/yolov8n-seg_320x416_fp16_trt8.2.1.engine"
VIDEO="$REPO_DIR/data/videos/demo_video.mp4"

[[ -x $APP ]] || { echo "App not built: $APP (run: bash ../scripts/clean_and_build.sh)"; exit 1; }
mkdir -p "$RESULTS"

echo "==> MAXN"
sudo nvpmodel -m 0
sudo jetson_clocks

echo
echo "==> nsys profile (NVTX + OS runtime, no CUDA)"
sudo nsys profile \
  --trace=nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --delay=10 \
  --force-overwrite=true \
  -o "$REPORT" \
  "$APP" \
    --engine "$ENGINE" \
    --video  "$VIDEO" \
    --loop-seconds 60 --stats 0

sudo chown "$USER:$USER" "$REPORT".*            # report was written by root

# nvtxppsum: one row per NVTX range (frame, capture, pre, trt, post, draw, stream)
#            with count, total, avg, median, min, max, stddev.
#
# TensorRT adds its own NVTX range for every layer ("TensorRT:/model.22/...",
# ~200 rows). They only measure the CPU time to enqueue each layer (~40 us), not
# GPU time, so the log keeps just TensorRT:ExecutionContext::enqueue. All layer
# ranges are still in the .nsys-rep for the GUI timeline.
echo
echo "==> nsys stats"
nsys stats --force-export=true \
  --report nvtxppsum,osrtsum \
  "$REPORT.nsys-rep" 2>&1 \
  | grep -v 'Exporting [0-9]* events' \
  | awk '!/TensorRT:/ || /ExecutionContext::enqueue/' \
  | tee "$LOG"                                  # stats tables only: no progress bar, no per-layer rows

echo
echo "Report: $REPORT.nsys-rep"
echo "Log:    $LOG"
