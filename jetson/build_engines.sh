#!/usr/bin/env bash
# Build FP32 and FP16 TensorRT engines from an ONNX model with trtexec.
# Each engine gets a descriptive name, and its trtexec log is saved next to it.
#
#   ./build_engines.sh [model.onnx] [HxW]
#   ./build_engines.sh /mnt/usb-drive/models/yolov8n-seg.onnx 320x416
#
# Output (next to the ONNX file):
#   yolov8n-seg_320x416_fp32_trt8.2.1.engine  + .log
#   yolov8n-seg_320x416_fp16_trt8.2.1.engine  + .log

set -euo pipefail

ONNX="${1:-/mnt/usb-drive/models/yolov8n-seg.onnx}"
SIZE="${2:-320x416}"     # input height x width the ONNX was exported with (used in the name)
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

[[ -f $ONNX ]]    || { echo "ONNX not found: $ONNX"; exit 1; }
[[ -x $TRTEXEC ]] || { echo "trtexec not found: $TRTEXEC"; exit 1; }

TRT_VERSION=$(dpkg-query -W -f='${Version}' libnvinfer8 | cut -d- -f1)   # e.g. 8.2.1
MODEL=$(basename "$ONNX" .onnx)
OUT_DIR=$(dirname "$ONNX")

for PRECISION in fp32 fp16; do
    NAME="${MODEL}_${SIZE}_${PRECISION}_trt${TRT_VERSION}"
    ENGINE="$OUT_DIR/$NAME.engine"
    LOG="$OUT_DIR/$NAME.log"

    FLAGS=()
    [[ $PRECISION == fp16 ]] && FLAGS+=(--fp16)

    echo "==> Building $ENGINE"
    "$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --workspace=1024 "${FLAGS[@]}" 2>&1 | tee "$LOG"
    echo "==> Saved $ENGINE"
    echo "==> Log   $LOG"
done
