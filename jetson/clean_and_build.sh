#!/usr/bin/env bash
# Clean build of the C++ app: delete the build directory, configure and build from scratch.
#
#   ./clean_and_build.sh [build-dir]       # default: /mnt/usb-drive/build/edge-ai-inference
#
# Used by the other scripts in this folder; can also be run on its own.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)       # .../Edge-AI-Jetson/jetson
SOURCE_DIR="$SCRIPT_DIR/cpp"
BUILD_DIR="${1:-/mnt/usb-drive/build/edge-ai-inference}"

echo "==> Clean build in $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$SOURCE_DIR"
cmake --build . -- -j4
echo "==> Built $BUILD_DIR/edge_ai_inference"
