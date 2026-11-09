#pragma once

#include "config.hpp"

namespace edge_ai {

// Runs the detector on image files instead of the camera.
//
// For every input image <name>.<ext>, writes to config.output_dir:
//   <name>.<ext>  annotated image (masks, boxes, labels)
//   <name>.txt    one detection per line, in original-image pixels:
//                 class_id confidence x1 y1 x2 y2 class_name
//
// The text format matches windows/scripts/compare_with_jetson.py, so the
// Jetson (TensorRT) and Windows (ONNX) results can be compared directly.
int runOnImages(const AppConfig& config);

}  // namespace edge_ai
