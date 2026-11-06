#pragma once

#include <opencv2/core.hpp>

namespace edge_ai {

// Tensor shapes read from the TensorRT engine bindings.
struct SegModelInfo
{
    int input_width = 0;    // e.g. 416
    int input_height = 0;   // e.g. 320
    int num_anchors = 0;    // e.g. 2730
    int num_classes = 0;    // 80 for COCO
    int mask_channels = 0;  // 32 prototype channels
    int proto_width = 0;    // e.g. 104
    int proto_height = 0;   // e.g. 80
};

// Mapping between the camera frame and the letterboxed model input:
//   input_xy = frame_xy * scale + pad
struct LetterboxInfo
{
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct Detection
{
    cv::Rect2f input_box;   // model-input (letterboxed) coordinates
    cv::Rect box;           // camera-frame coordinates, clipped to the frame
    float confidence = 0.0f;
    int class_id = -1;
    int anchor = -1;        // column in the raw output, used to fetch mask coefficients
    cv::Mat mask;           // CV_8U, box.size(), 255 = object; empty if masks disabled
};

}  // namespace edge_ai
