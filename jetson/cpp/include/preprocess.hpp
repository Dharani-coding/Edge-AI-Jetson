#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

namespace edge_ai {

// Letterbox resize + BGR->RGB + /255 + HWC->CHW, writing straight into the
// engine's input buffer. Working buffers are reused across frames.
class Letterbox
{
public:
    Letterbox(int input_width, int input_height);

    // `dst` must hold 3 * input_width * input_height floats.
    LetterboxInfo apply(const cv::Mat& bgr, float* dst);

private:
    void updateGeometry(const cv::Size& frame_size);

    int input_width_;
    int input_height_;

    cv::Size frame_size_;   // frame size the geometry below was computed for
    LetterboxInfo info_;
    cv::Rect content_;      // area of the canvas covered by the resized frame

    cv::Mat canvas_;        // CV_8UC3, padded with grey 114
    cv::Mat canvas_f32_;    // CV_32FC3, scaled to [0, 1]
};

}  // namespace edge_ai
