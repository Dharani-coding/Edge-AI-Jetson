#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace edge_ai {

const char* cocoClassName(int class_id);

class Visualizer
{
public:
    Visualizer(int num_classes, bool draw_masks);

    // Mask overlays, boxes and labels.
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) const;

    // Status line in the top-left corner.
    void drawHud(cv::Mat& frame, const std::string& text) const;

private:
    const cv::Scalar& colour(int class_id) const;

    bool draw_masks_;
    std::vector<cv::Scalar> palette_;
};

}  // namespace edge_ai
