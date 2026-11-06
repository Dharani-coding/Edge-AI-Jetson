#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace edge_ai {

Letterbox::Letterbox(int input_width, int input_height)
    : input_width_(input_width),
      input_height_(input_height),
      canvas_(input_height, input_width, CV_8UC3, cv::Scalar(114, 114, 114))
{
    if (input_width <= 0 || input_height <= 0)
        throw std::invalid_argument("Letterbox: invalid input size");
}

void Letterbox::updateGeometry(const cv::Size& frame_size)
{
    const float scale = std::min(static_cast<float>(input_width_) / frame_size.width,
                                 static_cast<float>(input_height_) / frame_size.height);

    const int resized_w = std::min(input_width_, static_cast<int>(std::round(frame_size.width * scale)));
    const int resized_h = std::min(input_height_, static_cast<int>(std::round(frame_size.height * scale)));

    info_.scale = scale;
    info_.pad_x = (input_width_ - resized_w) / 2;
    info_.pad_y = (input_height_ - resized_h) / 2;
    content_ = cv::Rect(info_.pad_x, info_.pad_y, resized_w, resized_h);

    // Padding only needs repainting when the geometry changes.
    canvas_.setTo(cv::Scalar(114, 114, 114));
    frame_size_ = frame_size;
}

LetterboxInfo Letterbox::apply(const cv::Mat& bgr, float* dst)
{
    if (bgr.empty() || bgr.type() != CV_8UC3)
        throw std::invalid_argument("Letterbox: expected a non-empty BGR CV_8UC3 frame");

    if (bgr.size() != frame_size_)
        updateGeometry(bgr.size());

    // Resize straight into the canvas region (no extra allocation or copy).
    cv::Mat content = canvas_(content_);
    cv::resize(bgr, content, content_.size(), 0, 0, cv::INTER_LINEAR);

    canvas_.convertTo(canvas_f32_, CV_32FC3, 1.0 / 255.0);

    // Split into planes that alias the engine input. The canvas channels are
    // B, G, R; writing them to planes 2, 1, 0 gives planar RGB, so the
    // BGR->RGB swap costs nothing.
    const size_t plane = static_cast<size_t>(input_width_) * input_height_;
    float* const targets[3] = {dst + 2 * plane, dst + plane, dst};   // B, G, R

    std::vector<cv::Mat> planes;
    planes.reserve(3);
    for (float* target : targets)
        planes.emplace_back(input_height_, input_width_, CV_32F, target);

    cv::split(canvas_f32_, planes);

    // cv::split keeps the existing buffers when size/type match, but stay safe
    // if an OpenCV build ever reallocates.
    for (int c = 0; c < 3; ++c)
    {
        if (planes[c].ptr<float>() != targets[c])
            std::memcpy(targets[c], planes[c].ptr<float>(), plane * sizeof(float));
    }

    return info_;
}

}  // namespace edge_ai
