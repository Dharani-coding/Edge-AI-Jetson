#include "visualizer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace edge_ai {

namespace {

constexpr std::array<const char*, 80> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

constexpr double kMaskAlpha = 0.5;
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;

// Well-separated, bright colours: hue steps by the golden ratio.
std::vector<cv::Scalar> makePalette(int count)
{
    std::vector<cv::Scalar> palette;
    palette.reserve(count);

    for (int i = 0; i < count; ++i)
    {
        const double hue = std::fmod(i * 0.618033988749895, 1.0) * 180.0;
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
        palette.emplace_back(c[0], c[1], c[2]);
    }
    return palette;
}

void drawLabel(cv::Mat& frame, const std::string& text, cv::Point origin,
               const cv::Scalar& background, double font_scale)
{
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, kFont, font_scale, 1, &baseline);

    // Keep the label inside the frame.
    origin.x = std::min(std::max(origin.x, 0), std::max(0, frame.cols - size.width - 4));
    origin.y = std::max(origin.y, size.height + 6);

    cv::rectangle(frame,
                  cv::Point(origin.x, origin.y - size.height - 6),
                  cv::Point(origin.x + size.width + 4, origin.y),
                  background, cv::FILLED);
    cv::putText(frame, text, cv::Point(origin.x + 2, origin.y - 4),
                kFont, font_scale, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

}  // namespace

const char* cocoClassName(int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(kCocoClasses.size()))
        return kCocoClasses[class_id];
    return "object";
}

Visualizer::Visualizer(int num_classes, bool draw_masks)
    : draw_masks_(draw_masks), palette_(makePalette(std::max(1, num_classes)))
{
}

const cv::Scalar& Visualizer::colour(int class_id) const
{
    const int n = static_cast<int>(palette_.size());
    return palette_[((class_id % n) + n) % n];
}

void Visualizer::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) const
{
    const double font_scale = std::max(0.4, frame.cols / 2400.0);
    const int thickness = std::max(1, frame.cols / 640);

    // Masks first so boxes and labels stay on top.
    if (draw_masks_)
    {
        for (const Detection& d : detections)
        {
            if (d.mask.empty())
                continue;

            cv::Mat roi = frame(d.box);
            const cv::Mat tinted = roi * (1.0 - kMaskAlpha) + colour(d.class_id) * kMaskAlpha;
            tinted.copyTo(roi, d.mask);
        }
    }

    char text[64];
    for (const Detection& d : detections)
    {
        const cv::Scalar& c = colour(d.class_id);
        cv::rectangle(frame, d.box, c, thickness);

        std::snprintf(text, sizeof(text), "%s %.2f", cocoClassName(d.class_id), d.confidence);
        drawLabel(frame, text, d.box.tl(), c, font_scale);
    }
}

void Visualizer::drawHud(cv::Mat& frame, const std::string& text) const
{
    const double font_scale = std::max(0.5, frame.cols / 1600.0);
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, kFont, font_scale, 2, &baseline);

    cv::rectangle(frame, cv::Rect(0, 0, size.width + 20, size.height + baseline + 16),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, text, cv::Point(10, size.height + 8),
                kFont, font_scale, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

}  // namespace edge_ai
