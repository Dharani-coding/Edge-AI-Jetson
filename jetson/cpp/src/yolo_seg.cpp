#include "yolo_seg.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace edge_ai {

namespace {

float iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float intersection = (a & b).area();
    const float union_area = a.area() + b.area() - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

}  // namespace

YoloSegDecoder::YoloSegDecoder(const SegModelInfo& info, const DetectorConfig& config)
    : info_(info), config_(config)
{
    best_score_.resize(info_.num_anchors);
    best_class_.resize(info_.num_anchors);
}

const std::vector<Detection>& YoloSegDecoder::process(const float* detections,
                                               const float* prototypes,
                                               const LetterboxInfo& letterbox,
                                               const cv::Size& frame_size)
{
    decode(detections);
    nonMaxSuppression();
    mapToFrame(letterbox, frame_size);

    if (config_.draw_masks && !results_.empty())
        buildMasks(detections, prototypes, letterbox);

    return results_;
}

void YoloSegDecoder::decode(const float* out)
{
    const int n = info_.num_anchors;
    const float in_w = static_cast<float>(info_.input_width);
    const float in_h = static_cast<float>(info_.input_height);

    // The output is channel-major ([C, N]), so walk one class row at a time
    // to read memory sequentially instead of striding per anchor.
    std::fill(best_score_.begin(), best_score_.end(), 0.0f);
    std::fill(best_class_.begin(), best_class_.end(), 0);

    for (int c = 0; c < info_.num_classes; ++c)
    {
        const float* row = out + static_cast<size_t>(4 + c) * n;
        for (int i = 0; i < n; ++i)
        {
            if (row[i] > best_score_[i])
            {
                best_score_[i] = row[i];
                best_class_[i] = c;
            }
        }
    }

    candidates_.clear();

    for (int i = 0; i < n; ++i)
    {
        if (best_score_[i] < config_.conf_threshold)
            continue;

        const float cx = out[i];
        const float cy = out[n + i];
        const float w = out[2 * n + i];
        const float h = out[3 * n + i];

        const float x1 = std::max(0.0f, cx - 0.5f * w);
        const float y1 = std::max(0.0f, cy - 0.5f * h);
        const float x2 = std::min(in_w, cx + 0.5f * w);
        const float y2 = std::min(in_h, cy + 0.5f * h);

        if (x2 <= x1 || y2 <= y1)
            continue;

        Detection d;
        d.input_box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        d.confidence = best_score_[i];
        d.class_id = best_class_[i];
        d.anchor = i;
        candidates_.push_back(std::move(d));
    }
}

void YoloSegDecoder::nonMaxSuppression()
{
    std::sort(candidates_.begin(), candidates_.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    results_.clear();

    for (Detection& candidate : candidates_)
    {
        bool suppressed = false;
        for (const Detection& kept : results_)
        {
            if (candidate.class_id == kept.class_id &&
                iou(candidate.input_box, kept.input_box) > config_.iou_threshold)
            {
                suppressed = true;
                break;
            }
        }

        if (!suppressed)
        {
            results_.push_back(std::move(candidate));
            if (static_cast<int>(results_.size()) >= config_.max_detections)
                break;
        }
    }
}

void YoloSegDecoder::mapToFrame(const LetterboxInfo& lb, const cv::Size& frame_size)
{
    auto toFrameX = [&](float x) { return (x - lb.pad_x) / lb.scale; };
    auto toFrameY = [&](float y) { return (y - lb.pad_y) / lb.scale; };

    // Compact in place, dropping boxes that lie entirely in the letterbox padding.
    size_t kept = 0;
    for (size_t i = 0; i < results_.size(); ++i)
    {
        Detection& d = results_[i];
        const cv::Rect2f& b = d.input_box;

        const int x1 = std::max(0, static_cast<int>(std::floor(toFrameX(b.x))));
        const int y1 = std::max(0, static_cast<int>(std::floor(toFrameY(b.y))));
        const int x2 = std::min(frame_size.width, static_cast<int>(std::ceil(toFrameX(b.x + b.width))));
        const int y2 = std::min(frame_size.height, static_cast<int>(std::ceil(toFrameY(b.y + b.height))));

        if (x2 <= x1 || y2 <= y1)
            continue;

        d.box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        if (kept != i)
            results_[kept] = std::move(d);
        ++kept;
    }

    results_.resize(kept);
}

void YoloSegDecoder::buildMasks(const float* out, const float* prototypes, const LetterboxInfo& lb)
{
    const int n = info_.num_anchors;
    const int k = info_.mask_channels;
    const int count = static_cast<int>(results_.size());
    const int coeff_row = 4 + info_.num_classes;

    // All masks in one matrix multiply: [count x K] * [K x (Ph*Pw)].
    coefficients_.create(count, k, CV_32F);
    for (int d = 0; d < count; ++d)
    {
        float* row = coefficients_.ptr<float>(d);
        const int anchor = results_[d].anchor;
        for (int c = 0; c < k; ++c)
            row[c] = out[static_cast<size_t>(coeff_row + c) * n + anchor];
    }

    const cv::Mat protos(k, info_.proto_width * info_.proto_height, CV_32F,
                         const_cast<float*>(prototypes));
    cv::gemm(coefficients_, protos, 1.0, cv::noArray(), 0.0, mask_logits_);

    // Frame pixel (u, v) inside the box -> prototype pixel, exactly:
    //   proto = ((frame + 0.5) * scale + pad) * (proto / input) - 0.5
    const float sx = static_cast<float>(info_.proto_width) / info_.input_width;
    const float sy = static_cast<float>(info_.proto_height) / info_.input_height;

    for (int d = 0; d < count; ++d)
    {
        Detection& det = results_[d];
        const cv::Mat logits = mask_logits_.row(d).reshape(1, info_.proto_height);

        const cv::Matx23f frame_to_proto(
            lb.scale * sx, 0.0f, ((det.box.x + 0.5f) * lb.scale + lb.pad_x) * sx - 0.5f,
            0.0f, lb.scale * sy, ((det.box.y + 0.5f) * lb.scale + lb.pad_y) * sy - 0.5f);

        // Only the box region is upsampled, never the whole frame.
        cv::warpAffine(logits, warped_, frame_to_proto, det.box.size(),
                       cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);

        // sigmoid(x) > 0.5  <=>  x > 0, so no exp() is needed.
        cv::compare(warped_, 0.0, det.mask, cv::CMP_GT);
    }
}

}  // namespace edge_ai
