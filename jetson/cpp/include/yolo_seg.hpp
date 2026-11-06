#pragma once

#include "config.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace edge_ai {

// YOLOv8-seg post-processing:
//   output0 [1, 4 + classes + K, anchors]  boxes (cx, cy, w, h), class scores, mask coefficients
//   output1 [1, K, proto_h, proto_w]       mask prototypes
class YoloSegDecoder
{
public:
    YoloSegDecoder(const SegModelInfo& info, const DetectorConfig& config);

    // Decode, NMS, map boxes to the frame and (optionally) build per-box masks.
    // The returned reference stays valid until the next call.
    const std::vector<Detection>& process(const float* detections,
                                   const float* prototypes,
                                   const LetterboxInfo& letterbox,
                                   const cv::Size& frame_size);

private:
    void decode(const float* detections);
    void nonMaxSuppression();
    void mapToFrame(const LetterboxInfo& letterbox, const cv::Size& frame_size);
    void buildMasks(const float* detections, const float* prototypes,
                    const LetterboxInfo& letterbox);

    SegModelInfo info_;
    DetectorConfig config_;

    // Reused across frames to avoid per-frame allocations.
    std::vector<float> best_score_;
    std::vector<int> best_class_;
    std::vector<Detection> candidates_;
    std::vector<Detection> results_;
    cv::Mat coefficients_;   // N x K
    cv::Mat mask_logits_;    // N x (proto_h * proto_w)
    cv::Mat warped_;         // box-sized logits
};

}  // namespace edge_ai
