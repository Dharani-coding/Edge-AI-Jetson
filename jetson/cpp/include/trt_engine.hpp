#pragma once

#include "cuda_utils.hpp"
#include "types.hpp"

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

namespace edge_ai {

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* message) noexcept override;
};

// Loads a YOLOv8-seg TensorRT engine and runs synchronous inference.
//
// Expected bindings (FP32 I/O, static shapes):
//   input       [1, 3, H, W]
//   detections  [1, 4 + classes + mask_channels, anchors]
//   prototypes  [1, mask_channels, proto_h, proto_w]
class TrtEngine
{
public:
    explicit TrtEngine(const std::string& engine_path);

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Host buffer to fill with the CHW float input before calling infer().
    float* input() { return host_input_.data(); }

    // Copies input to the GPU, runs the network, copies outputs back. Throws on failure.
    void infer();

    const float* detections() const { return host_detections_.data(); }
    const float* prototypes() const { return host_prototypes_.data(); }

    const SegModelInfo& info() const { return info_; }

private:
    // Declaration order matters: members are destroyed in reverse order,
    // so the context goes before the engine, and the engine before the runtime.
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    CudaStream stream_;
    std::vector<DeviceBuffer> device_buffers_;
    std::vector<void*> bindings_;

    int input_binding_ = -1;
    int detection_binding_ = -1;
    int prototype_binding_ = -1;

    std::vector<float> host_input_;
    std::vector<float> host_detections_;
    std::vector<float> host_prototypes_;

    SegModelInfo info_;
};

}  // namespace edge_ai
