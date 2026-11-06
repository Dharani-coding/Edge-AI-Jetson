#include "trt_engine.hpp"

#include <NvInferPlugin.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace edge_ai {

void TrtLogger::log(Severity severity, const char* message) noexcept
{
    if (severity <= Severity::kWARNING)
        std::cerr << "[TensorRT] " << message << std::endl;
}

namespace {

std::vector<char> readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open TensorRT engine: " + path);

    const std::streamsize size = file.tellg();
    if (size <= 0)
        throw std::runtime_error("TensorRT engine file is empty: " + path);

    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!file.read(data.data(), size))
        throw std::runtime_error("Failed to read TensorRT engine: " + path);

    return data;
}

std::string toString(const nvinfer1::Dims& dims)
{
    std::ostringstream out;
    out << "[";
    for (int d = 0; d < dims.nbDims; ++d)
        out << dims.d[d] << (d + 1 < dims.nbDims ? ", " : "");
    out << "]";
    return out.str();
}

size_t volume(const nvinfer1::Dims& dims)
{
    size_t elements = 1;
    for (int d = 0; d < dims.nbDims; ++d)
    {
        if (dims.d[d] <= 0)
            throw std::runtime_error("Dynamic dimensions are not supported: " + toString(dims));
        elements *= static_cast<size_t>(dims.d[d]);
    }
    return elements;
}

}  // namespace

TrtEngine::TrtEngine(const std::string& engine_path)
{
    const std::vector<char> blob = readFile(engine_path);
    std::cout << "Engine: " << engine_path << " ("
              << blob.size() / (1024.0 * 1024.0) << " MB)" << std::endl;

    initLibNvInferPlugins(&logger_, "");

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
        throw std::runtime_error("TensorRT runtime creation failed");

    // Two-argument overload: the IPluginFactory variant is deprecated.
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_)
        throw std::runtime_error(
            "Engine deserialization failed. Engines must be built with the same "
            "TensorRT version on the same device type.");

    context_.reset(engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("Execution context creation failed");

    const int num_bindings = engine_->getNbBindings();
    device_buffers_.reserve(num_bindings);
    bindings_.resize(num_bindings, nullptr);

    std::cout << "\nTensorRT bindings:\n";

    for (int i = 0; i < num_bindings; ++i)
    {
        const nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        const bool is_input = engine_->bindingIsInput(i);

        std::cout << "  " << i << ": " << engine_->getBindingName(i) << " "
                  << toString(dims) << (is_input ? "  (input)" : "") << "\n";

        // FP16 engines still use FP32 I/O unless built with explicit I/O formats.
        if (engine_->getBindingDataType(i) != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("Only FP32 engine I/O is supported (binding "
                                     + std::to_string(i) + ")");

        if (is_input)
            input_binding_ = i;
        else if (dims.nbDims == 3)
            detection_binding_ = i;
        else if (dims.nbDims == 4)
            prototype_binding_ = i;

        device_buffers_.emplace_back(volume(dims) * sizeof(float));
        bindings_[i] = device_buffers_.back().get();
    }

    if (input_binding_ < 0 || detection_binding_ < 0 || prototype_binding_ < 0)
        throw std::runtime_error(
            "Unexpected engine bindings. Expected input [1,3,H,W], "
            "detections [1,C,N] and prototypes [1,K,H,W] (YOLOv8-seg).");

    const nvinfer1::Dims in = engine_->getBindingDimensions(input_binding_);
    const nvinfer1::Dims det = engine_->getBindingDimensions(detection_binding_);
    const nvinfer1::Dims proto = engine_->getBindingDimensions(prototype_binding_);

    if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3)
        throw std::runtime_error("Unexpected input dimensions " + toString(in));
    if (det.d[0] != 1 || proto.d[0] != 1)
        throw std::runtime_error("Only batch size 1 is supported");

    info_.input_height = in.d[2];
    info_.input_width = in.d[3];
    info_.num_anchors = det.d[2];
    info_.mask_channels = proto.d[1];
    info_.proto_height = proto.d[2];
    info_.proto_width = proto.d[3];
    info_.num_classes = det.d[1] - 4 - info_.mask_channels;

    if (info_.num_classes <= 0)
        throw std::runtime_error("Detection output " + toString(det)
                                 + " does not match " + std::to_string(info_.mask_channels)
                                 + " mask channels");

    host_input_.resize(volume(in));
    host_detections_.resize(volume(det));
    host_prototypes_.resize(volume(proto));

    std::cout << "\nInput " << info_.input_width << "x" << info_.input_height
              << " | anchors " << info_.num_anchors
              << " | classes " << info_.num_classes
              << " | prototypes " << info_.mask_channels << "x"
              << info_.proto_width << "x" << info_.proto_height << std::endl;
}

void TrtEngine::infer()
{
    cudaStream_t stream = stream_.get();

    checkCuda(cudaMemcpyAsync(bindings_[input_binding_], host_input_.data(),
                              host_input_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "Input H2D copy");

    if (!context_->enqueueV2(bindings_.data(), stream, nullptr))
        throw std::runtime_error("TensorRT inference failed");

    checkCuda(cudaMemcpyAsync(host_detections_.data(), bindings_[detection_binding_],
                              host_detections_.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream),
              "Detection D2H copy");

    checkCuda(cudaMemcpyAsync(host_prototypes_.data(), bindings_[prototype_binding_],
                              host_prototypes_.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream),
              "Prototype D2H copy");

    checkCuda(cudaStreamSynchronize(stream), "CUDA stream synchronize");
}

}  // namespace edge_ai
