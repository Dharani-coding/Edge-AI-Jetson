#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace edge_ai {

inline void checkCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Owns a cudaMalloc allocation (freed automatically, also on exceptions).
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        checkCuda(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_)
    {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    void* get() const { return ptr_; }
    size_t bytes() const { return bytes_; }

private:
    void reset() noexcept
    {
        if (ptr_)
            cudaFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

// Owns a cudaStream_t.
class CudaStream
{
public:
    CudaStream() { checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~CudaStream() { if (stream_) cudaStreamDestroy(stream_); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

}  // namespace edge_ai
