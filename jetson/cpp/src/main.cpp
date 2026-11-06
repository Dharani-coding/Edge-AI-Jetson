#include <iostream>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

int main()
{
    std::cout << "C++17: " << __cplusplus << std::endl;

    std::cout << "CUDA Runtime version: ";

    int cuda_version = 0;
    cudaRuntimeGetVersion(&cuda_version);

    std::cout << cuda_version << std::endl;

    std::cout << "TensorRT version: "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << std::endl;

    std::cout << "OpenCV version: "
              << CV_VERSION << std::endl;

    return 0;
}