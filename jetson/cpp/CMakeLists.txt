cmake_minimum_required(VERSION 3.10)

project(edge_ai_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu
)

find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu
)

# CUDA Runtime
find_library(CUDART_LIBRARY cudart
    PATHS /usr/local/cuda-10.2/targets/aarch64-linux/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR
   NOT TENSORRT_LIBRARY OR
   NOT TENSORRT_PLUGIN_LIBRARY OR
   NOT CUDART_LIBRARY)
    message(FATAL_ERROR "TensorRT or CUDA Runtime dependencies not found")
endif()

add_executable(edge_ai_inference
    src/main.cpp
)

target_include_directories(edge_ai_inference PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
     /usr/local/cuda-10.2/targets/aarch64-linux/include
)

target_link_libraries(edge_ai_inference PRIVATE
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${CUDART_LIBRARY}
    ${OpenCV_LIBS}
)