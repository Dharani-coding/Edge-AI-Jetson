cmake_minimum_required(VERSION 3.10)

project(edge_ai_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build optimised code by default. The old build had no build type, i.e. -O0.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ---------------------------------------------------------------- Dependencies

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

set(CUDA_ROOT "/usr/local/cuda-10.2" CACHE PATH "CUDA toolkit root")

find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    PATHS ${CUDA_ROOT}/targets/aarch64-linux/include ${CUDA_ROOT}/include
)
find_library(CUDART_LIBRARY cudart
    PATHS ${CUDA_ROOT}/targets/aarch64-linux/lib ${CUDA_ROOT}/lib64
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu
)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY
   OR NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIBRARY)
    message(FATAL_ERROR "TensorRT or CUDA runtime not found (CUDA_ROOT=${CUDA_ROOT})")
endif()

# ---------------------------------------------------------------- Core library
# Everything except main(), so unit tests (GTest) can link against it later.

add_library(edge_ai_core STATIC
    src/config.cpp
    src/trt_engine.cpp
    src/preprocess.cpp
    src/yolo_seg.cpp
    src/visualizer.cpp
    src/video_io.cpp
    src/profiler.cpp
)

target_include_directories(edge_ai_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# SYSTEM: keep warnings from vendor headers out of our build output.
target_include_directories(edge_ai_core SYSTEM PUBLIC
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(edge_ai_core PUBLIC
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${CUDART_LIBRARY}
    ${OpenCV_LIBS}
)

target_compile_options(edge_ai_core PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------- Executable

add_executable(edge_ai_inference src/main.cpp)
target_link_libraries(edge_ai_inference PRIVATE edge_ai_core)
target_compile_options(edge_ai_inference PRIVATE -Wall -Wextra)
