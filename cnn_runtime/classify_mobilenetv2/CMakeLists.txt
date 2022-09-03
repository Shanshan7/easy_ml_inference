cmake_minimum_required(VERSION 3.5)

# set(TOOLS_CHAIN_DIR /usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4) # 交叉工具链所在目录

set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_C_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.0)

project(demo)

include_directories(${PROJECT_SOURCE_DIR})
# include_directories(${PROJECT_SOURCE_DIR}/onnx_runtime_cpp/include)

# onnxruntime
include_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/onnxruntime_gpu/include)
link_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/onnxruntime_gpu/lib)

#collect image acquisition files
file(GLOB SRCS
    "${PROJECT_SOURCE_DIR}/onnx_runtime_cpp/src/*.cpp"
	)

# opencv
include_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/opencv/include)
link_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/opencv/lib)

# set(CUDA_BIN_PATH /usr/local/cuda-11.0/)
# add_definitions(-DENABLE_GPU=1)

# if(CUDA_FOUND)
#   message("use cuda version")
#   add_definitions(-DENABLE_GPU=1)
# else()
#   add_definitions(-DENABLE_GPU=0)
# endif()

add_executable(test "")

target_sources(test
        PRIVATE
        "${PROJECT_SOURCE_DIR}/main.cpp"
        )

# link libraries
target_link_libraries(test 
                      onnxruntime
                      opencv_core
                      opencv_imgcodecs
                      opencv_imgproc)