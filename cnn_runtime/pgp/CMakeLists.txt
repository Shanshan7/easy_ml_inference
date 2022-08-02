cmake_minimum_required(VERSION 3.5)

# set(TOOLS_CHAIN_DIR /usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4) # 交叉工具链所在目录

set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_C_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(demo)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/onnx_runtime_cpp/include)

# onnxruntime
include_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/onnxruntime/include)
link_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/onnxruntime/lib)

#collect image acquisition files
file(GLOB SRCS
    "${PROJECT_SOURCE_DIR}/onnx_runtime_cpp/src/*.cpp"
	)

# find_package(CUDA QUIET)
find_package(OpenCV REQUIRED)

add_executable(test "")

target_sources(test
        PRIVATE
        "${PROJECT_SOURCE_DIR}/test.cpp"
        ${SRCS}
        )
# link libraries
target_link_libraries(test 
                      onnxruntime
                      ${OpenCV_LIBS} )