cmake_minimum_required(VERSION 2.8)

# set(TOOLS_CHAIN_DIR /usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4) # 交叉工具链所在目录

# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_C_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-g++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

project(det3d_tracking)

add_definitions(-DUSE_SMOKE)
# add_definitions(-DUSE_DET_FILES)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_library(NVINFER NAMES nvinfer)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem thread system program_options)

include_directories(
  include
  ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS}
)
message(${PCL_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/smoke)
include_directories(${PROJECT_SOURCE_DIR}/multi_tracking)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../../3rd_party/tensorrt/include
)
link_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/tensorrt/lib)
# link_directories(/opt/TensorRT-8.0.3.4/lib)

#collect image acquisition files
file(GLOB SRCS
    "${PROJECT_SOURCE_DIR}/*cpp"
    "${PROJECT_SOURCE_DIR}/smoke/*.cc"
    "${PROJECT_SOURCE_DIR}/multi_tracking/*.cpp"
	)

add_executable(${PROJECT_NAME} "")

target_sources(${PROJECT_NAME}
        PRIVATE
        "${PROJECT_SOURCE_DIR}/test_det3d_tracking.cpp"
        ${SRCS}
        )

# link libraries
target_link_libraries(${PROJECT_NAME}
  # ${NVINFER}
  # ${NVINFER_PLUGIN}
  libnvinfer.so
  libnvinfer_plugin.so
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES} 
  ${PCL_LIBRARIES}
)
