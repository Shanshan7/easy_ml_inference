cmake_minimum_required(VERSION 2.8)

# set(TOOLS_CHAIN_DIR /usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4) # 交叉工具链所在目录

# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_C_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER ${TOOLS_CHAIN_DIR}/bin/aarch64-linux-gnu-g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

project(prediction_pgp)

add_definitions(-DENABLE_DEBUG)
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

include_directories(${PROJECT_SOURCE_DIR}/prediction)
include_directories(${PROJECT_SOURCE_DIR}/lib/postprocess)
include_directories(${PROJECT_SOURCE_DIR}/lib/predictor/kalman_predictor)
include_directories(${PROJECT_SOURCE_DIR}/lib/predictor/pgp_predictor)
include_directories(${PROJECT_SOURCE_DIR}/lib/preprocess)

# onnxruntime
include_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/onnxruntime_gpu/include)
link_directories(${PROJECT_SOURCE_DIR}/../../3rd_party/onnxruntime_gpu/lib)

#collect image acquisition files
file(GLOB SRCS
    "${PROJECT_SOURCE_DIR}/prediction/*.cpp"
    "${PROJECT_SOURCE_DIR}/lib/postprocess/*.cpp"
    "${PROJECT_SOURCE_DIR}/lib/predictor/pgp_predictor/*.cpp"
    "${PROJECT_SOURCE_DIR}/lib/preprocess/JoysonPredictionPre.cpp"
	)

add_executable(${PROJECT_NAME} "")

target_sources(${PROJECT_NAME}
        PRIVATE
        "${PROJECT_SOURCE_DIR}/test/testJoysonPrecition.cpp"
        ${SRCS}
        )

# link libraries
target_link_libraries(${PROJECT_NAME}
  # ${NVINFER}
  # ${NVINFER_PLUGIN}
  onnxruntime
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES} 
  ${PCL_LIBRARIES}
)