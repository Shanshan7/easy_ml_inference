#1.cmake verson
cmake_minimum_required(VERSION 3.5.0)

#2.project name
project(yolov5rt)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)

#add_definitions(-DFEATURE_MATCH_EN)

#include MOT headers
include_directories(${PROJECT_SOURCE_DIR}/)
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PROJECT_SOURCE_DIR}/onnx)
include_directories(${PROJECT_SOURCE_DIR}/../3rd_party/json/include)

#collect image acquisition files
# find_package(OpenCV REQUIRED)
link_directories(${PROJECT_SOURCE_DIR}/../3rd_party/opencv/lib)

file(GLOB SRCS
    "${PROJECT_SOURCE_DIR}/*.cpp"
	)

file(GLOB ONNXSRCS
    "${PROJECT_SOURCE_DIR}/onnx/*.cpp"
)

aux_source_directory(${PROJECT_SOURCE_DIR}/../3rd_party/json/src SRC_DIR)
add_executable(test_traffic_lights_classifier ${SRC_DIR})

target_sources(test_traffic_lights_classifier
        PRIVATE
        "${PROJECT_SOURCE_DIR}/test_traffic_lights_classifier.cpp"
        ${SRCS}
        ${ONNXSRCS}
        )

# link libraries
target_link_libraries(test_traffic_lights_classifier
                      opencv_core
                      opencv_imgcodecs
                      opencv_imgproc
                      opencv_videoio
		                  opencv_dnn)
