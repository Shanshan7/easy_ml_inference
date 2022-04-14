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

#collect image acquisition files
find_package(OpenCV REQUIRED)

file(GLOB SRCS
    "${PROJECT_SOURCE_DIR}/*.cpp"
	)

add_executable(test_traffic_lights_classifier "")

target_sources(test_traffic_lights_classifier
        PRIVATE
        "${PROJECT_SOURCE_DIR}/test_traffic_lights_classifier.cpp"
        ${SRCS}
        )

# link libraries
target_link_libraries(test_traffic_lights_classifier ${OpenCV_LIBS} -lglog)
