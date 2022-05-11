cmake_minimum_required(VERSION 3.10)
project(yolov5)

set(CMAKE_CXX_STANDARD 14)

add_definitions(-DIS_SHOW -DIS_SAVE_TXT)

include_directories(${PROJECT_SOURCE_DIR}/../../utility)

add_executable(yolov5 main_yolo.cpp)
find_package(OpenCV REQUIRED)

#collect source files
file(GLOB UTILS_SRCS
    "${PROJECT_SOURCE_DIR}/../../utility/*.cpp"
	)

target_link_libraries(yolov5 ${OpenCV_LIBS} ${UTILS_SRCS})
