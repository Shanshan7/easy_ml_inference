CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
project(one_class_test)
find_package(OpenCV REQUIRED)

add_definitions(-Ofast)

set(PROJECTDIR ../../cnn_runtime/)

include_directories(${PROJECTDIR}/one_class/)

#collect source files
file(GLOB_RECURSE SRC_LIST
     ${PROJECTDIR}/one_class/patchcore_postprocess.cpp
)

add_executable(one_class_test ./one_class.cpp ${SRC_LIST})
target_link_libraries(one_class_test ${OpenCV_LIBS} -lpthread -lm)