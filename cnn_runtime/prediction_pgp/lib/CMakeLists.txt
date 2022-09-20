project(perception_prediction_lib)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

include_directories(${perception_prediction_BINARY_DIR})
include_directories(${perception_prediction_SOURCE_DIR})

include_directories("postprocess")
include_directories("preprocess")
include_directories("predictor/kalman_predictor")
include_directories("predictor/pgp_predictor")

# add_subdirectory(obstacle)
# add_subdirectory(tracker)

file(GLOB _SRCS
    "*.cpp"
    "predictor/pgp_predictor/*.cpp"
    "postprocess/*.cpp"
    "preprocess/*.cpp"
	)

add_library(${PROJECT_NAME} SHARED ${_SRCS})
target_link_libraries(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# install(FILES "${ai_algorithm_BINARY_DIR}/perception_camera_base.so" DESTINATION lib ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

