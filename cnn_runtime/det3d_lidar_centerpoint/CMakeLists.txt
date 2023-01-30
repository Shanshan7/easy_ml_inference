cmake_minimum_required(VERSION 3.18)
project(det3d_lidar_centerpoint)
# Find installed Open3D, which exports Open3D::Open3D
find_package(yaml-cpp REQUIRED)
find_package(CUDA REQUIRED)
set(Open3D_ROOT /docker_data/easy_ml_inference/3rd_party/open3d)
find_package(Open3D REQUIRED)

include_directories(${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")
add_compile_options(-W)
add_compile_options(-std=c++11)

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -ccbin ${CMAKE_CXX_COMPILER}
  -Xcompiler -DWIN_INTERFACE_CUSTOM
  -Xcompiler -I/usr/aarch64-linux-gnu/include/
  -Xlinker -lsocket
  -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
  -Xlinker -rpath=/usr/aarch64-linux-gnu/lib/
  -Xlinker -L/usr/lib/aarch64-linux-gnu/
  -Xlinker -L/usr/aarch64-linux-gnu/lib/
)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

# set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
# set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
set(TENSORRT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../../3rd_party/tensorrt/include)
set(TENSORRT_LIBRARY_DIRS ${PROJECT_SOURCE_DIR}/../../3rd_party/tensorrt/lib)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

link_directories(
	${TENSORRT_LIBRARY_DIRS}
  ${PROJECT_SOURCE_DIR}/../../3rd_party/open3d/lib
	/usr/lib/aarch64-linux-gnu
  /usr/aarch64-linux-gnu/lib/
)

file(GLOB_RECURSE SOURCE_FILES
  centerpoint/*.cu
  centerpoint/*.cc
)

cuda_add_executable(${PROJECT_NAME} test_det3d_lidar_centerpoint.cpp ${SOURCE_FILES})
# add_executable(${PROJECT_NAME} test_open3d_vis.cpp)

target_link_libraries(${PROJECT_NAME}
  libnvinfer.so
  libnvonnxparser.so
  yaml-cpp
  ${Open3D_LIBRARIES}
)