cmake_minimum_required(VERSION 3.0)
project(kinectfusion)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -O3 -march=native -Wno-error=unused-parameter -Wno-error=unused-variable ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -O3 -march=native -Wno-error=unused-parameter -Wno-error=unused-variable ")

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(../KinectFusionLib/include)
include_directories(../KinectFusionApp/include)
cuda_add_library(kinectfusion-original SHARED kinectfusion_library.cpp)
target_link_libraries(kinectfusion-original ${OpenCV_LIBS} KinectFusion)
generate_slam_library(kinectfusion-original "")
