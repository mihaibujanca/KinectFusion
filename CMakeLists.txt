cmake_minimum_required(VERSION 3.0)

project(KinectFusionApplication)
# Use modern C++
set(CMAKE_CXX_STANDARD 14)
# Targets
add_subdirectory(KinectFusionLib)
add_subdirectory(KinectFusionApp)

include_directories(KinectFusionLib/include)
include_directories(KinectFusionApp/include)
cuda_add_library(kinectfusion-original SHARED kinectfusion_library.cpp)
target_link_libraries(kinectfusion-original ${OpenCV_LIBS} KinectFusion)
generate_slam_library(kinectfusion-original "")
