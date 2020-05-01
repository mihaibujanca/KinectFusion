cmake_minimum_required(VERSION 3.0)

project(KinectFusionApplication)
# Use modern C++
set(CMAKE_CXX_STANDARD 14)
# Targets
add_subdirectory(KinectFusionLib)
add_subdirectory(KinectFusionApp)
add_subdirectory(SLAMBenchApp)
