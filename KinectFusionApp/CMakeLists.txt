cmake_minimum_required(VERSION 3.0)

project(KinectFusionApp)

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
## CUDA
find_package(CUDA REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
ENDIF ()

## OpenCV
# Optional: Set OpenCV_DIR if you want to use a custom version of OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

## Eigen3
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)

## OpenNI2
find_package(openni2)
if (OPENNI2_FOUND)
    include_directories("${OPENNI2_INCLUDE_DIR}")
endif (OPENNI2_FOUND)

## Intel RealSense
find_package(realsense2)
if (realsense2_FOUND)
    include_directories("${realsense2_INCLUDE_DIR}")
endif (realsense2_FOUND)
# ------------------------------------------------

include_directories(${PROJECT_INCLUDE_DIR})
include_directories(../KinectFusionLib/include)

file(GLOB KinectFusionApp_SRCS ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(KinectFusionApp ${KinectFusionApp_SRCS})
target_link_libraries(KinectFusionApp ${OpenCV_LIBS} ${OPENNI2_LIBRARY} ${realsense2_LIBRARY} KinectFusion)