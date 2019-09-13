cmake_minimum_required(VERSION 2.8)
project( dirt_detection_spectral )
set(OpenCV_DIR /home/rmb-jx/Xinjie/opencv-3.4.2/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")
add_executable(dirt_detection
        dirt_detection_standalone_spectral.cpp
        )
target_link_libraries( dirt_detection ${OpenCV_LIBS} )
