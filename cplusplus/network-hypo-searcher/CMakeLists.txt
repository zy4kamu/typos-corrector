cmake_minimum_required(VERSION 2.8)
project(network-hypo-searcher)

# NOTE: Ubuntu 16.04 by default contains libraries for OpenCL 1.2.
# Had to add this workaround to eliminate corresponding problems
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DROOT_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

if (${PLATFORM} MATCHES ANDROID)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../opencl-connector")
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/../opencl-connector/lib")
endif ()

add_library(network-hypo-searcher SHARED compressed-lstm.cpp
                                         compressed-lstm.cl
                                         hypo-searcher.cpp
                                         network-automata.cpp
                                         network-automata.cl
                                         common.h)
target_link_libraries(network-hypo-searcher dataset opencl-connector utils)

add_executable(network-hypo-searcher-app main.cpp)
target_link_libraries(network-hypo-searcher-app network-hypo-searcher)
