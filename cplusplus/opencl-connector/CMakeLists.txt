cmake_minimum_required(VERSION 2.8)
project(opencl-connector)

# NOTE: Ubuntu 16.04 by default contains libraries for OpenCL 1.2.
# Had to add this workaround to eliminate corresponding problems
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DROOT_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/CLBLAS")

file(GLOB OpenCLHeaders "CL/*.h")

add_library(opencl-connector SHARED opencl-connector.cpp cl2.hpp common.h ${OpenCLHeaders})
target_link_libraries(opencl-connector OpenCL clBLAS)

add_executable(opencl-connector-dummy main.cpp)
target_link_libraries(opencl-connector-dummy opencl-connector)
