cmake_minimum_required(VERSION 2.8)
project(lstm)

add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=200)
add_definitions(-DROOT_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

add_library(lstm SHARED sum-counter.cpp)
target_link_libraries(lstm)

add_executable(lstm-app lstm.cpp)
target_link_libraries(lstm-app lstm OpenCL)
