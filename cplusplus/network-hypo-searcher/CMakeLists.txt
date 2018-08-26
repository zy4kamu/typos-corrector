cmake_minimum_required(VERSION 2.8)
project(lstm)

# NOTE: Ubuntu 16.04 by default contains libraries for OpenCL 1.2.
# Had to add this workaround to eliminate corresponding problems
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DROOT_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

add_library(lstm SHARED compressed-lstm.cpp 
                        compressed-lstm.cl 
                        hypo-searcher.cpp
                        network-automata.cpp 
                        network-automata.cl 
                        opencl-connector.cpp 
                        common.h)
target_link_libraries(lstm utils python-bindings OpenCL cblas clBLAS boost_filesystem boost_system)

add_executable(lstm-app main.cpp)
target_link_libraries(lstm-app lstm)
