cmake_minimum_required(VERSION 2.8)
project(opencl-connector)

# NOTE: Ubuntu 16.04 by default contains libraries for OpenCL 1.2.
# Had to add this workaround to eliminate corresponding problems
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DROOT_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

if (${PLATFORM} MATCHES ANDROID)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")
endif ()

file(GLOB OpenCLHeaders "CL/*.h")

add_library(opencl-connector SHARED gemm-processor.cpp 
                                    gemm-processor.cl
                                    opencl-connector.cpp 
                                    cl2.hpp 
                                    common.h 
                                    ${OpenCLHeaders})
target_link_libraries(opencl-connector OpenCL)

add_executable(opencl-connector-dummy main.cpp)
target_link_libraries(opencl-connector-dummy opencl-connector)
