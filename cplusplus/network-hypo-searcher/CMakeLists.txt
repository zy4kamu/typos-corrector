cmake_minimum_required(VERSION 2.8)
project(network-hypo-searcher)

add_definitions(-DROOT_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

if (USE_OPENCL)
    # NOTE: Ubuntu 16.04 by default contains libraries for OpenCL 1.2.
    # Had to add this workaround to eliminate corresponding problems
    add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)
    add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)

    if (${PLATFORM} MATCHES ANDROID)
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../opencl-connector")
        link_directories("${CMAKE_CURRENT_SOURCE_DIR}/../opencl-connector/lib")
    endif ()

    add_library(network-hypo-searcher SHARED common.cpp
                                             compressed-lstm-gpu.cpp
                                             i-database-requester.cpp
                                             hypo-searcher.cpp
                                             multi-hypo-searcher.cpp
                                             network-automata-gpu.cpp
                                             vw-model.cpp)
    target_link_libraries(network-hypo-searcher opencl-connector prefix-tree)
    add_executable(network-hypo-searcher-app main.cpp)
    target_link_libraries(network-hypo-searcher-app network-hypo-searcher prefix-tree)
else()
    add_library(network-hypo-searcher SHARED common.cpp
                                             compressed-lstm-cpu.cpp
                                             i-database-requester.cpp
                                             hypo-searcher.cpp
                                             multi-hypo-searcher.cpp
                                             network-automata-cpu.cpp
                                             utils.cpp
                                             vw-model.cpp)
    target_link_libraries(network-hypo-searcher prefix-tree)
    add_executable(network-hypo-searcher-app main.cpp)
    target_link_libraries(network-hypo-searcher-app network-hypo-searcher dataset prefix-tree)
endif()
