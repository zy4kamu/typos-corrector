cmake_minimum_required(VERSION 2.8)
project(typos-corrector)

SET(COMMON_CXX_FLAGS "-std=c++14 -m64 -fPIC -Wall -Werror")
add_definitions(-DROOT_DIRECTORY="${CMAKE_SOURCE_DIR}")

if (${CMAKE_BUILD_TYPE} MATCHES RELEASE)
    SET(CMAKE_CXX_FLAGS "-Ofast ${COMMON_CXX_FLAGS}")
else ()
    MESSAGE(WARNING "Building DEBUG version: -O0 is used")
    SET(CMAKE_CXX_FLAGS "-O0 -g ${COMMON_CXX_FLAGS}")
endif ()

add_library(python-bindings SHARED contaminator.cpp random-batch-generator.cpp compressor.cpp update-regions.cpp python-bindings.cpp utils.cpp ngrams.cpp)
target_link_libraries(python-bindings boost_filesystem boost_system)

add_subdirectory(proto)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(network-talker network-talker.cpp)
target_link_libraries(network-talker proto protobuf)
