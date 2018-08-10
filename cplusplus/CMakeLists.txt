cmake_minimum_required(VERSION 2.8)
project(typos-corrector)

SET(COMMON_CXX_FLAGS "-std=c++14 -m64 -fPIC -Wall -Werror")

if (${CMAKE_BUILD_TYPE} MATCHES RELEASE)
    SET(CMAKE_CXX_FLAGS "-Ofast ${COMMON_CXX_FLAGS}")
else ()
    MESSAGE(WARNING "Building DEBUG version: -O0 is used")
    SET(CMAKE_CXX_FLAGS "-O0 -g ${COMMON_CXX_FLAGS}")
endif ()

add_subdirectory(network-hypo-searcher)
add_subdirectory(network-talker)
add_subdirectory(python-bindings)
add_subdirectory(utils)
