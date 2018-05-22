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

add_library(prefix-tree SHARED prefix-tree.cpp)

add_executable(prepare-dataset dataset-preparer.cpp)
target_link_libraries(prepare-dataset boost_filesystem boost_system boost_program_options prefix-tree)
