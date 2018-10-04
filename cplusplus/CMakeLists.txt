cmake_minimum_required(VERSION 2.8)
project(typos-corrector)

option(BUILD_PYTHON_BINDINGS "Build library used by python wrapper" ON)
option(BUILD_HYPO_SEARCHER   "Build complete C++ application"       ON)
option(USE_OPENCL            "Uses OpenCL for maths computations"   ON)

if (USE_OPENCL)
    add_definitions(-DUSE_OPENCL)
endif()

################### SET COMPILER AND LINKER FLAGS ###################
if (${PLATFORM} MATCHES ANDROID)
    MESSAGE(WARNING "Building for Android")
    SET(COMMON_CXX_FLAGS "-std=c++14 -Wall -Werror -Wno-missing-braces -ffast-math -ffinite-math-only -funroll-loops -funsafe-math-optimizations -fvectorize ")
    set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++")
else ()
    MESSAGE(WARNING "Building for Linux")
    SET(COMMON_CXX_FLAGS "-std=c++14 -m64 -fPIC -Wall -Werror")
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES RELEASE)
    MESSAGE(WARNING "Building RELEASE version: -Ofast is used")
    SET(CMAKE_CXX_FLAGS "-Ofast ${COMMON_CXX_FLAGS}")
else ()
    MESSAGE(WARNING "Building DEBUG version: -O0 is used")
    SET(CMAKE_CXX_FLAGS "-O0 -g ${COMMON_CXX_FLAGS}")
endif ()

########################## LOAD PROJECTS #############################

add_subdirectory(dataset)
add_subdirectory(network-hypo-searcher)
add_subdirectory(dataset-generator)

if (BUILD_PYTHON_BINDINGS)
    add_subdirectory(python-bindings)
endif (BUILD_PYTHON_BINDINGS)

if (USE_OPENCL)
    add_subdirectory(opencl-connector)
endif()
