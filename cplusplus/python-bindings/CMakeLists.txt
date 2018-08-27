cmake_minimum_required(VERSION 2.8)
project(python-bindings)

add_library(python-bindings SHARED contaminator.cpp random-batch-generator.cpp compressor.cpp python-bindings.cpp ngrams.cpp dataset.cpp)
target_link_libraries(python-bindings boost_filesystem boost_system utils)
