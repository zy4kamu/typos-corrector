cmake_minimum_required(VERSION 2.8)
project(python-bindings)

add_library(python-bindings SHARED contaminator.cpp random-batch-generator.cpp compressor.cpp update-regions.cpp python-bindings.cpp utils.cpp ngrams.cpp)
target_link_libraries(python-bindings boost_filesystem boost_system)
