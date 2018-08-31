cmake_minimum_required(VERSION 2.8)
project(python-bindings)

add_library(python-bindings SHARED python-bindings.cpp)
target_link_libraries(python-bindings dataset dataset-generator utils)
