cmake_minimum_required(VERSION 2.8)
project(utils)

add_library(utils SHARED utils.cpp)
target_link_libraries(utils)
