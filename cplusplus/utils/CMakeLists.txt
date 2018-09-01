cmake_minimum_required(VERSION 2.8)
project(utils)

add_library(utils STATIC utils.cpp)
target_link_libraries(utils)
