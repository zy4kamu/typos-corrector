cmake_minimum_required(VERSION 2.8)
project(dataset)

add_library(dataset STATIC dataset.cpp)
target_link_libraries(dataset utils)
