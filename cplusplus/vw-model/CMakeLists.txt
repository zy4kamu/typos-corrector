cmake_minimum_required(VERSION 2.8)
project(vw-model)

add_library(vw-model SHARED vw-model.cpp)
add_executable(vw-model-app vw-model.cpp main.cpp)
