cmake_minimum_required(VERSION 2.8)
project(dataset-generator)

add_library(dataset-generator SHARED contaminator.cpp random-batch-generator.cpp ngrams.cpp)
target_link_libraries(dataset-generator)

add_executable(dataset-generator-app main.cpp)
target_link_libraries(dataset-generator-app dataset-generator dataset)
