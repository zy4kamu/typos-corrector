cmake_minimum_required(VERSION 2.8)
project(prefix-tree)

add_library(prefix-tree SHARED prefix-tree.cpp prefix-tree-builder.cpp)
target_link_libraries(prefix-tree)

add_executable(prefix-tree-app main.cpp)
target_link_libraries(prefix-tree-app prefix-tree)
