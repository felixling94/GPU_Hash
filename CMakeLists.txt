cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

SET(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)

project(GPU_Hash LANGUAGES CXX CUDA)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61")

include_directories(${CMAKE_SOURCE_DIR}/include   /core   /test   /tools)

add_subdirectory(test)
