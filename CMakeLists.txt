cmake_minimum_required(VERSION 3.20)

# Match the original Makefile behavior by preferring this nvcc path when available.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-13.0/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc")
endif()

project(H100ASMTestBench LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Global CUDA compile option: keep intermediate nvcc compilation files.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--keep>)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(compare)
add_subdirectory(convert)
add_subdirectory(demo)
add_subdirectory(math)
