#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <format>
#include <iostream>

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t error = (cmd);                                                 \
    if (error != cudaSuccess) {                                                \
      std::cerr << std::format("CUDA error: {} at {}:{}\n",                    \
                               cudaGetErrorString(error), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// vim: et ts=2 sw=2
