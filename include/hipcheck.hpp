#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <format>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      std::cerr << std::format("HIP error: {} at {}:{}\n",                     \
                               hipGetErrorString(error), __FILE__, __LINE__);  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

