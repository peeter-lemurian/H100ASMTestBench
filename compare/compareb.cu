#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include "cuda_check.hpp"

inline __device__ uint32_t cmp_lt(float a, float b) {
  uint32_t result = (a < b);
  // future: inline asm implementation.
  return result;
}

inline __device__ uint32_t cmp_gt(float a, float b) {
  uint32_t result = (a > b);
  // future: inline asm implementation.
  return result;
}

inline __device__ uint32_t cmp_eq(float a, float b) {
  uint32_t result = (a == b);
  // future: inline asm implementation.
  return result;
}

struct Comparator {
  static constexpr size_t N = 16;
  float input[N];
  uint32_t output[N - 1];

  __host__ Comparator();

  void __host__ reset() { std::memset(output, 0xff, sizeof(output)); }

  void __host__
  displayAndCheckResults(const char *what, const char *op,
                         std::function<float(float, float)> expected) const;
};

////////////////////////////////////////////////////////////////
/// Single-element kernels (for assembly inspection):
///
__global__ void convertKernelOneLess(Comparator *self) {
  self->output[0] = (self->input[0] < self->input[1]);
}

__global__ void convertKernelOneGreater(Comparator *self) {
  self->output[0] = (self->input[0] > self->input[1]);
}

__global__ void convertKernelOneEqual(Comparator *self) {
  self->output[0] = (self->input[0] == self->input[1]);
}

////////////////////////////////////////////////////////////////
/// Parallel kernels: one thread per output element.
/// Launch with blockSize(N-1), gridSize(1).
///
__global__ void convertKernelLessC(Comparator *self) {
  int i = threadIdx.x;  // i in [0, N-2]
  self->output[i] = (self->input[i] < self->input[i + 1]);
}

__global__ void convertKernelGreaterC(Comparator *self) {
  int i = threadIdx.x;
  self->output[i] = (self->input[i] > self->input[i + 1]);
}

__global__ void convertKernelEqualC(Comparator *self) {
  int i = threadIdx.x;
  self->output[i] = (self->input[i] == self->input[i + 1]);
}

__global__ void convertKernelLessASM(Comparator *self) {
  int i = threadIdx.x;
  self->output[i] = cmp_lt(self->input[i], self->input[i + 1]);
}

__global__ void convertKernelGreaterASM(Comparator *self) {
  int i = threadIdx.x;
  self->output[i] = cmp_gt(self->input[i], self->input[i + 1]);
}

__global__ void convertKernelEqualASM(Comparator *self) {
  int i = threadIdx.x;
  self->output[i] = cmp_eq(self->input[i], self->input[i + 1]);
}

// kernels for just looking at the ASM
__global__ void convertKernelOnegLessASM(Comparator *self) {
  int i = 0;
  self->output[i] = cmp_lt(self->input[i], self->input[i + 1]);
}

__global__ void convertKernelOnegGreaterASM(Comparator *self) {
  int i = 0;
  self->output[i] = cmp_gt(self->input[i], self->input[i + 1]);
}

__global__ void convertKernelOnegEqualASM(Comparator *self) {
  int i = 0;
  self->output[i] = cmp_eq(self->input[i], self->input[i + 1]);
}

__host__ Comparator::Comparator() {
  input[0]  = 1.5f;
  input[1]  = 1.5001f;
  input[2]  = -0.0f;
  input[3]  = +0.0f;
  input[4]  = std::numeric_limits<float>::infinity();
  input[5]  = std::numeric_limits<float>::infinity();
  input[6]  = -std::numeric_limits<float>::infinity();
  input[7]  = -1.5f;
  input[8]  = std::numeric_limits<float>::max();
  input[9]  = std::numeric_limits<float>::infinity();
  input[10] = std::numeric_limits<float>::quiet_NaN();
  input[11] = 1.0f;
  input[12] = 1.0f;
  input[13] = std::numeric_limits<float>::quiet_NaN();
  input[14] = std::numeric_limits<float>::min();
  input[15] = std::numeric_limits<float>::denorm_min();
}

void __host__ Comparator::displayAndCheckResults(
    const char *what, const char *op,
    std::function<float(float, float)> expected) const {
  for (int i = 0; i < (int)(N - 1); i++) {
    uint32_t exp = expected(input[i], input[i + 1]);
    bool ok = (output[i] == exp);
    assert(ok);
    std::cout << std::format("{}: {} {} {} = {} (expected: {}) {}\n", what,
                             input[i], op, input[i + 1], output[i], exp,
                             ok ? "OK" : "FAIL");
  }
}

int main() {
  Comparator *converter;
  CUDA_CHECK(cudaMallocManaged(&converter, sizeof(Comparator)));
  new (converter) Comparator();

  // 15 threads, one per adjacent pair
  dim3 blockSize(Comparator::N - 1);
  dim3 gridSize(1);

  auto lt = [](float a, float b) { return a < b ? 1 : 0; };
  auto gt = [](float a, float b) { return a > b ? 1 : 0; };
  auto eq = [](float a, float b) { return a == b ? 1 : 0; };

  converter->reset();
  convertKernelLessC<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("[C++]", "<", lt);

  converter->reset();
  convertKernelGreaterC<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("[C++]", ">", gt);

  converter->reset();
  convertKernelEqualC<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("[C++]", "==", eq);

  converter->reset();
  convertKernelLessASM<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("[ASM]", "<", lt);

  converter->reset();
  convertKernelGreaterASM<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("[ASM]", ">", gt);

  converter->reset();
  convertKernelEqualASM<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("[ASM]", "==", eq);

  CUDA_CHECK(cudaFree(converter));
  return 0;
}

// vim: et ts=2 sw=2
