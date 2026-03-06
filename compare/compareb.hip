#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      std::cerr << std::format("HIP error: {} at {}:{}\n",                     \
                               hipGetErrorString(error), __FILE__, __LINE__);  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

inline __device__ uint32_t vcmp_lt_select_vcc(float a, float b) {
  uint32_t result;
  __asm__ __volatile__("v_cmp_lt_f32_e32 vcc, %1, %2\n\t"
               "s_nop 1\n\t"
               "v_cndmask_b32_e64 %0, 0, 1, vcc\n\t"
               : "=v"(result) // %0
               : "v"(a), // %1
                 "v"(b) // %2
               : "vcc");
  return result;
}

inline __device__ uint32_t vcmp_gt_select_vcc(float a, float b) {
  uint32_t result;
  __asm__ __volatile__("v_cmp_gt_f32_e32 vcc, %1, %2\n\t"
               "s_nop 1\n\t"
               "v_cndmask_b32_e64 %0, 0, 1, vcc\n\t"
               : "=v"(result) // %0
               : "v"(a), // %1
                 "v"(b) // %2
               : "vcc");
  return result;
}

inline __device__ uint32_t vcmp_eq_select_vcc(float a, float b) {
  uint32_t result;
  __asm__ __volatile__("v_cmp_eq_f32_e32 vcc, %1, %2\n\t"
               "s_nop 1\n\t"
               "v_cndmask_b32_e64 %0, 0, 1, vcc\n\t"
               : "=v"(result) // %0
               : "v"(a), // %1
                 "v"(b) // %2
               : "vcc");
  return result;
}

inline __device__ uint32_t vcmp_lt_select_stemp(float a, float b) {
  uint32_t result;
  uint64_t compareTemp;
  __asm__ __volatile__("v_cmp_lt_f32 %1, %2, %3\n\t"
               "s_nop 1\n\t"
               "v_cndmask_b32_e64 %0, 0, 1, %1\n\t"
               : "=v"(result), // %0
                 "=s"(compareTemp) // %1
               : "v"(a), // %2
                 "v"(b) // %3
               : /* no clobbers */ );
  return result;
}

inline __device__ uint32_t vcmp_eq_select_stemp(float a, float b) {
  uint32_t result;
  uint64_t compareTemp;
  __asm__ __volatile__("v_cmp_eq_f32 %1, %2, %3\n\t"
               "s_nop 1\n\t"
               "v_cndmask_b32_e64 %0, 0, 1, %1\n\t"
               : "=v"(result), // %0
                 "=s"(compareTemp) // %1
               : "v"(a), // %2
                 "v"(b) // %3
               : /* no clobbers */ );
  return result;
}

inline __device__ uint32_t vcmp_gt_select_stemp(float a, float b) {
  uint32_t result;
  uint64_t compareTemp;
  __asm__ __volatile__("v_cmp_gt_f32 %1, %2, %3\n\t"
               "s_nop 1\n\t"
               "v_cndmask_b32_e64 %0, 0, 1, %1\n\t"
               : "=v"(result), // %0
                 "=s"(compareTemp) // %1
               : "v"(a), // %2
                 "v"(b) // %3
               : /* no clobbers */ );
  return result;
}

class Comparator {
public:
  static constexpr size_t N = 16;
  float input[N];
  uint32_t output[N - 1];

  __host__ Comparator();

  void __host__ reset() { std::memset(output, 0xff, sizeof(output)); }

  void __host__
  displayAndCheckResults(const char *what, const char *op,
                         std::function<float(float, float)> expected) const;

  ////////////////////////////////////////////////////////////////
  /// Single-element kernels (for assembly inspection):
  ///
  __global__ static void convertKernelOneLess(Comparator *self) {
    self->output[0] = (self->input[0] < self->input[1]);
  }

  __global__ static void convertKernelOneGreater(Comparator *self) {
    self->output[0] = (self->input[0] > self->input[1]);
  }

  __global__ static void convertKernelOneEqual(Comparator *self) {
    self->output[0] = (self->input[0] == self->input[1]);
  }

  ////////////////////////////////////////////////////////////////
  /// Parallel kernels: one thread per output element.
  /// Launch with blockSize(N-1), gridSize(1).
  ///
  __global__ static void convertKernelLessC(Comparator *self) {
    int i = threadIdx.x;  // i in [0, N-2]
    self->output[i] = (self->input[i] < self->input[i + 1]);
  }

  __global__ static void convertKernelGreaterC(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = (self->input[i] > self->input[i + 1]);
  }

  __global__ static void convertKernelEqualC(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = (self->input[i] == self->input[i + 1]);
  }

#if 0
  __global__ static void convertKernelLessASM(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = vcmp_lt_select_vcc(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelGreaterASM(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = vcmp_gt_select_vcc(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelEqualASM(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = vcmp_eq_select_vcc(self->input[i], self->input[i + 1]);
  }
#else
  __global__ static void convertKernelLessASM(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = vcmp_lt_select_stemp(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelGreaterASM(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = vcmp_gt_select_stemp(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelEqualASM(Comparator *self) {
    int i = threadIdx.x;
    self->output[i] = vcmp_eq_select_stemp(self->input[i], self->input[i + 1]);
  }
#endif

  // kernels for just looking at the ASM
  __global__ static void convertKernelOneVCCLessASM(Comparator *self) {
    int i = 0;
    self->output[i] = vcmp_lt_select_vcc(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelOneVCCGreaterASM(Comparator *self) {
    int i = 0;
    self->output[i] = vcmp_gt_select_vcc(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelOneVCCEqualASM(Comparator *self) {
    int i = 0;
    self->output[i] = vcmp_eq_select_vcc(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelOneStempLessASM(Comparator *self) {
    int i = 0;
    self->output[i] = vcmp_lt_select_stemp(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelOneStempGreaterASM(Comparator *self) {
    int i = 0;
    self->output[i] = vcmp_gt_select_stemp(self->input[i], self->input[i + 1]);
  }

  __global__ static void convertKernelOneStempEqualASM(Comparator *self) {
    int i = 0;
    self->output[i] = vcmp_eq_select_stemp(self->input[i], self->input[i + 1]);
  }
};

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
  HIP_CHECK(hipMallocManaged(&converter, sizeof(Comparator)));
  new (converter) Comparator();

  // 15 threads, one per adjacent pair
  dim3 blockSize(Comparator::N - 1);
  dim3 gridSize(1);

  auto lt = [](float a, float b) { return a < b ? 1 : 0; };
  auto gt = [](float a, float b) { return a > b ? 1 : 0; };
  auto eq = [](float a, float b) { return a == b ? 1 : 0; };

  converter->reset();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Comparator::convertKernelLessC),
                     gridSize, blockSize, 0, 0, converter);
  HIP_CHECK(hipDeviceSynchronize());
  converter->displayAndCheckResults("[C++]", "<", lt);

  converter->reset();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Comparator::convertKernelGreaterC),
                     gridSize, blockSize, 0, 0, converter);
  HIP_CHECK(hipDeviceSynchronize());
  converter->displayAndCheckResults("[C++]", ">", gt);

  converter->reset();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Comparator::convertKernelEqualC),
                     gridSize, blockSize, 0, 0, converter);
  HIP_CHECK(hipDeviceSynchronize());
  converter->displayAndCheckResults("[C++]", "==", eq);

  converter->reset();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Comparator::convertKernelLessASM),
                     gridSize, blockSize, 0, 0, converter);
  HIP_CHECK(hipDeviceSynchronize());
  converter->displayAndCheckResults("[ASM]", "<", lt);

  converter->reset();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Comparator::convertKernelGreaterASM),
                     gridSize, blockSize, 0, 0, converter);
  HIP_CHECK(hipDeviceSynchronize());
  converter->displayAndCheckResults("[ASM]", ">", gt);

  converter->reset();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Comparator::convertKernelEqualASM),
                     gridSize, blockSize, 0, 0, converter);
  HIP_CHECK(hipDeviceSynchronize());
  converter->displayAndCheckResults("[ASM]", "==", eq);

  HIP_CHECK(hipFree(converter));
  return 0;
}

// vim: et ts=2 sw=2
