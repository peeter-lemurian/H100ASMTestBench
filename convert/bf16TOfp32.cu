#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include "cuda_check.hpp"

// BF16 to FP32 conversion using shift (zero-extension)
// This is the canonical method - BF16 is just FP32 with lower 16 bits zeroed
inline float __device__ bf16TOfp32Shift(__nv_bfloat16 input) {
  uint16_t bf16_bits;
  uint32_t result;

  // Copy bfloat16 to uint16_t
  __builtin_memcpy(&bf16_bits, &input, sizeof(__nv_bfloat16));

  result = bf16_bits;
  result <<= 16;

  float fp32_result;
  __builtin_memcpy(&fp32_result, &result, sizeof(float));
  return fp32_result;
}

class Bf16ToFp32Converter {
public:
  static constexpr size_t N = 24;
  __nv_bfloat16 input[N];
  float output[N];

  // Helper to create a bfloat16 from raw bits
  static inline __nv_bfloat16 __host__ bfloat16FromBits(uint16_t bits) {
    __nv_bfloat16 bf;
    std::memcpy(&bf, &bits, sizeof(bf));
    return bf;
  }

  // Helper to create a float from raw bits
  static inline float __host__ floatFromBits(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }

  __host__ Bf16ToFp32Converter();

  void __host__ reset()
  {
    std::memset(output, 0xff, sizeof(output));
  }

  void __host__ displayAndCheckResults(const char *how) const;
};

////////////////////////////////////////////////////////////////
/// Kernels just for looking at the generated assembly:
///
__global__ void convertKernelOneValue(Bf16ToFp32Converter *self) {
  self->output[0] = __bfloat162float(self->input[0]);
}

__global__ void convertKernelOneValueShift(Bf16ToFp32Converter *self) {
  self->output[0] = bf16TOfp32Shift(self->input[0]);
}


////////////////////////////////////////////////////////////////
/// Kernels that use each of the conversion methods:
///
__global__ void convertKernel(Bf16ToFp32Converter *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Bf16ToFp32Converter::N) {
    self->output[idx] = __bfloat162float(self->input[idx]);
  }
}

__global__ void convertKernelShift(Bf16ToFp32Converter *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Bf16ToFp32Converter::N) {
    self->output[idx] = bf16TOfp32Shift(self->input[idx]);
  }
}

__host__ Bf16ToFp32Converter::Bf16ToFp32Converter() {
  // Initialize input data with various test cases
  input[0] = bfloat16FromBits(0x3f80);  // 1.0
  input[1] = bfloat16FromBits(0xbf80);  // -1.0
  input[2] = bfloat16FromBits(0x4049);  // 3.140625 (Pi-ish)

  // Various normal values
  input[3] = bfloat16FromBits(0x3f81);  // ~1.0078125
  input[4] = bfloat16FromBits(0x4000);  // 2.0
  input[5] = bfloat16FromBits(0x4040);  // 3.0
  input[6] = bfloat16FromBits(0x4080);  // 4.0

  // Special values - Infinity
  input[7] = bfloat16FromBits(0x7f80);  // +Inf
  input[8] = bfloat16FromBits(0xff80);  // -Inf

  // Quiet NaNs (bit 6 = 1, the quiet bit in BF16)
  input[9] = bfloat16FromBits(0x7fc0);   // QNaN (canonical)
  input[10] = bfloat16FromBits(0xffc0);  // -QNaN
  input[11] = bfloat16FromBits(0x7fc1);  // QNaN with payload
  input[12] = bfloat16FromBits(0x7fff);  // QNaN with max payload

  // Signaling NaNs (bit 6 = 0, the quiet bit is clear)
  input[13] = bfloat16FromBits(0x7f81);  // SNaN with minimal payload
  input[14] = bfloat16FromBits(0xff81);  // -SNaN with minimal payload
  input[15] = bfloat16FromBits(0x7fa0);  // SNaN with some payload
  input[16] = bfloat16FromBits(0x7fbf);  // SNaN with max payload (bit 6=0)
  input[17] = bfloat16FromBits(0xffa0);  // -SNaN with some payload
  input[18] = bfloat16FromBits(0xffbf);  // -SNaN with max payload

  // Denormal (subnormal) values
  input[19] = bfloat16FromBits(0x0001);  // Smallest denorm
  input[20] = bfloat16FromBits(0x007f);  // Largest denorm
  input[21] = bfloat16FromBits(0x8001);  // Negative denorm

  // Edge cases
  input[22] = bfloat16FromBits(0x0000);  // Positive zero
  input[23] = bfloat16FromBits(0x8000);  // Negative zero
}

void __host__
Bf16ToFp32Converter::displayAndCheckResults(const char *how) const {
  // Verify results on CPU with hex dumps
  std::cout << std::format("All conversions: method: {}\n"
                           "{:>4} {:>15} {:>6} {:>14} {:>10} {:>12} {:>12}\n",
                           how, "Idx", "BF16 Value", "BF16", "FP32 Value",
                           "FP32 Hex", "BF16 Type", "FP32 Type");
  std::cout << std::string(91, '-') << "\n";

  for (size_t i = 0; i < Bf16ToFp32Converter::N; i++) {
    float original = __bfloat162float(input[i]);
    float converted = output[i];

    // Get raw bits for hex dump
    uint16_t bf16_bits;
    uint32_t fp32_bits;
    std::memcpy(&bf16_bits, &input[i], sizeof(bf16_bits));
    std::memcpy(&fp32_bits, &converted, sizeof(fp32_bits));

    // Determine BF16 input type
    std::string bf16_notes;
    if (std::isnan(original)) {
      bool bf16_is_qnan = (bf16_bits & 0x0040) != 0; // bit 6
      std::string sign = (bf16_bits & 0x8000) ? "-" : "+";
      bf16_notes = std::format("{}{}", sign, bf16_is_qnan ? "QNaN" : "SNaN");
    } else if (std::isinf(original)) {
      bf16_notes = original > 0 ? "+Inf" : "-Inf";
    } else if (bf16_bits == 0x0000) {
      bf16_notes = "+0";
    } else if (bf16_bits == 0x8000) {
      bf16_notes = "-0";
    } else if ((bf16_bits & 0x7f80) == 0) {
      bf16_notes = "Denorm";
    } else {
      bf16_notes = "Normal";
    }

    // Determine FP32 output type
    std::string fp32_notes;
    if (std::isnan(converted)) {
      bool fp32_is_qnan = (fp32_bits & 0x00400000) != 0; // bit 22
      std::string sign = (fp32_bits & 0x80000000) ? "-" : "+";
      fp32_notes = std::format("{}{}", sign, fp32_is_qnan ? "QNaN" : "SNaN");
    } else if (std::isinf(converted)) {
      fp32_notes = converted > 0 ? "+Inf" : "-Inf";
    } else if (fp32_bits == 0x00000000) {
      fp32_notes = "+0";
    } else if (fp32_bits == 0x80000000) {
      fp32_notes = "-0";
    } else if ((fp32_bits & 0x7f800000) == 0) {
      fp32_notes = "Denorm";
    } else {
      fp32_notes = "Normal";
    }

    std::cout << std::format(
        "{:>4} {:>15.8g} 0x{:04x} {:>14.8g} 0x{:08x} {:>12} {:>12}\n", i,
        original, bf16_bits, converted, fp32_bits, bf16_notes, fp32_notes);
  }

  // Verify accuracy - BF16 to FP32 should be lossless (exact)
  std::cout << "\nAccuracy check:\n";
  size_t errors = 0;
  size_t checked = 0;
  for (size_t i = 0; i < Bf16ToFp32Converter::N; i++) {
    float expected = __bfloat162float(input[i]);
    float converted = output[i];

    // Get raw bits
    uint32_t expected_bits, converted_bits;
    std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
    std::memcpy(&converted_bits, &converted, sizeof(converted_bits));

    // BF16 to FP32 conversion should be exact - just zero-extending
    // Expected FP32 bits should have upper 16 bits from BF16, lower 16 bits zero
    uint16_t bf16_bits;
    std::memcpy(&bf16_bits, &input[i], sizeof(bf16_bits));
    uint32_t expected_fp32_bits = static_cast<uint32_t>(bf16_bits) << 16;

    if (converted_bits != expected_fp32_bits) {
      errors++;
      std::cout << std::format(
          "  [{}] Mismatch: BF16=0x{:04x} -> Expected FP32=0x{:08x}, Got=0x{:08x}\n",
          i, bf16_bits, expected_fp32_bits, converted_bits);
    }
    checked++;
  }

  std::cout << std::format("\nTotal elements: {}\n", Bf16ToFp32Converter::N);
  std::cout << std::format("Elements checked: {}\n", checked);
  std::cout << std::format("Conversion errors (should be 0): {}\n", errors);
}

int main() {
  // Allocate converter object in managed memory
  Bf16ToFp32Converter *converter;
  CUDA_CHECK(cudaMallocManaged(&converter, sizeof(Bf16ToFp32Converter)));

  new (converter) Bf16ToFp32Converter();

  // Launch kernel
  dim3 blockSize(Bf16ToFp32Converter::N);
  dim3 gridSize(1);

  converter->reset();
  convertKernel<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("__bfloat162float");

  converter->reset();
  convertKernelShift<<<gridSize, blockSize>>>(converter);
  CUDA_CHECK(cudaDeviceSynchronize());
  converter->displayAndCheckResults("bf16TOfp32Shift");

  // Cleanup
  CUDA_CHECK(cudaFree(converter));

  return 0;
}

// vim: et ts=2 sw=2
