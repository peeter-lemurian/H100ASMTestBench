//
// fp32TObf16.hip
//
// Tests FP32->BF16 conversion on AMD MI300 hardware, comparing:
//   1. __float2bfloat16 (ROCm builtin reference)
//   2. custom truncation (shift-right only)
//   3. custom round-to-nearest-even-like conversion
//
// A single result table is printed. For mismatch rows, a hex row and
// an error row are printed beneath, in the same style as newer test programs.
//
// Compile:
//   hipcc -O2 -std=c++20 fp32TObf16.hip -o fp32TObf16
//
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <getopt.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>

#include "OneResult16.hpp"
#include "colors.hpp"
#include "hipcheck.hpp"

// clang-format off
inline __hip_bfloat16 __device__ fp32TObf16Rounding(float input) {
  uint32_t input_bits;
  uint32_t result;
  uint32_t round_bit;

  __builtin_memcpy(&input_bits, &input, sizeof(float));

  __asm__ __volatile__(
      "v_bfe_u32 %0, %2, 16, 1\n\t"      // Extract bit 16 for rounding into round_bit
      "v_add_u32_e32 %0, %2, %0\n\t"     // Add bit 16 to original
      "v_add_u32_e32 %1, 0x7fff, %0\n\t" // Add 0x7fff for round-to-nearest
      "v_lshrrev_b32 %1, 16, %1\n\t"     // Shift right 16 bits to get BF16
      : "=&v"(round_bit), // %0
        "=v"(result) // %1
      : "v"(input_bits) // %2
      );

  uint16_t bf16_bits = static_cast<uint16_t>(result);
  return *reinterpret_cast<__hip_bfloat16 *>(&bf16_bits);
}

inline __hip_bfloat16 __device__ fp32TObf16Truncation(float input) {
  uint32_t input_bits;
  uint32_t result;

  __builtin_memcpy(&input_bits, &input, sizeof(float));

  __asm__ __volatile__(
      "v_lshrrev_b32 %0, 16, %1\n\t"
      : "=v"(result) // %0
      : "v"(input_bits) // %1
      );

  uint16_t bf16_bits = static_cast<uint16_t>(result);
  return *reinterpret_cast<__hip_bfloat16 *>(&bf16_bits);
}
// clang-format on

struct ConvertCase {
  float x;
  const char *label;
};

static constexpr ConvertCase kCases[] = {
    {1.0f, "1"},
    {-1.0f, "-1"},
    {3.140625f, "3.140625"},

    // Rounding-sensitive values
    {__builtin_bit_cast(float, 0x3f808000u), "1.00390625 tie"},
    {__builtin_bit_cast(float, 0x3f808001u), "1.00390625+eps"},
    {__builtin_bit_cast(float, 0x3f818000u), "round up"},
    {__builtin_bit_cast(float, 0x3f810000u), "exact bf16"},

    // Infinity
    {std::numeric_limits<float>::infinity(), "+inf"},
    {-std::numeric_limits<float>::infinity(), "-inf"},

    // Quiet NaNs
    {std::numeric_limits<float>::quiet_NaN(), "qNaN"},
    {-std::numeric_limits<float>::quiet_NaN(), "-qNaN"},
    {__builtin_bit_cast(float, 0x7fc00001u), "qNaN payload"},
    {__builtin_bit_cast(float, 0x7fffffffu), "qNaN max payload"},

    // Signaling NaNs
    {std::numeric_limits<float>::signaling_NaN(), "sNaN"},
    {-std::numeric_limits<float>::signaling_NaN(), "-sNaN"},
    {__builtin_bit_cast(float, 0x7f800001u), "sNaN min payload"},
    {__builtin_bit_cast(float, 0x7fbfffffu), "sNaN max payload"},
    {__builtin_bit_cast(float, 0xff800001u), "-sNaN min payload"},
    {__builtin_bit_cast(float, 0xffbfffffu), "-sNaN max payload"},

    // Denormals
    {std::numeric_limits<float>::denorm_min(), "min subnorm"},
    {__builtin_bit_cast(float, 0x007fffffu), "max subnorm"},
    {-std::numeric_limits<float>::denorm_min(), "-min subnorm"},

    // Signed zero
    {0.0f, "+0"},
    {-0.0f, "-0"},
};
static_assert(sizeof(kCases) / sizeof(kCases[0]) == 24,
              "Update Fp32ToBf16Tester::N to match kCases length");

class Fp32ToBf16Tester {
public:
  static constexpr size_t N = 24;

  float input[N];
  __hip_bfloat16 output_ref[N];
  __hip_bfloat16 output_trunc[N];
  __hip_bfloat16 output_round[N];

  __host__ Fp32ToBf16Tester() {
    for (size_t i = 0; i < N; i++) {
      input[i] = kCases[i].x;
    }
  }

  void __host__ reset() {
    std::memset(output_ref, 0xff, sizeof(output_ref));
    std::memset(output_trunc, 0xff, sizeof(output_trunc));
    std::memset(output_round, 0xff, sizeof(output_round));
  }

  void __host__ displayResults() const;

  // -- kernels for full-table testing ---------------------------------------
  __global__ static void testKernelRef(Fp32ToBf16Tester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_ref[idx] = __float2bfloat16(self->input[idx]);
    }
  }

  __global__ static void testKernelTrunc(Fp32ToBf16Tester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_trunc[idx] = fp32TObf16Truncation(self->input[idx]);
    }
  }

  __global__ static void testKernelRound(Fp32ToBf16Tester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_round[idx] = fp32TObf16Rounding(self->input[idx]);
    }
  }

  // -- kernels for one-value asm inspection ---------------------------------
  __global__ static void testKernelOneRef(Fp32ToBf16Tester *self) {
    self->output_ref[0] = __float2bfloat16(self->input[0]);
  }

  __global__ static void testKernelOneTrunc(Fp32ToBf16Tester *self) {
    self->output_trunc[0] = fp32TObf16Truncation(self->input[0]);
  }

  __global__ static void testKernelOneRound(Fp32ToBf16Tester *self) {
    self->output_round[0] = fp32TObf16Rounding(self->input[0]);
  }
};

bool verbose{};
bool useColor{};
bool quiet{};

static std::string fp32Hex(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return std::format("0x{:08x}", bits);
}

static std::string bf16Hex(__hip_bfloat16 v) {
  uint16_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return std::format("0x{:04x}", bits);
}

void __host__ Fp32ToBf16Tester::displayResults() const {
  if (useColor) {
    std::cout << RED;
  }
  std::cout << "FP32 -> BF16 CONVERSION\n\n";
  if (useColor) {
    std::cout << RESET;
  }

  std::cout << std::format("{:>4}"     // Idx
                           "{:>16}"    // x
                           "{:>24}"    // label
                           "{:>21}"    // ref f32
                           "{:>21}"    // trunc f32
                           "{:>21}\n", // round f32
                           "Idx", "x", "Label",
                           "__float2bfloat16",
                           "fp32TObf16Truncation",
                           "fp32TObf16Rounding"

  );

  std::cout << std::string(107, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float x = input[i];

    // Reference for error-flagging is BF16 converted back to FP32.
    float ref_f32 = __bfloat162float(output_ref[i]);

    OneResult16 v_trunc(output_ref[i], output_trunc[i], true, verbose);
    OneResult16 v_round(output_ref[i], output_round[i], true, verbose);

    bool allMatch = v_trunc.match and v_round.match;

    if (!quiet or !allMatch) {
      std::cout << std::format("{:>4}"
                               "{:>16g}"
                               "{:>24}"
                               "{:>21g}"
                               "{:>21}"
                               "{:>21}\n",
                               i, x, kCases[i].label, ref_f32, v_trunc.value(),
                               v_round.value());
    }

    if (verbose or !allMatch) {
      std::cout << std::format("{:>4}"
                               "{:>16}"
                               "{:>24}"
                               "{:>21}"
                               "{:>21}"
                               "{:>21}\n",
                               "",
                               fp32Hex(x),
                               "",
                               bf16Hex(output_ref[i]),
                               v_trunc.hexValue(),
                               v_round.hexValue()
                               );
    }

    if (!allMatch) {
      std::string es_trunc = v_trunc.errorString();
      std::string es_round = v_round.errorString();

      const char *color = YELLOW;
      if ((es_trunc == "ERROR") or (es_round == "ERROR")) {
        color = RED;
      }

      if (useColor) {
        std::cout << color;
      }

      std::cout << std::format("{:>4}"
                               "{:>16}"
                               "{:>24}"
                               "{:>21}"
                               "{:>21}"
                               "{:>21}\n",
                               "", "", "", "", es_trunc, es_round);

      if (useColor) {
        std::cout << RESET;
      }
    }
  }
}

int main(int argc, char **argv) {
  int c{};
  constexpr struct option longOptions[]{{"help", 0, nullptr, 'h'},
                                        {"verbose", 0, nullptr, 'v'},
                                        {"quiet", 0, nullptr, 'q'},
                                        {"color", 0, nullptr, 'c'},
                                        {nullptr, 0, nullptr, 0}};

  while (-1 != (c = getopt_long(argc, argv, "", longOptions, nullptr))) {
    switch (c) {
    case 'v': {
      verbose = true;
      break;
    }
    case 'q': {
      quiet = true;
      break;
    }
    case 'c': {
      useColor = true;
      break;
    }
    case 'h': {
      std::cout
          << "fp32TObf16"
             " [--verbose]"
             " [--quiet]"
             " [--color]"
             "\n\n"
             "\t--verbose.  Use exact bit-match comparisons and always show "
             "hex subrows\n"
             "\t--quiet.    Suppress rows where all compared results match\n"
             "\t--color.    Highlight mismatch diagnostics\n"
             "\t--help.     Show this output and exit\n";
      return 0;
    }
    default: {
      std::cerr << "fp32TObf16: unknown option\n";
      return 1;
    }
    }
  }

  Fp32ToBf16Tester *tester;
  HIP_CHECK(hipMallocManaged(&tester, sizeof(Fp32ToBf16Tester)));

  new (tester) Fp32ToBf16Tester();

  dim3 blockSize(Fp32ToBf16Tester::N);
  dim3 gridSize(1);

  tester->reset();

  hipLaunchKernelGGL(HIP_KERNEL_NAME(Fp32ToBf16Tester::testKernelRef), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  hipLaunchKernelGGL(HIP_KERNEL_NAME(Fp32ToBf16Tester::testKernelTrunc),
                     gridSize, blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  hipLaunchKernelGGL(HIP_KERNEL_NAME(Fp32ToBf16Tester::testKernelRound),
                     gridSize, blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  tester->displayResults();

  HIP_CHECK(hipFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
