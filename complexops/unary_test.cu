//
// unary_test.cu
//
// Tests float32 unary operations (floor, trunc, ceil, abs, trig, exp, log, etc.)
// on NVIDIA H100 hardware, comparing CUDA library functions against custom
// inline-asm PTX implementations.
//
// Ported from Mi300xASMTestBench/complexops/unary_test.hip to CUDA.
//
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <getopt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

#include "OneResult32.hpp"
#include "colors.hpp"
#include "cuda_check.hpp"
#include "readbinary.hpp"
#include "custom_asm.hpp"

//////////////////////////////////////////////////////
//
// Helper functions for reference
//

// Compile-time bit-cast for test case constants (C++20)
constexpr float uint_as_float_constexpr(uint32_t u) {
  return std::bit_cast<float>(u);
}

// Device-side bit-cast helpers (using PTX mov.b32)
__device__ inline float uint_as_float(uint32_t u) {
  float f;
  asm("mov.b32 %0, %1;" : "=f"(f) : "r"(u));
  return f;
}

__device__ inline uint32_t float_as_uint(float f) {
  uint32_t u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "f"(f));
  return u;
}

//////////////////////////////////////////////////////

enum class UnaryOp {
  Unknown, Floor, Trunc, Ceil, Abs, Sin, Cos, Exp2f, Expf, Rcp, Sqrtf, Rsqrtf,
  Log2f, Logf, Log10f, Erff, Erfcf, Tanf, Tanhf, Cbrtf, Expm1f, Log1pf,
  Acoshf, Asinhf, Atanhf, Acosf, Asinf, Atanf,
  Roundf, Nearbyintf, Sinhf, Coshf, Exp10f, Logbf,
  Lgammaf, Tgammaf, Sinpif, Cospif, Tanpif
};

static const char *opName(UnaryOp op) {
  switch (op) {
  case UnaryOp::Floor:
    return "floorf";
  case UnaryOp::Trunc:
    return "truncf";
  case UnaryOp::Ceil:
    return "ceilf";
  case UnaryOp::Abs:
    return "absf";
  case UnaryOp::Sin:
    return "sinf";
  case UnaryOp::Cos:
    return "cosf";
  case UnaryOp::Exp2f:
    return "exp2f";
  case UnaryOp::Expf:
    return "expf";
  case UnaryOp::Rcp:
    return "reciprocal";
  case UnaryOp::Sqrtf:
    return "sqrtf";
  case UnaryOp::Rsqrtf:
    return "rsqrtf";
  case UnaryOp::Log2f:
    return "log2f";
  case UnaryOp::Logf:
    return "logf";
  case UnaryOp::Log10f:
    return "log10f";
  case UnaryOp::Erff:
    return "erff";
  case UnaryOp::Erfcf:
    return "erfcf";
  case UnaryOp::Tanf:
    return "tanf";
  case UnaryOp::Tanhf:
    return "tanhf";
  case UnaryOp::Cbrtf:
    return "cbrtf";
  case UnaryOp::Expm1f:
    return "expm1f";
  case UnaryOp::Log1pf:
    return "log1pf";
  case UnaryOp::Acoshf:
    return "acoshf";
  case UnaryOp::Asinhf:
    return "asinhf";
  case UnaryOp::Atanhf:
    return "atanhf";
  case UnaryOp::Acosf:
    return "acosf";
  case UnaryOp::Asinf:
    return "asinf";
  case UnaryOp::Atanf:
    return "atanf";
  case UnaryOp::Roundf:
    return "roundf";
  case UnaryOp::Nearbyintf:
    return "nearbyintf";
  case UnaryOp::Sinhf:
    return "sinhf";
  case UnaryOp::Coshf:
    return "coshf";
  case UnaryOp::Exp10f:
    return "exp10f";
  case UnaryOp::Logbf:
    return "logbf";
  case UnaryOp::Lgammaf:
    return "lgammaf";
  case UnaryOp::Tgammaf:
    return "tgammaf";
  case UnaryOp::Sinpif:
    return "sinpif";
  case UnaryOp::Cospif:
    return "cospif";
  case UnaryOp::Tanpif:
    return "tanpif";
  case UnaryOp::Unknown:
    return "unknown";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------
struct UnaryCase {
  float x;
  const char *label;
};

static constexpr float pi = std::numbers::pi_v<float>;
static constexpr float pi_4 = pi / 4.0f;
static constexpr float sqrt2_2 = 0.70710678118654752f;
static constexpr float sqrt3_2 = 0.86602540378443865f;
static constexpr float sqrt3   = 1.73205080756887729f;
static constexpr float inv_sqrt3 = 0.57735026918962576f;

static constexpr UnaryCase kCases[] = {
    {0.0f, "+0"}, {-0.0f, "-0"}, {1.0f, "1"}, {-1.0f, "-1"}, {2.0f, "2"},
    {-2.0f, "-2"}, {3.0f, "3"}, {-3.0f, "-3"}, {4.0f, "4"}, {-4.0f, "-4"},
    {5.0f, "5"}, {-5.0f, "-5"}, {7.0f, "7"}, {9.0f, "9"}, {10.0f, "10"},
    {-10.0f, "-10"}, {16.0f, "16"}, {25.0f, "25"}, {49.0f, "49"},
    {64.0f, "64"}, {100.0f, "100"}, {-100.0f, "-100"}, {127.0f, "127"},
    {128.0f, "128"}, {255.0f, "255"}, {256.0f, "256"}, {-256.0f, "-256"},
    {1000.0f, "1000"}, {-1000.0f, "-1000"}, {10000.0f, "10000"},
    {8.0f, "8 (2^3)"}, {-8.0f, "-8 (-2^3)"}, {27.0f, "27 (3^3)"},
    {-27.0f, "-27 (-3^3)"}, {125.0f, "125 (5^3)"}, {-125.0f, "-125 (-5^3)"},
    {0.125f, "0.125 (0.5^3)"}, {-0.125f, "-0.125 (-0.5^3)"}, {0.5f, "0.5"},
    {-0.5f, "-0.5"}, {1.5f, "1.5"}, {-1.5f, "-1.5"}, {2.5f, "2.5"},
    {-2.5f, "-2.5"}, {3.5f, "3.5"}, {-3.5f, "-3.5"}, {7.5f, "7.5"},
    {uint_as_float_constexpr(0x3effffffu), "0.5-ulp"},
    {uint_as_float_constexpr(0x3f000001u), "0.5+ulp"},
    {uint_as_float_constexpr(0xbeffffffu), "-0.5+ulp"},
    {uint_as_float_constexpr(0xbf000001u), "-0.5-ulp"},
    {0.1f, "0.1"}, {-0.1f, "-0.1"}, {0.25f, "0.25"}, {0.9f, "0.9"},
    {-0.9f, "-0.9"}, {0.99f, "0.99"}, {0.999f, "0.999"},
    {-0.999f, "-0.999"}, {0.9999f, "0.9999"}, {-0.9999f, "-0.9999"},
    {0.367879441f, "1/e"}, {1.01f, "1.01"}, {1.1f, "1.1"},
    {0.0625f, "0.0625"}, {100.7f, "100.7"}, {-100.7f, "-100.7"},
    {0.01f, "0.01"}, {-0.01f, "-0.01"}, {0.001f, "0.001"},
    {0.75f, "3/4"}, {-0.75f, "-3/4"}, {1.25f, "5/4"},
    {-1.25f, "-5/4"}, {1.75f, "7/4"}, {-1.75f, "-7/4"}, {2.25f, "9/4"},
    {-2.25f, "-9/4"},
    {1.5f - 1e-6f, "3/2 - 1e-6 (tanpi pole nbr)"},
    {1.5f + 1e-6f, "3/2 + 1e-6 (tanpi pole nbr)"},
    {-1.5f - 1e-6f, "-3/2 - 1e-6 (tanpi pole nbr)"},
    {-1.5f + 1e-6f, "-3/2 + 1e-6 (tanpi pole nbr)"},
    {3.14159265f, "pi"}, {-3.14159265f, "-pi"}, {2.71828183f, "e"},
    {-2.71828183f, "-e"}, {0.693147180559945309417f, "ln(2)"},
    {3.140625f, "3.140625"}, {pi_4, "pi/4"}, {2.0f * pi_4, "pi/2"},
    {3.0f * pi_4, "3pi/4"}, {4.0f * pi_4, "pi (4pi/4)"},
    {5.0f * pi_4, "5pi/4"}, {6.0f * pi_4, "3pi/2"}, {7.0f * pi_4, "7pi/4"},
    {8.0f * pi_4, "2pi"}, {-pi_4, "-pi/4"}, {-2.0f * pi_4, "-pi/2"},
    {-4.0f * pi_4, "-pi (4pi/4)"}, {pi / 6.0f, "pi/6"},
    {pi / 3.0f, "pi/3"}, {2.0f * pi / 3.0f, "2pi/3"},
    {5.0f * pi / 6.0f, "5pi/6"}, {-pi / 6.0f, "-pi/6"},
    {-pi / 3.0f, "-pi/3"}, {pi_4 - 0.001f, "pi/4 - 0.001"},
    {pi_4 + 0.001f, "pi/4 + 0.001"},
    {2.0f * pi_4 - 0.001f, "pi/2 - 0.001"},
    {2.0f * pi_4 + 0.001f, "pi/2 + 0.001"},
    {4.0f * pi_4 - 0.001f, "pi - 0.001"},
    {4.0f * pi_4 + 0.001f, "pi + 0.001"},
    {6.0f * pi_4 - 0.001f, "3pi/2 - 0.001"},
    {6.0f * pi_4 + 0.001f, "3pi/2 + 0.001"},
    {2.0f * pi_4 - 1e-7f, "pi/2 - 1e-7"},
    {2.0f * pi_4 + 1e-7f, "pi/2 + 1e-7"},
    {4.0f * pi_4 - 1e-7f, "pi - 1e-7"},
    {4.0f * pi_4 + 1e-7f, "pi + 1e-7"},
    {6.0f * pi_4 - 1e-7f, "3pi/2 - 1e-7"},
    {6.0f * pi_4 + 1e-7f, "3pi/2 + 1e-7"},
    {8.0f * pi_4 - 1e-7f, "2pi - 1e-7"},
    {8.0f * pi_4 + 1e-7f, "2pi + 1e-7"},
    {uint_as_float_constexpr(0x3f800001u), "1+ulp"},
    {uint_as_float_constexpr(0xbf800001u), "-1-ulp"},
    {uint_as_float_constexpr(0x3f7fffffu), "1-ulp"},
    {uint_as_float_constexpr(0xbf7fffffu), "-1+ulp"},
    {1.0f + 1e-7f, "1 + 1e-7"}, {1.0f - 1e-7f, "1 - 1e-7"},
    {1.0f + 1e-5f, "1 + 1e-5"}, {1.0f - 1e-5f, "1 - 1e-5"},
    {sqrt2_2, "1/sqrt(2)"}, {-sqrt2_2, "-1/sqrt(2)"},
    {sqrt2_2 - 1e-5f, "1/sqrt(2) - 1e-5"},
    {sqrt2_2 + 1e-5f, "1/sqrt(2) + 1e-5"},
    {sqrt2_2 - 1e-7f, "1/sqrt(2) - 1e-7"},
    {sqrt2_2 + 1e-7f, "1/sqrt(2) + 1e-7"},
    {sqrt3_2, "sqrt(3)/2"}, {-sqrt3_2, "-sqrt(3)/2"},
    {sqrt3_2 - 1e-5f, "sqrt(3)/2 - 1e-5"},
    {sqrt3_2 + 1e-5f, "sqrt(3)/2 + 1e-5"},
    {sqrt3_2 - 1e-7f, "sqrt(3)/2 - 1e-7"},
    {sqrt3_2 + 1e-7f, "sqrt(3)/2 + 1e-7"},
    {inv_sqrt3, "1/sqrt(3)"}, {-inv_sqrt3, "-1/sqrt(3)"}, {sqrt3, "sqrt(3)"},
    {-sqrt3, "-sqrt(3)"}, {0.5f - 1e-5f, "0.5 - 1e-5"},
    {0.5f + 1e-5f, "0.5 + 1e-5"}, {0.5f - 1e-7f, "0.5 - 1e-7"},
    {0.5f + 1e-7f, "0.5 + 1e-7"}, {-0.5f - 1e-5f, "-0.5 - 1e-5"},
    {-0.5f + 1e-5f, "-0.5 + 1e-5"},
    {pi_4 - 1e-6f, "pi/4 - 1e-6"}, {pi_4 + 1e-6f, "pi/4 + 1e-6"},
    {2.0f * pi_4 - 1e-4f, "pi/2 - 1e-4"},
    {2.0f * pi_4 + 1e-4f, "pi/2 + 1e-4"},
    {2.0f * pi_4 - 1e-6f, "pi/2 - 1e-6"},
    {2.0f * pi_4 + 1e-6f, "pi/2 + 1e-6"},
    {0.99999f, "0.99999"}, {-0.99999f, "-0.99999"},
    {1.0f - 1e-6f, "1 - 1e-6"}, {-(1.0f - 1e-6f), "-(1 - 1e-6)"},
    {1.71828183f, "e-1"}, {-1.71828183f, "-(e-1)"},
    {-1.0f + 1e-3f, "-1 + 1e-3"}, {-1.0f + 1e-5f, "-1 + 1e-5"},
    {-1.0f + 1e-6f, "-1 + 1e-6"}, {-1.0f + 1e-7f, "-1 + 1e-7"},
    {-2.0f + 1e-6f, "-2 + 1e-6 (gamma pole nbr)"},
    {-2.0f - 1e-6f, "-2 - 1e-6 (gamma pole nbr)"},
    {-3.0f + 1e-6f, "-3 + 1e-6 (gamma pole nbr)"},
    {-3.0f - 1e-6f, "-3 - 1e-6 (gamma pole nbr)"},
    {-4.0f + 1e-6f, "-4 + 1e-6 (gamma pole nbr)"},
    {-4.0f - 1e-6f, "-4 - 1e-6 (gamma pole nbr)"},
    {1.0f + 1e-3f, "1 + 1e-3"}, {1.0f + 1e-4f, "1 + 1e-4"},
    {1.543080635f, "cosh(1)"}, {0.4769362762f, "erf=0.5 (x≈0.4769)"},
    {-0.4769362762f, "-erf=0.5"},
    {0.4769362762f - 1e-5f, "erf=0.5 - 1e-5"},
    {0.4769362762f + 1e-5f, "erf=0.5 + 1e-5"}, {6.0f, "6 (erf≈1)"},
    {-6.0f, "-6 (erfc≈2)"}, {10.0f + 1e-1f, "10.1 (erfc underflow)"},
    {0.54930614f, "tanh⁻¹(0.5)"}, {-0.54930614f, "-tanh⁻¹(0.5)"},
    {4.0f - 1e-3f, "4 - 1e-3"}, {4.0f + 1e-3f, "4 + 1e-3"},
    {8.0f, "8 (tanh→1)"}, {-8.0f, "-8 (tanh→-1)"},
    {20.0f, "20 (tanh=1 exact)"}, {1e-2f, "1e-2 (cbrt)"},
    {-1e-2f, "-1e-2 (cbrt)"}, {1e-4f, "1e-4 (cbrt)"},
    {-1e-4f, "-1e-4 (cbrt)"}, {2.30258509f, "ln(10)"},
    {-2.30258509f, "-ln(10)"},
    {uint_as_float_constexpr(0x3f808000u), "bf16 tie"},
    {uint_as_float_constexpr(0x3f808001u), "bf16 tie+eps"},
    {uint_as_float_constexpr(0x3f818000u), "bf16 round up"},
    {uint_as_float_constexpr(0x3f810000u), "bf16 exact"},
    {88.0f, "88"}, {-88.0f, "-88"}, {88.72f, "88.72"}, {88.73f, "88.73"},
    {89.0f, "89"}, {-87.0f, "-87"}, {-87.34f, "-87.34"}, {-104.0f, "-104"},
    {-125.0f, "-125"}, {-126.0f, "-126"}, {126.0f, "126"}, {-149.0f, "-149"},
    {-150.0f, "-150"}, {-200.0f, "-200"}, {-500.0f, "-500"},
    {0.30103f, "log10(2)"}, {-0.30103f, "-log10(2)"}, {0.69897f, "log10(5)"},
    {-0.69897f, "-log10(5)"}, {38.53f, "38.53 (exp10 near max)"},
    {38.54f, "38.54 (exp10 overflow edge)"},
    {-44.8f, "-44.8 (exp10 subnorm edge)"},
    {-45.0f, "-45 (exp10 underflow edge)"}, {16777216.0f, "2^24"},
    {-16777216.0f, "-2^24"}, {16777216.5f, "2^24+0.5"}, {16777217.0f, "2^24+1"},
    {uint_as_float_constexpr(0x4b800001u), "2^24+2"},
    {1e4f, "1e4"}, {1e6f, "1e6"}, {-1e6f, "-1e6"}, {1e10f, "1e10"},
    {-1e10f, "-1e10"}, {1e11f, "1e11"}, {1e12f, "1e12"}, {-1e12f, "-1e12"},
    {1e13f, "1e13"}, {1e14f, "1e14"}, {-1e14f, "-1e14"}, {1e15f, "1e15"},
    {1e16f, "1e16"}, {-1e16f, "-1e16"}, {1e17f, "1e17"}, {1e18f, "1e18"},
    {-1e18f, "-1e18"}, {1e19f, "1e19"}, {1e20f, "1e20"}, {-1e20f, "-1e20"},
    {1e22f, "1e22"}, {-1e22f, "-1e22"}, {1e25f, "1e25"}, {-1e25f, "-1e25"},
    {1e28f, "1e28"}, {1e30f, "1e30"}, {-1e30f, "-1e30"}, {1e33f, "1e33"},
    {1e35f, "1e35"}, {-1e35f, "-1e35"}, {1e37f, "1e37"}, {-1e37f, "-1e37"},
    {1e38f, "1e38"}, {-1e38f, "-1e38"}, {3.4e38f, "3.4e38"},
    {pi_4 + 100.0f * 2.0f * pi, "pi/4 + 200pi"},
    {pi_4 + 1000.0f * 2.0f * pi, "pi/4 + 2000pi"},
    {pi_4 + 10000.0f * 2.0f * pi, "pi/4 + 20000pi"},
    {2.0f * pi_4 + 100.0f * 2.0f * pi, "pi/2 + 200pi"},
    {2.0f * pi_4 + 1000.0f * 2.0f * pi, "pi/2 + 2000pi"},
    {4.0f * pi_4 + 100.0f * 2.0f * pi, "pi + 200pi"},
    {4.0f * pi_4 + 1000.0f * 2.0f * pi, "pi + 2000pi"},
    {pi_4 - 100.0f * 2.0f * pi, "pi/4 - 200pi"},
    {pi_4 - 1000.0f * 2.0f * pi, "pi/4 - 2000pi"},
    {pi_4 - 10000.0f * 2.0f * pi, "pi/4 - 20000pi"},
    {2.0f * pi_4 - 100.0f * 2.0f * pi, "pi/2 - 200pi"},
    {2.0f * pi_4 - 1000.0f * 2.0f * pi, "pi/2 - 2000pi"},
    {4.0f * pi_4 - 100.0f * 2.0f * pi, "pi - 200pi"},
    {4.0f * pi_4 - 1000.0f * 2.0f * pi, "pi - 2000pi"},
    {1e-3f, "1e-3"}, {-1e-3f, "-1e-3"}, {1e-5f, "1e-5"}, {-1e-5f, "-1e-5"},
    {1e-6f, "1e-6"}, {-1e-6f, "-1e-6"}, {1e-7f, "1e-7"}, {-1e-7f, "-1e-7"},
    {1e-10f, "1e-10"}, {1e-20f, "1e-20"}, {-1e-20f, "-1e-20"},
    {1e-30f, "1e-30"}, {1e-38f, "1e-38"}, {-1e-38f, "-1e-38"},
    {1e-40f, "1e-40"}, {std::numeric_limits<float>::denorm_min(), "denorm_min"},
    {-std::numeric_limits<float>::denorm_min(), "-denorm_min"},
    {5.877472e-39f, "mid subnorm"},
    {uint_as_float_constexpr(0x007fffffu), "max subnorm"},
    {uint_as_float_constexpr(0x807fffffu), "-max subnorm"},
    {std::numeric_limits<float>::min(), "FLT_MIN"},
    {-std::numeric_limits<float>::min(), "-FLT_MIN"},
    {std::numeric_limits<float>::max(), "FLT_MAX"},
    {-std::numeric_limits<float>::max(), "-FLT_MAX"},
    {std::numeric_limits<float>::infinity(), "+inf"},
    {-std::numeric_limits<float>::infinity(), "-inf"},
    {std::numeric_limits<float>::quiet_NaN(), "qNaN"},
    {-std::numeric_limits<float>::quiet_NaN(), "-qNaN"},
    {std::numeric_limits<float>::signaling_NaN(), "sNaN"},
    {-std::numeric_limits<float>::signaling_NaN(), "-sNaN"},
    {uint_as_float_constexpr(0x7fc00001u), "qNaN payload"},
    {uint_as_float_constexpr(0x7fffffffu), "qNaN max payload"},
    {uint_as_float_constexpr(0x7f800001u), "sNaN min payload"},
    {uint_as_float_constexpr(0x7fbfffffu), "sNaN max payload"},
    {uint_as_float_constexpr(0xff800001u), "-sNaN min payload"},
    {uint_as_float_constexpr(0xffbfffffu), "-sNaN max payload"},
};
static constexpr size_t kNumUnaryCases = sizeof(kCases) / sizeof(kCases[0]);

// ---------------------------------------------------------------------------
// Tester class
// ---------------------------------------------------------------------------
class UnaryTester {
public:
  static constexpr size_t N = kNumUnaryCases;

  float input[N];
  float out_cuda[N];
  float out_custom[N];

  __host__ UnaryTester() {
    for (size_t i = 0; i < N; i++) {
      input[i] = kCases[i].x;
    }
  }

  void __host__ reset() {
    std::memset(out_cuda, 0xff, sizeof(out_cuda));
    std::memset(out_custom, 0xff, sizeof(out_custom));
  }
};

bool verbose{};
bool useColor{};
bool quiet{};
bool csvOutput{};

// ---------------------------------------------------------------------------
// CUDA Kernels (free functions, not class members for CUDA compatibility)
// ---------------------------------------------------------------------------

#define KERNEL_CUDA(func) __global__ void test##func##Cuda(UnaryTester *self) { \
  size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < UnaryTester::N) self->out_cuda[i] = func(self->input[i]); \
}

#define KERNEL_CUSTOM(func) __global__ void test##func##Custom(UnaryTester *self) { \
  size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < UnaryTester::N) self->out_custom[i] = custom_##func(self->input[i]); \
}

KERNEL_CUDA(floorf)
KERNEL_CUDA(truncf)
KERNEL_CUDA(ceilf)
KERNEL_CUDA(fabsf)
KERNEL_CUDA(sinf)
KERNEL_CUDA(cosf)
KERNEL_CUDA(exp2f)
KERNEL_CUDA(expf)
KERNEL_CUDA(sqrtf)
KERNEL_CUDA(rsqrtf)
KERNEL_CUDA(log2f)
KERNEL_CUDA(logf)
KERNEL_CUDA(log10f)
KERNEL_CUDA(erff)
KERNEL_CUDA(erfcf)
KERNEL_CUDA(tanf)
KERNEL_CUDA(tanhf)
KERNEL_CUDA(cbrtf)
KERNEL_CUDA(expm1f)
KERNEL_CUDA(log1pf)
KERNEL_CUDA(acoshf)
KERNEL_CUDA(asinhf)
KERNEL_CUDA(atanhf)
KERNEL_CUDA(acosf)
KERNEL_CUDA(asinf)
KERNEL_CUDA(atanf)
KERNEL_CUDA(roundf)
KERNEL_CUDA(nearbyintf)
KERNEL_CUDA(sinhf)
KERNEL_CUDA(coshf)
KERNEL_CUDA(exp10f)
KERNEL_CUDA(logbf)
KERNEL_CUDA(lgammaf)
KERNEL_CUDA(tgammaf)

__global__ void testRcpCuda(UnaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < UnaryTester::N) self->out_cuda[i] = 1.0f / self->input[i];
}

__global__ void testSinpifCuda(UnaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < UnaryTester::N) self->out_cuda[i] = sinpif(self->input[i]);
}

__global__ void testCospifCuda(UnaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < UnaryTester::N) self->out_cuda[i] = cospif(self->input[i]);
}

__global__ void testTanpifCuda(UnaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < UnaryTester::N) self->out_cuda[i] = tanf(3.14159265358979323846f * self->input[i]);
}

// Custom kernels
KERNEL_CUSTOM(floorf)
KERNEL_CUSTOM(truncf)
KERNEL_CUSTOM(ceilf)
KERNEL_CUSTOM(absf)
KERNEL_CUSTOM(sinf)
KERNEL_CUSTOM(cosf)
KERNEL_CUSTOM(exp2f)
KERNEL_CUSTOM(expf)
KERNEL_CUSTOM(sqrtf)
KERNEL_CUSTOM(rsqrtf)
KERNEL_CUSTOM(log2f)
KERNEL_CUSTOM(logf)
KERNEL_CUSTOM(log10f)
KERNEL_CUSTOM(erff)
KERNEL_CUSTOM(erfcf)
KERNEL_CUSTOM(tanf)
KERNEL_CUSTOM(tanhf)
KERNEL_CUSTOM(cbrtf)
KERNEL_CUSTOM(expm1f)
KERNEL_CUSTOM(log1pf)
KERNEL_CUSTOM(acoshf)
KERNEL_CUSTOM(asinhf)
KERNEL_CUSTOM(atanhf)
KERNEL_CUSTOM(acosf)
KERNEL_CUSTOM(asinf)
KERNEL_CUSTOM(atanf)
KERNEL_CUSTOM(roundf)
KERNEL_CUSTOM(nearbyintf)
KERNEL_CUSTOM(sinhf)
KERNEL_CUSTOM(coshf)
KERNEL_CUSTOM(exp10f)
KERNEL_CUSTOM(logbf)
KERNEL_CUSTOM(lgammaf)
KERNEL_CUSTOM(tgammaf)
KERNEL_CUSTOM(sinpif)
KERNEL_CUSTOM(cospif)
KERNEL_CUSTOM(tanpif)

__global__ void testRcpCustom(UnaryTester *self) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < UnaryTester::N) self->out_custom[i] = custom_rcp(self->input[i]);
}

// ---------------------------------------------------------------------------
// displayResults
// ---------------------------------------------------------------------------
static std::string fp32Hex(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return std::format("0x{:08x}", bits);
}

static void displayResults(const UnaryTester &t, UnaryOp op, const float *torcheager, const float *torchinductor) {
  const char *name = opName(op);

  if (csvOutput) {
    std::cout << std::format("op,idx,x,label,row,{0}(cuda),{0}(custom),torch_eager,torch_inductor\n", name);
    for (size_t i = 0; i < UnaryTester::N; i++) {
      float x = t.input[i];
      float ref = t.out_cuda[i];
      OneResult32 v_custom(ref, t.out_custom[i], true, verbose);
      OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f, torcheager != nullptr, verbose);
      OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f, torchinductor != nullptr, verbose);
      bool allMatch = v_custom.match and v_eager.match and v_inductor.match;
      if (!quiet or !allMatch) {
        std::cout << std::format("{},{},{:g},\"{}\",VALUE,{:g},{:g},{},{}\n", name, i, x, kCases[i].label, ref,
                                 t.out_custom[i], torcheager ? std::format("{:g}", torcheager[i]) : "",
                                 torchinductor ? std::format("{:g}", torchinductor[i]) : "");
      }
      if (!allMatch) {
        std::cout << std::format("{},{},,\"\",HEX,{},{},{},{}\n", name, i, fp32Hex(ref), v_custom.hexValue(),
                                 v_eager.hexValue(), v_inductor.hexValue());
        std::cout << std::format("{},{},,\"\",DIFF,,{},{},{}\n", name, i, v_custom.errorString(), v_eager.errorString(),
                                 v_inductor.errorString());
      }
    }
    return;
  }

  if (useColor) std::cout << RED;
  std::cout << std::format("UNARY OP: {}\n\n", name);
  if (useColor) std::cout << RESET;

  std::cout << std::format("{:>4}{:>16}{:>30}{:>16}{:>16}{:>16}{:>16}\n", "Idx", "x", "Label",
                           std::format("{}(cuda)", name), std::format("{}(custom)", name),
                           torcheager ? "torch-eager" : "", torchinductor ? "torch-inductor" : "");
  std::cout << std::string(114, '-') << "\n";

  for (size_t i = 0; i < UnaryTester::N; i++) {
    float x = t.input[i];
    float ref = t.out_cuda[i];
    OneResult32 v_custom(ref, t.out_custom[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f, torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f, torchinductor != nullptr, verbose);
    bool allMatch = v_custom.match and v_eager.match and v_inductor.match;
    if (!quiet or !allMatch) {
      std::cout << std::format("{:>4}{:>16g}{:>30}{:>16.6g}{}{}{}\n", i, x, kCases[i].label, ref, v_custom.value(),
                               v_eager.value(), v_inductor.value());
    }
    if (verbose or !allMatch) {
      std::cout << std::format("{:>4}{:>16}{:>30}{:>16}{:>16}{:>16}{:>16}\n", "", fp32Hex(x), "", fp32Hex(ref),
                               v_custom.hexValue(), v_eager.hexValue(), v_inductor.hexValue());
    }
    if (!allMatch) {
      std::string es_custom = v_custom.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();
      const char *color = YELLOW;
      if ((es_custom == "ERROR") or (es_eager == "ERROR") or (es_inductor == "ERROR")) {
        color = RED;
      }
      if (useColor) std::cout << color;
      std::cout << std::format("{:>4}{:>16}{:>30}{:>16}{:>16}{:>16}{:>16}\n", "", "", "", "", es_custom, es_eager,
                               es_inductor);
      if (useColor) std::cout << RESET;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel dispatch table
// ---------------------------------------------------------------------------
using KernelFn = void (*)(UnaryTester *);

struct KernelPair {
  KernelFn cuda;
  KernelFn custom;
};

static KernelPair kernelsForOp(UnaryOp op) {
  switch (op) {
  case UnaryOp::Floor:
    return {testfloorfCuda, testfloorfCustom};
  case UnaryOp::Trunc:
    return {testtruncfCuda, testtruncfCustom};
  case UnaryOp::Ceil:
    return {testceilfCuda, testceilfCustom};
  case UnaryOp::Abs:
    return {testfabsfCuda, testabsfCustom};
  case UnaryOp::Sin:
    return {testsinfCuda, testsinfCustom};
  case UnaryOp::Cos:
    return {testcosfCuda, testcosfCustom};
  case UnaryOp::Exp2f:
    return {testexp2fCuda, testexp2fCustom};
  case UnaryOp::Expf:
    return {testexpfCuda, testexpfCustom};
  case UnaryOp::Rcp:
    return {testRcpCuda, testRcpCustom};
  case UnaryOp::Sqrtf:
    return {testsqrtfCuda, testsqrtfCustom};
  case UnaryOp::Rsqrtf:
    return {testrsqrtfCuda, testrsqrtfCustom};
  case UnaryOp::Log2f:
    return {testlog2fCuda, testlog2fCustom};
  case UnaryOp::Logf:
    return {testlogfCuda, testlogfCustom};
  case UnaryOp::Log10f:
    return {testlog10fCuda, testlog10fCustom};
  case UnaryOp::Erff:
    return {testerffCuda, testerffCustom};
  case UnaryOp::Erfcf:
    return {testerfcfCuda, testerfcfCustom};
  case UnaryOp::Tanf:
    return {testtanfCuda, testtanfCustom};
  case UnaryOp::Tanhf:
    return {testtanhfCuda, testtanhfCustom};
  case UnaryOp::Cbrtf:
    return {testcbrtfCuda, testcbrtfCustom};
  case UnaryOp::Expm1f:
    return {testexpm1fCuda, testexpm1fCustom};
  case UnaryOp::Log1pf:
    return {testlog1pfCuda, testlog1pfCustom};
  case UnaryOp::Acoshf:
    return {testacoshfCuda, testacoshfCustom};
  case UnaryOp::Asinhf:
    return {testasinhfCuda, testasinhfCustom};
  case UnaryOp::Atanhf:
    return {testatanhfCuda, testatanhfCustom};
  case UnaryOp::Acosf:
    return {testacosfCuda, testacosfCustom};
  case UnaryOp::Asinf:
    return {testasinfCuda, testasinfCustom};
  case UnaryOp::Atanf:
    return {testatanfCuda, testatanfCustom};
  case UnaryOp::Roundf:
    return {testroundfCuda, testroundfCustom};
  case UnaryOp::Nearbyintf:
    return {testnearbyintfCuda, testnearbyintfCustom};
  case UnaryOp::Sinhf:
    return {testsinhfCuda, testsinhfCustom};
  case UnaryOp::Coshf:
    return {testcoshfCuda, testcoshfCustom};
  case UnaryOp::Exp10f:
    return {testexp10fCuda, testexp10fCustom};
  case UnaryOp::Logbf:
    return {testlogbfCuda, testlogbfCustom};
  case UnaryOp::Lgammaf:
    return {testlgammafCuda, testlgammafCustom};
  case UnaryOp::Tgammaf:
    return {testtgammafCuda, testtgammafCustom};
  case UnaryOp::Sinpif:
    return {testSinpifCuda, testsinpifCustom};
  case UnaryOp::Cospif:
    return {testCospifCuda, testcospifCustom};
  case UnaryOp::Tanpif:
    return {testTanpifCuda, testtanpifCustom};
  case UnaryOp::Unknown:
    return {nullptr, nullptr};
  }
  return {nullptr, nullptr};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  UnaryOp selectedOp = UnaryOp::Unknown;
  const char *dumpFile{};
  const char *torchinductorFile{};
  const char *torcheagerFile{};

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--floorf") selectedOp = UnaryOp::Floor;
    else if (arg == "--truncf") selectedOp = UnaryOp::Trunc;
    else if (arg == "--ceilf") selectedOp = UnaryOp::Ceil;
    else if (arg == "--absf") selectedOp = UnaryOp::Abs;
    else if (arg == "--sinf") selectedOp = UnaryOp::Sin;
    else if (arg == "--cosf") selectedOp = UnaryOp::Cos;
    else if (arg == "--exp2f") selectedOp = UnaryOp::Exp2f;
    else if (arg == "--expf") selectedOp = UnaryOp::Expf;
    else if (arg == "--reciprocal") selectedOp = UnaryOp::Rcp;
    else if (arg == "--sqrtf") selectedOp = UnaryOp::Sqrtf;
    else if (arg == "--rsqrtf") selectedOp = UnaryOp::Rsqrtf;
    else if (arg == "--log2f") selectedOp = UnaryOp::Log2f;
    else if (arg == "--logf") selectedOp = UnaryOp::Logf;
    else if (arg == "--log10f") selectedOp = UnaryOp::Log10f;
    else if (arg == "--erff") selectedOp = UnaryOp::Erff;
    else if (arg == "--erfcf") selectedOp = UnaryOp::Erfcf;
    else if (arg == "--tanf") selectedOp = UnaryOp::Tanf;
    else if (arg == "--tanhf") selectedOp = UnaryOp::Tanhf;
    else if (arg == "--cbrtf") selectedOp = UnaryOp::Cbrtf;
    else if (arg == "--expm1f") selectedOp = UnaryOp::Expm1f;
    else if (arg == "--log1pf") selectedOp = UnaryOp::Log1pf;
    else if (arg == "--acoshf") selectedOp = UnaryOp::Acoshf;
    else if (arg == "--asinhf") selectedOp = UnaryOp::Asinhf;
    else if (arg == "--atanhf") selectedOp = UnaryOp::Atanhf;
    else if (arg == "--acosf") selectedOp = UnaryOp::Acosf;
    else if (arg == "--asinf") selectedOp = UnaryOp::Asinf;
    else if (arg == "--atanf") selectedOp = UnaryOp::Atanf;
    else if (arg == "--roundf") selectedOp = UnaryOp::Roundf;
    else if (arg == "--nearbyintf") selectedOp = UnaryOp::Nearbyintf;
    else if (arg == "--sinhf") selectedOp = UnaryOp::Sinhf;
    else if (arg == "--coshf") selectedOp = UnaryOp::Coshf;
    else if (arg == "--exp10f") selectedOp = UnaryOp::Exp10f;
    else if (arg == "--logbf") selectedOp = UnaryOp::Logbf;
    else if (arg == "--lgammaf") selectedOp = UnaryOp::Lgammaf;
    else if (arg == "--tgammaf") selectedOp = UnaryOp::Tgammaf;
    else if (arg == "--sinpif") selectedOp = UnaryOp::Sinpif;
    else if (arg == "--cospif") selectedOp = UnaryOp::Cospif;
    else if (arg == "--tanpif") selectedOp = UnaryOp::Tanpif;
    else if (arg == "--verbose") verbose = true;
    else if (arg == "--quiet") quiet = true;
    else if (arg == "--color") useColor = true;
    else if (arg == "--csv") csvOutput = true;
    else if (arg == "--dump-inputs" && i + 1 < argc) dumpFile = argv[++i];
    else if (arg == "--torcheager" && i + 1 < argc) torcheagerFile = argv[++i];
    else if (arg == "--torchinductor" && i + 1 < argc) torchinductorFile = argv[++i];
    else if (arg == "--help") {
      std::cout << "unary_test --[op] [--verbose] [--quiet] [--color] [--csv]\n"
                   "            [--dump-inputs file] [--torcheager file] [--torchinductor file]\n";
      return 0;
    }
  }

  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << std::endl;
      return 2;
    }
    UnaryTester tmp;
    for (size_t i = 0; i < UnaryTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&tmp.input[i]), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input values to " << dumpFile << std::endl;
    return 0;
  }

  if (selectedOp == UnaryOp::Unknown) {
    std::cerr << "unary_test: --[op] is required\n";
    return 1;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, UnaryTester::N);
    if (torchinductorOut.empty()) torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, UnaryTester::N);
    if (torcheagerOut.empty()) torcheagerFile = nullptr;
  }

  UnaryTester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(UnaryTester)));
  new (tester) UnaryTester();

  dim3 blockSize(256);
  dim3 gridSize((UnaryTester::N + 255) / 256);

  tester->reset();

  auto [cudaKernel, customKernel] = kernelsForOp(selectedOp);

  cudaKernel<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());
  customKernel<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  displayResults(*tester, selectedOp, torcheagerFile ? torcheagerOut.data() : nullptr,
                 torchinductorFile ? torchinductorOut.data() : nullptr);

  CUDA_CHECK(cudaFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
