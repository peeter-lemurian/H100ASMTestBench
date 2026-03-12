//
// cossin_test.cu
//
// Tests cosf/sinf on NVIDIA H100 hardware, comparing:
//   1. CUDA cosf()/sinf() (library reference)
//   2. CUDA __cosf()/__sinf() (fast approximate)
//   3. Custom inline-asm wrappers (placeholder for future PTX)
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

// clang-format off
// ---------------------------------------------------------------------------
// Custom cos and sin: placeholders for future inline asm versions
// ---------------------------------------------------------------------------
inline float __device__ custom_cos_simple(float radians) {
  float result = cosf(radians);
  return result;
}

inline float __device__ custom_sin_simple(float radians) {
  float result = sinf(radians);
  return result;
}
// clang-format on

// ---------------------------------------------------------------------------
// Test-case table
// ---------------------------------------------------------------------------
struct TrigCase {
  float x;
  const char *label;
};

static constexpr float pi = std::numbers::pi_v<float>;
static constexpr float pi_4 = pi / 4.0f;

static constexpr TrigCase kCases[] = {
    // Multiples of π/4 in [0, 2π]
    {0.0f * pi_4, "0"},    //  0: cos(0) = 1
    {1.0f * pi_4, "π/4"},  //  1: cos(π/4) ≈ 0.707
    {2.0f * pi_4, "π/2"},  //  2: cos(π/2) = 0
    {3.0f * pi_4, "3π/4"}, //  3: cos(3π/4) ≈ -0.707
    {4.0f * pi_4, "π"},    //  4: cos(π) = -1
    {5.0f * pi_4, "5π/4"}, //  5: cos(5π/4) ≈ -0.707
    {6.0f * pi_4, "3π/2"}, //  6: cos(3π/2) = 0
    {7.0f * pi_4, "7π/4"}, //  7: cos(7π/4) ≈ 0.707
    {8.0f * pi_4, "2π"},   //  8: cos(2π) = 1

    // Near multiples of π/4
    {pi_4 - 0.001f, "π/4 - 0.001"},        //  9
    {pi_4 + 0.001f, "π/4 + 0.001"},        // 10
    {2.0f * pi_4 - 0.001f, "π/2 - 0.001"}, // 11
    {2.0f * pi_4 + 0.001f, "π/2 + 0.001"}, // 12
    {4.0f * pi_4 - 0.001f, "π - 0.001"},   // 13
    {4.0f * pi_4 + 0.001f, "π + 0.001"},   // 14

    // Large positive periods: π/4 + N*2π for large N
    {pi_4 + 100.0f * 2.0f * pi, "π/4 + 200π"},     // 15
    {pi_4 + 1000.0f * 2.0f * pi, "π/4 + 2000π"},   // 16
    {pi_4 + 10000.0f * 2.0f * pi, "π/4 + 20000π"}, // 17

    {2.0f * pi_4 + 100.0f * 2.0f * pi, "π/2 + 200π"},   // 18
    {2.0f * pi_4 + 1000.0f * 2.0f * pi, "π/2 + 2000π"}, // 19

    {4.0f * pi_4 + 100.0f * 2.0f * pi, "π + 200π"},   // 20
    {4.0f * pi_4 + 1000.0f * 2.0f * pi, "π + 2000π"}, // 21

    // Large negative periods: π/4 - N*2π for large N
    {pi_4 - 100.0f * 2.0f * pi, "π/4 - 200π"},     // 22
    {pi_4 - 1000.0f * 2.0f * pi, "π/4 - 2000π"},   // 23
    {pi_4 - 10000.0f * 2.0f * pi, "π/4 - 20000π"}, // 24

    {2.0f * pi_4 - 100.0f * 2.0f * pi, "π/2 - 200π"},   // 25
    {2.0f * pi_4 - 1000.0f * 2.0f * pi, "π/2 - 2000π"}, // 26

    {4.0f * pi_4 - 100.0f * 2.0f * pi, "π - 200π"},   // 27
    {4.0f * pi_4 - 1000.0f * 2.0f * pi, "π - 2000π"}, // 28

    // Additional special values
    {0.0f, "0 (exact zero)"},  // 29: cos(0) = 1
    {-pi_4, "-π/4"},           // 30: cos(-π/4) ≈ 0.707
    {-2.0f * pi_4, "-π/2"},    // 31: cos(-π/2) = 0
    {-4.0f * pi_4, "-π"},      // 32: cos(-π) = -1
    {1.0f, "1 radian"},        // 33
    {10.0f, "10 radians"},     // 34
    {100.0f, "100 radians"},   // 35
    {1000.0f, "1000 radians"}, // 36

    // Very small values
    {1e-6f, "1e-6"},   // 37: cos(≈0) ≈ 1
    {-1e-6f, "-1e-6"}, // 38: cos(≈0) ≈ 1

    // Special IEEE values
    {std::numeric_limits<float>::infinity(), "+inf"},  // 39: cos(+inf) = NaN
    {-std::numeric_limits<float>::infinity(), "-inf"}, // 40: cos(-inf) = NaN
    {std::numeric_limits<float>::quiet_NaN(), "NaN"},  // 41: cos(NaN) = NaN

    // -----------------------------------------------------------------------
    // Negative zero
    // -----------------------------------------------------------------------
    {-0.0f, "-0"}, // 42: cos(-0) = 1, sin(-0) = -0

    // -----------------------------------------------------------------------
    // Negative NaN and signaling NaN
    // -----------------------------------------------------------------------
    {-std::numeric_limits<float>::quiet_NaN(), "-NaN"},    // 43
    {std::numeric_limits<float>::signaling_NaN(), "sNaN"}, // 44

    // -----------------------------------------------------------------------
    // Subnormal (denormalized) inputs
    // -----------------------------------------------------------------------
    {1.401298e-45f, "min subnorm"},    // 45: smallest positive subnormal
    {5.877472e-39f, "mid subnorm"},    // 46
    {1.1754942e-38f, "max subnorm"},   // 47: largest subnormal
    {-1.401298e-45f, "-min subnorm"},  // 48
    {-1.1754942e-38f, "-max subnorm"}, // 49

    // -----------------------------------------------------------------------
    // FLT_MIN (smallest normal) and FLT_MAX (range-reduction stress)
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::min(), "FLT_MIN"},   // 50
    {-std::numeric_limits<float>::min(), "-FLT_MIN"}, // 51
    {std::numeric_limits<float>::max(), "FLT_MAX"},   // 52
    {-std::numeric_limits<float>::max(), "-FLT_MAX"}, // 53

    // -----------------------------------------------------------------------
    // Very small neighbourhood (sin(x) ≈ x, cos(x) ≈ 1)
    // -----------------------------------------------------------------------
    {1e-7f, "1e-7"},     // 54
    {-1e-7f, "-1e-7"},   // 55
    {1e-20f, "1e-20"},   // 56
    {-1e-20f, "-1e-20"}, // 57
    {1e-38f, "1e-38"},   // 58
    {-1e-38f, "-1e-38"}, // 59

    // -----------------------------------------------------------------------
    // Tight neighbourhood of π/2 (cos ≈ 0)
    // -----------------------------------------------------------------------
    {2.0f * pi_4 - 1e-7f, "π/2 - 1e-7"}, // 60
    {2.0f * pi_4 + 1e-7f, "π/2 + 1e-7"}, // 61

    // Tight neighbourhood of π (sin ≈ 0)
    {4.0f * pi_4 - 1e-7f, "π - 1e-7"}, // 62
    {4.0f * pi_4 + 1e-7f, "π + 1e-7"}, // 63

    // Tight neighbourhood of 3π/2 (cos ≈ 0)
    {6.0f * pi_4 - 0.001f, "3π/2 - 0.001"}, // 64
    {6.0f * pi_4 + 0.001f, "3π/2 + 0.001"}, // 65
    {6.0f * pi_4 - 1e-7f, "3π/2 - 1e-7"},   // 66
    {6.0f * pi_4 + 1e-7f, "3π/2 + 1e-7"},   // 67

    // Tight neighbourhood of 2π (sin ≈ 0)
    {8.0f * pi_4 - 1e-7f, "2π - 1e-7"}, // 68
    {8.0f * pi_4 + 1e-7f, "2π + 1e-7"}, // 69

    // Tight neighbourhood of 0 (sin ≈ x)
    {1e-7f, "0 + 1e-7"},  // 70  (same as 54, intentional anchor)
    {-1e-7f, "0 - 1e-7"}, // 71  (same as 55, intentional anchor)

    // -----------------------------------------------------------------------
    // Multiples of π/6 and π/3 (well-known exact values)
    // sin(π/6) = 0.5, cos(π/3) = 0.5
    // -----------------------------------------------------------------------
    {pi / 6.0f, "π/6"},         // 72
    {pi / 3.0f, "π/3"},         // 73
    {2.0f * pi / 3.0f, "2π/3"}, // 74
    {5.0f * pi / 6.0f, "5π/6"}, // 75
    {-pi / 6.0f, "-π/6"},       // 76
    {-pi / 3.0f, "-π/3"},       // 77

    // -----------------------------------------------------------------------
    // Extreme large angles (progressive range-reduction stress)
    // -----------------------------------------------------------------------
    {1e4f, "1e4"},     // 78
    {1e6f, "1e6"},     // 79
    {1e10f, "1e10"},   // 80
    {1e20f, "1e20"},   // 81
    {-1e6f, "-1e6"},   // 82
    {-1e10f, "-1e10"}, // 83
    {-1e20f, "-1e20"}, // 84

    // -----------------------------------------------------------------------
    // Progressive large-angle stress: filling 2^33 → 2^127
    // f64 rint valid to |x/π| < 2^52  (≈ 1.42e16)
    // Payne-Hanek needed beyond that
    // -----------------------------------------------------------------------
    // 1e10→1e20 gap (f64 reduction should still work up to ~1e16)
    {1e11f, "1e11"},   // 85  ≈ 2^36.5
    {1e12f, "1e12"},   // 86  ≈ 2^39.9
    {1e13f, "1e13"},   // 87  ≈ 2^43.3
    {1e14f, "1e14"},   // 88  ≈ 2^46.5
    {1e15f, "1e15"},   // 89  ≈ 2^49.8
    {1e16f, "1e16"},   // 90  ≈ 2^53.2  <-- f64 rint boundary
    {1e17f, "1e17"},   // 91  ≈ 2^56.5
    {1e18f, "1e18"},   // 92  ≈ 2^59.8
    {1e19f, "1e19"},   // 93  ≈ 2^63.1
    // Beyond 1e20 up to FLT_MAX
    {1e22f, "1e22"},   // 94  ≈ 2^73.1
    {1e25f, "1e25"},   // 95  ≈ 2^83.0
    {1e28f, "1e28"},   // 96  ≈ 2^93.0
    {1e30f, "1e30"},   // 97  ≈ 2^99.7
    {1e33f, "1e33"},   // 98  ≈ 2^109.6
    {1e35f, "1e35"},   // 99  ≈ 2^116.3
    {1e37f, "1e37"},   // 100 ≈ 2^122.9
    // Negative counterparts (representative subset)
    {-1e12f, "-1e12"}, // 101
    {-1e14f, "-1e14"}, // 102
    {-1e16f, "-1e16"}, // 103
    {-1e18f, "-1e18"}, // 104
    {-1e22f, "-1e22"}, // 105
    {-1e25f, "-1e25"}, // 106
    {-1e30f, "-1e30"}, // 107
    {-1e35f, "-1e35"}, // 108
    {-1e37f, "-1e37"}, // 109
};
static constexpr size_t kNumTrigCases = sizeof(kCases) / sizeof(kCases[0]);

// ---------------------------------------------------------------------------
// Tester class
// ---------------------------------------------------------------------------
enum class TrigOp { Cos, Sin };

class TrigTester {
public:
  static constexpr size_t N = kNumTrigCases;

  float input[N];
  float output_libf[N];     // CUDA cosf() or sinf()
  float output_fast[N];     // CUDA __cosf() or __sinf() (fast approximate)
  float output_simple[N];   // custom_cos_simple() or custom_sin_simple()

  __host__ TrigTester() {
    for (size_t i = 0; i < N; i++)
      input[i] = kCases[i].x;
  }

  __host__ void reset() {
    std::memset(output_libf, 0xff, sizeof(output_libf));
    std::memset(output_fast, 0xff, sizeof(output_fast));
    std::memset(output_simple, 0xff, sizeof(output_simple));
  }

  // -- display ---------------------------------------------------------------

  void __host__ displayResults(TrigOp op, const float *torchinductor, const float *torcheager) const;
};

// -- kernels ---------------------------------------------------------------

__global__ void testKernelCosf(TrigTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < TrigTester::N)
    self->output_libf[idx] = cosf(self->input[idx]);
}

__global__ void testKernelCustomCosSimple(TrigTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < TrigTester::N)
    self->output_simple[idx] = custom_cos_simple(self->input[idx]);
}

__global__ void testKernelFastCos(TrigTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < TrigTester::N)
    self->output_fast[idx] = __cosf(self->input[idx]);
}

__global__ void testKernelSinf(TrigTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < TrigTester::N)
    self->output_libf[idx] = sinf(self->input[idx]);
}

__global__ void testKernelFastSin(TrigTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < TrigTester::N)
    self->output_fast[idx] = __sinf(self->input[idx]);
}

__global__ void testKernelCustomSinSimple(TrigTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < TrigTester::N)
    self->output_simple[idx] = custom_sin_simple(self->input[idx]);
}

// -- kernels for just looking at asm listings ------------------------------

__global__ void testKernelOneCosf(TrigTester *self) { self->output_libf[0] = cosf(self->input[0]); }

__global__ void testKernelOneCustomCosSimple(TrigTester *self) {
  self->output_simple[0] = custom_cos_simple(self->input[0]);
}

__global__ void testKernelOneFastCos(TrigTester *self) { self->output_fast[0] = __cosf(self->input[0]); }

__global__ void testKernelOneSinf(TrigTester *self) { self->output_libf[0] = sinf(self->input[0]); }

__global__ void testKernelOneCustomSinSimple(TrigTester *self) {
  self->output_simple[0] = custom_sin_simple(self->input[0]);
}

__global__ void testKernelOneFastSin(TrigTester *self) { self->output_fast[0] = __sinf(self->input[0]); }

bool verbose{};
bool useColor{};
bool quiet{};
bool csvOutput{};

// ---------------------------------------------------------------------------
// displayResults
// ---------------------------------------------------------------------------
void __host__ TrigTester::displayResults(TrigOp op, const float *torchinductor, const float *torcheager) const {
  const char *opName = (op == TrigOp::Cos) ? "cos" : "sin";

  if (csvOutput) {
    auto ref_func =
        (op == TrigOp::Cos) ? static_cast<float (*)(float)>(cosf) : static_cast<float (*)(float)>(sinf);

    std::string cudaCol = (op == TrigOp::Cos) ? "CUDA cosf" : "CUDA sinf";
    std::string fastCol = (op == TrigOp::Cos) ? "CUDA __cosf" : "CUDA __sinf";

    std::cout << std::format(
      "op,idx,x,label,row,std,\"{}\",\"{}\",asm,torch_eager,torch_inductor\n",
      cudaCol, fastCol);

    for (size_t i = 0; i < N; i++) {
      float x = input[i];
      float ref = ref_func(x);

      OneResult32 v_libf(ref, output_libf[i], true, verbose);
      OneResult32 v_fast(ref, output_fast[i], true, verbose);
      OneResult32 v_simple(ref, output_simple[i], true, verbose);
      OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f, torcheager != nullptr, verbose);
      OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f, torchinductor != nullptr, verbose);

      uint32_t rbits;
      std::memcpy(&rbits, &ref, sizeof(ref));
      std::string ref_hex = std::format("0x{:08x}", rbits);

      bool allMatch = v_libf.match and v_fast.match and v_simple.match and
                      v_inductor.match and v_eager.match;

      if (!quiet or !allMatch) {
        std::cout << std::format("{},{},{:g},\"{}\",VALUE,{:g},{:g},{:g},{:g},{},{}\n",
                                 opName, i, x, kCases[i].label, ref, output_libf[i], output_fast[i],
                                 output_simple[i],
                                 torcheager ? std::format("{:g}", torcheager[i]) : "",
                                 torchinductor ? std::format("{:g}", torchinductor[i]) : "");
      }

      if ((verbose or !allMatch) and (!quiet or !allMatch)) {
        std::cout << std::format("{},{},,\"\",HEX,{},{},{},{},{},{}\n", opName, i, ref_hex,
                                 v_libf.hexValue(), v_fast.hexValue(),
                                 v_simple.hexValue(),
                                 v_eager.hexValue(), v_inductor.hexValue());
      }

      if (!allMatch and (!quiet or !allMatch)) {
        int thresh = 4096;
        std::string es_libf = v_libf.errorString(thresh);
        std::string es_fast = v_fast.errorString(thresh);
        std::string es_simple = v_simple.errorString(thresh);
        std::string es_eager = v_eager.errorString(thresh);
        std::string es_inductor = v_inductor.errorString(thresh);

        std::cout << std::format("{},{},,\"\",DIFF,,{},{},{},{},{}\n", opName, i, es_libf, es_fast,
                                 es_simple, es_eager, es_inductor);
      }
    }
    return;
  }

  if (useColor) {
    std::cout << RED;
  }
  std::cout << ((op == TrigOp::Cos) ? "COSINE: cosf(x)\n\n\n\n" : "SINE: sinf(x)\n\n\n\n");
  if (useColor) {
    std::cout << RESET;
  }

  auto ref_func =
      (op == TrigOp::Cos) ? static_cast<float (*)(float)>(cosf) : static_cast<float (*)(float)>(sinf);

  std::string refCol = std::format("std::{}f", opName);
  std::string libCol = (op == TrigOp::Cos) ? "CUDA cosf" : "CUDA sinf";
  std::string fastCol = (op == TrigOp::Cos) ? "CUDA __cosf" : "CUDA __sinf";

  std::cout << std::format("{:>4}"     // Idx
                           "{:>16}"    // x
                           "{:>20}"    // Label
                           "{:>16}"    // std::cosf/sinf
                           "{:>16}"    // cosf/sinf
                           "{:>16}"    // __cosf/__sinf
                           "{:>16}"    // ASM simple
                           "{:>16}"    // torch-eager
                           "{:>16}\n", // torch-inductor
                           "Idx", "x", "Label", refCol, libCol, fastCol, "ASM",
                           torcheager ? "torch-eager" : "", torchinductor ? "torch-inductor" : "");

  std::cout << std::string(136, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float x = input[i];
    float ref = ref_func(x);

    OneResult32 v_libf(ref, output_libf[i], true, verbose);
    OneResult32 v_fast(ref, output_fast[i], true, verbose);
    OneResult32 v_simple(ref, output_simple[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f, torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f, torchinductor != nullptr, verbose);

    uint32_t rbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::string ref_hex = std::format("0x{:08x}", rbits);

    bool allMatch = v_libf.match and v_fast.match and v_simple.match and
                    v_inductor.match and v_eager.match;

    if (!quiet or !allMatch) {
      std::cout << std::format("{:>4}"
                               "{:>16g}"
                               "{:>20}"
                               "{:>16.6g}"
                               "{}"
                               "{}"
                               "{}"
                               "{}"
                               "{}\n",
                               i, x, kCases[i].label, ref, v_libf.value(), v_fast.value(), v_simple.value(),
                               v_eager.value(), v_inductor.value());
    }

    if (!allMatch) {
      std::string hexline =
          std::format("{:>4}"
                      "{:>16}"
                      "{:>20}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}\n",
                      "", "", "", ref_hex, v_libf.hexValue(), v_fast.hexValue(), v_simple.hexValue(),
                      v_eager.hexValue(), v_inductor.hexValue());
      std::cout << hexline;

      int thresh = 4096;
      std::string es_libf = v_libf.errorString(thresh);
      std::string es_fast = v_fast.errorString(thresh);
      std::string es_simple = v_simple.errorString(thresh);
      std::string es_eager = v_eager.errorString(thresh);
      std::string es_inductor = v_inductor.errorString(thresh);

      const char *color = YELLOW;

      if ((es_libf == "ERROR") or (es_fast == "ERROR") or (es_simple == "ERROR") or
          (es_eager == "ERROR") or (es_inductor == "ERROR")) {
        color = RED;
      }

      if (useColor) {
        std::cout << color;
      }
      std::string matchline =
          std::format("{:>4}"
                      "{:>16}"
                      "{:>20}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}\n",
                      "", "", "", "", es_libf, es_fast, es_simple, es_eager, es_inductor);
      std::cout << matchline;
      if (useColor) {
        std::cout << RESET;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  int c{};
  const char *dumpFile{};
  const char *torchinductorFile = nullptr;
  const char *torcheagerFile = nullptr;
  TrigOp op = TrigOp::Cos;
  bool opSet = false;
  constexpr struct option longOptions[]{
      {"help", 0, nullptr, 'h'},          {"verbose", 0, nullptr, 'v'},    {"quiet", 0, nullptr, 'q'},
      {"color", 0, nullptr, 'c'},         {"csv", 0, nullptr, 'm'},        {"op", 1, nullptr, 'o'},
      {"dump-inputs", 1, nullptr, 'd'},
      {"torchinductor", 1, nullptr, 'y'}, {"torcheager", 1, nullptr, 't'}, {nullptr, 0, nullptr, 0}};

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
    case 'm': {
      csvOutput = true;
      break;
    }
    case 'o': {
      std::string s = optarg;
      if (s == "cos") {
        op = TrigOp::Cos;
        opSet = true;
      } else if (s == "sin") {
        op = TrigOp::Sin;
        opSet = true;
      } else {
        std::cerr << "cossin_test: --op must be 'cos' or 'sin'\n";
        return 1;
      }
      break;
    }
    case 'd': {
      dumpFile = optarg;
      break;
    }
    case 'h': {
      std::cout << "cossin_test --op cos|sin"
                   " [--verbose]"
                   " [--quiet]"
                   " [--color]"
                   " [--csv]"
                   " [--dump-inputs filename]"
                   " [--torcheager torcheagercos.bin]"
                   " [--torchinductor torchinductorcos.bin]"
                   "\n\n"
                   "Run with:\n"
                   "  ./math/cossin_test --dump-inputs ./trigunarytest.in\n"
                   "  ../torch/torchunary.py --op sin --file ./trigunarytest.in\n"
                   "  ../torch/torchunary.py --op cos --file ./trigunarytest.in\n"
                   "  ./math/cossin_test --op sin --torchinductor "
                   "torchinductorsin.bin --torcheager torcheagersin.bin "
                   "--verbose --quiet --color | less -R\n"
                   "  ./math/cossin_test --op cos --torchinductor "
                   "torchinductorcos.bin --torcheager torcheagercos.bin "
                   "--verbose --quiet --color | less -R\n"
                   "\n\n"
                   "\t--op cos|sin.  Select cos or sin operation (required "
                   "unless --dump)\n"
                   "\t--dump-inputs filename.  Write input values as binary "
                   "floats to file (x0,x1,x2,...)\n"
                   "\t--verbose.  Show hex values, even if not mismatches\n"
                   "\t--quiet.  Suppress non-matching output\n"
                   "\t--color.  Highlight mismatches in color\n"
                   "\t--csv.  Emit CSV output (VALUE/HEX/DIFF rows)\n"
                   "\t--help.  Show this output and exit\n";
      return 0;
    }
    case 'y': {
      torchinductorFile = optarg;
      break;
    }
    case 't': {
      torcheagerFile = optarg;
      break;
    }
    default: {
      std::cerr << "cossin_test: unknown option\n";
      return 1;
    }
    }
  }

  // Dump input values to binary file if requested
  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << std::endl;
      return 2;
    }
    for (size_t i = 0; i < TrigTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&kCases[i].x), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input values to " << dumpFile << std::endl;
    return 0;
  }

  // dump input doesn't care about the op... check if we get this far
  if (!opSet) {
    std::cerr << "cossin_test: --op cos|sin is required\n";
    return 1;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, TrigTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, TrigTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  TrigTester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(TrigTester)));
  new (tester) TrigTester();
  tester->reset();

  dim3 blockSize(TrigTester::N);
  dim3 gridSize(1);

  if (op == TrigOp::Cos) {
    testKernelCosf<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());

    testKernelFastCos<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());

    testKernelCustomCosSimple<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    testKernelSinf<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());

    testKernelFastSin<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());

    testKernelCustomSinSimple<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  tester->displayResults(op, torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  CUDA_CHECK(cudaFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
