//
// sqrtrsq_test.cu
//
// Tests sqrt and rsqrt on NVIDIA H100 hardware,
// comparing results against std::sqrt() and 1/std::sqrt() computed on the CPU.
//
// The intent is to:
//   1. Establish ground truth via CUDA's sqrtf() and rsqrtf().
//   2. Test hand-coded inline-asm sqrt(x) and rsqrt(x).
//
// v_sqrt_f32: computes sqrt(x)
// v_rsq_f32:  computes 1/sqrt(x) (reciprocal square root)
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
#include <vector>

#include "readbinary.hpp"
#include "OneResult32.hpp"
#include "colors.hpp"
#include "cuda_check.hpp"

// ---------------------------------------------------------------------------
// Custom sqrt and rsqrt -- inline ASM versions of the CUDA operations.
// ---------------------------------------------------------------------------
// clang-format off
inline float __device__ custom_sqrt(float x) {
  float result;

  __asm__ __volatile__(
      "// %0 = sqrt(%1)\n\t"
      "sqrt.rn.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );

  return result;
}

inline float __device__ custom_rsqrt(float x) {
  float result;

  __asm__ __volatile__(
      "// %0 = rsqrt(%1)\n\t"
      "rsqrt.approx.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );

  return result;
}
// clang-format on

#define CUSTOM_SQRT custom_sqrt
#define CUSTOM_RSQRT custom_rsqrt

// ---------------------------------------------------------------------------
// Test-case table
// ---------------------------------------------------------------------------
struct SqrtCase {
  float x;
  const char *label;
};

static constexpr SqrtCase kCases[] = {
    // Powers of 2 (exact results expected)
    {1.0f, "1.0"},                    //  0: sqrt(1) = 1, rsqrt(1) = 1
    {4.0f, "4.0"},                    //  1: sqrt(4) = 2, rsqrt(4) = 0.5
    {16.0f, "16.0"},                  //  2: sqrt(16) = 4, rsqrt(16) = 0.25
    {64.0f, "64.0"},                  //  3: sqrt(64) = 8
    {256.0f, "256.0"},                //  4: sqrt(256) = 16
    {0.25f, "0.25"},                  //  5: sqrt(0.25) = 0.5, rsqrt(0.25) = 2
    {0.0625f, "0.0625"},              //  6: sqrt(1/16) = 0.25

    // Non-power-of-2 values
    {2.0f, "2.0"},                    //  7: sqrt(2) ≈ 1.414
    {3.0f, "3.0"},                    //  8: sqrt(3) ≈ 1.732
    {5.0f, "5.0"},                    //  9: sqrt(5) ≈ 2.236
    {10.0f, "10.0"},                  // 10: sqrt(10) ≈ 3.162
    {100.0f, "100.0"},                // 11: sqrt(100) = 10
    {1000.0f, "1000.0"},              // 12: sqrt(1000) ≈ 31.62

    // Values near 1
    {0.9f, "0.9"},                    // 13
    {1.1f, "1.1"},                    // 14
    {0.99f, "0.99"},                  // 15
    {1.01f, "1.01"},                  // 16

    // Small positive values
    {1e-6f, "1e-6"},                  // 17: sqrt(1e-6) = 1e-3
    {1e-10f, "1e-10"},                // 18: sqrt(1e-10) = 1e-5
    {1e-20f, "1e-20"},                // 19: sqrt(1e-20) = 1e-10
    {1e-30f, "1e-30"},                // 20: sqrt(1e-30) = 1e-15

    // Large positive values
    {1e6f, "1e6"},                    // 21: sqrt(1e6) = 1e3
    {1e10f, "1e10"},                  // 22: sqrt(1e10) = 1e5
    {1e20f, "1e20"},                  // 23: sqrt(1e20) = 1e10
    {1e30f, "1e30"},                  // 24: sqrt(1e30) = 1e15

    // Near subnormal boundary
    {1.17549435e-38f, "FLT_MIN"},     // 25: smallest normal float
    {1e-40f, "1e-40 (subnormal)"},    // 26: subnormal value

    // Special value: zero
    {0.0f, "0.0"},                    // 27: sqrt(0) = 0, rsqrt(0) = inf
    {-0.0f, "-0.0"},                  // 28: sqrt(-0) = -0, rsqrt(-0) = -inf

    // Negative values (should produce NaN)
    {-1.0f, "-1.0"},                  // 29: sqrt(-1) = NaN
    {-4.0f, "-4.0"},                  // 30: sqrt(-4) = NaN
    {-100.0f, "-100.0"},              // 31: sqrt(-100) = NaN

    // Special IEEE values
    {std::numeric_limits<float>::infinity(), "+inf"},         // 32: sqrt(inf) = inf, rsqrt(inf) = 0
    {-std::numeric_limits<float>::infinity(), "-inf"},        // 33: sqrt(-inf) = NaN
    {std::numeric_limits<float>::quiet_NaN(), "NaN"},         // 34: sqrt(NaN) = NaN

    // Large finite value near overflow
    {3.4e38f, "3.4e38 (near max)"},   // 35: sqrt(3.4e38) ≈ 1.84e19

    // Values that test precision
    {0.5f, "0.5"},                    // 36: sqrt(0.5) ≈ 0.707
    {1.5f, "1.5"},                    // 37: sqrt(1.5) ≈ 1.225
    {7.0f, "7.0"},                    // 38: sqrt(7) ≈ 2.646
    {0.1f, "0.1"},                    // 39: sqrt(0.1) ≈ 0.316

    // -----------------------------------------------------------------------
    // Negative NaN and signaling NaN
    // -----------------------------------------------------------------------
    {-std::numeric_limits<float>::quiet_NaN(), "-NaN"},       // 40
    {std::numeric_limits<float>::signaling_NaN(), "sNaN"},    // 41

    // -----------------------------------------------------------------------
    // Subnormal (denormalized) inputs
    // -----------------------------------------------------------------------
    {1.401298e-45f, "min subnorm"},          // 42: smallest positive subnormal
    {5.877472e-39f, "mid subnorm"},          // 43
    {1.1754942e-38f, "max subnorm"},         // 44: largest subnormal
    {-1.401298e-45f, "-min subnorm"},        // 45: sqrt(-subnorm) = NaN
    {-1.1754942e-38f, "-max subnorm"},       // 46

    // -----------------------------------------------------------------------
    // FLT_MIN and FLT_MAX via <limits>
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::min(), "FLT_MIN (limits)"},  // 47
    {-std::numeric_limits<float>::min(), "-FLT_MIN"},         // 48
    {std::numeric_limits<float>::max(), "FLT_MAX"},           // 49
    {-std::numeric_limits<float>::max(), "-FLT_MAX"},         // 50

    // -----------------------------------------------------------------------
    // Near-1 tight neighbourhood (sqrt(1+ε) ≈ 1+ε/2)
    // -----------------------------------------------------------------------
    {1.0f + 1e-7f, "1 + 1e-7"},      // 51
    {1.0f - 1e-7f, "1 - 1e-7"},      // 52
    {1.0f + 1e-5f, "1 + 1e-5"},      // 53
    {1.0f - 1e-5f, "1 - 1e-5"},      // 54

    // -----------------------------------------------------------------------
    // Very small (near underflow boundary for rsqrt)
    // -----------------------------------------------------------------------
    {1e-38f, "1e-38"},                // 55
    {1e-7f, "1e-7"},                  // 56

    // -----------------------------------------------------------------------
    // Perfect squares (exact results expected)
    // -----------------------------------------------------------------------
    {9.0f, "9.0"},                    // 57
    {25.0f, "25.0"},                  // 58
    {49.0f, "49.0"},                  // 59
    {10000.0f, "10000.0"},            // 60

    // -----------------------------------------------------------------------
    // Large values stressing rsqrt precision
    // -----------------------------------------------------------------------
    {1e35f, "1e35"},                  // 61
    {1e38f, "1e38"},                  // 62

    // -----------------------------------------------------------------------
    // Very small but negative (near underflow boundary for rsqrt)
    // -----------------------------------------------------------------------
    {-1e-38f, "-1e-38"},                // 63
    {-1e-7f, "-1e-7"},                  // 64
                                        //
    // -----------------------------------------------------------------------
    // Large values stressing rsqrt precision (but negative)
    // -----------------------------------------------------------------------
    {-1e35f, "-1e35"},                  // 65
    {-1e38f, "-1e38"},                  // 66

};
static constexpr size_t kNumSqrtCases = sizeof(kCases) / sizeof(kCases[0]);

// ---------------------------------------------------------------------------
// Tester class
// ---------------------------------------------------------------------------
enum class SqrtOp { Sqrt, Rsqrt };

class SqrtRsqTester {
public:
  static constexpr size_t N = kNumSqrtCases;

  float input[N];
  float output_libf[N];    // CUDA sqrtf() or rsqrtf()
  float output_custom[N];  // CUSTOM_SQRT() or CUSTOM_RSQRT()

  __host__ SqrtRsqTester() {
    for (size_t i = 0; i < N; i++)
      input[i] = kCases[i].x;
  }

  __host__ void reset() {
    std::memset(output_libf, 0xff, sizeof(output_libf));
    std::memset(output_custom, 0xff, sizeof(output_custom));
  }

  // -- display ---------------------------------------------------------------

  void __host__ displayResults(SqrtOp op,
                               const float *torchinductor,
                               const float *torcheager) const;
};

// -- kernels ---------------------------------------------------------------

__global__ void testKernelSqrtf(SqrtRsqTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < SqrtRsqTester::N)
    self->output_libf[idx] = sqrtf(self->input[idx]);
}

__global__ void testKernelCustomSqrt(SqrtRsqTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < SqrtRsqTester::N)
    self->output_custom[idx] = CUSTOM_SQRT(self->input[idx]);
}

__global__ void testKernelRsqrtf(SqrtRsqTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < SqrtRsqTester::N)
    self->output_libf[idx] = rsqrtf(self->input[idx]);
}

__global__ void testKernelCustomRsqrt(SqrtRsqTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < SqrtRsqTester::N)
    self->output_custom[idx] = CUSTOM_RSQRT(self->input[idx]);
}

// -- kernels for just looking at asm listings ------------------------------

__global__ void testKernelOneSqrtf(SqrtRsqTester *self) {
  self->output_libf[0] = sqrtf(self->input[0]);
}

__global__ void testKernelOneCustomSqrt(SqrtRsqTester *self) {
  self->output_custom[0] = CUSTOM_SQRT(self->input[0]);
}

__global__ void testKernelOneRsqrtf(SqrtRsqTester *self) {
  self->output_libf[0] = rsqrtf(self->input[0]);
}

__global__ void testKernelOneCustomRsqrt(SqrtRsqTester *self) {
  self->output_custom[0] = CUSTOM_RSQRT(self->input[0]);
}

bool verbose{};
bool useColor{};
bool quiet{};

// ---------------------------------------------------------------------------
// displayResults
// ---------------------------------------------------------------------------
void __host__ SqrtRsqTester::displayResults(SqrtOp op,
                                             const float *torchinductor,
                                             const float *torcheager) const {
  const bool isSqrt = (op == SqrtOp::Sqrt);
  const char *opName = isSqrt ? "sqrt" : "rsqrt";

  if (useColor) {
    std::cout << RED;
  }
  std::cout << (isSqrt ? "SQRT: sqrtf(x)\n\n\n\n" : "RSQRT: rsqrtf(x)\n\n\n\n");
  if (useColor) {
    std::cout << RESET;
  }

  // Reference function: std::sqrt or 1/std::sqrt
  auto ref_func = isSqrt
      ? static_cast<float (*)(float)>([](float x) -> float { return std::sqrt(x); })
      : static_cast<float (*)(float)>([](float x) -> float { return 1.0f / std::sqrt(x); });

  std::string refCol = std::format("std::{}", opName);
  std::string libCol = std::format("{}f", opName);
  std::string asmCol = std::format("ASM({})", opName);

  std::cout << std::format("{:>4}"     // Idx
                           "{:>16}"    // x
                           "{:>20}"    // Label
                           "{:>16}"    // std::sqrt / std::rsqrt
                           "{:>16}"    // sqrtf / rsqrtf
                           "{:>16}"    // ASM
                           "{:>16}"    // torch-eager
                           "{:>16}\n", // torch-inductor
                           "Idx", "x", "Label", refCol, libCol,
                           asmCol,
                           torcheager ? "torch-eager" : "",
                           torchinductor ? "torch-inductor" : "");

  std::cout << std::string(120, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float x = input[i];
    float ref = ref_func(x);

    OneResult32 v_libf(ref, output_libf[i], true, verbose);
    OneResult32 v_asm(ref, output_custom[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                      torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                         torchinductor != nullptr, verbose);

    uint32_t rbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::string ref_hex = std::format("0x{:08x}", rbits);

    bool allMatch = v_libf.match and v_asm.match and
                    v_inductor.match and v_eager.match;

    if (!quiet or !allMatch) {
      std::cout << std::format("{:>4}"
                               "{:>16g}"
                               "{:>20}"
                               "{:>16.6g}"
                               "{}"
                               "{}"
                               "{}"
                               "{}\n",
                               i, x, kCases[i].label, ref, v_libf.value(),
                               v_asm.value(), v_eager.value(),
                               v_inductor.value());
    }

    if (!allMatch) {
      std::string hexline = std::format(
          "{:>4}"
          "{:>16}"
          "{:>20}"
          "{:>16}"
          "{:>16}"
          "{:>16}"
          "{:>16}"
          "{:>16}\n",
          "", "", "", ref_hex, v_libf.hexValue(), v_asm.hexValue(),
          v_eager.hexValue(), v_inductor.hexValue());
      std::cout << hexline;

      std::string es_libf = v_libf.errorString();
      std::string es_asm = v_asm.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();

      const char *color = YELLOW;

      if ((es_libf == "ERROR") or (es_asm == "ERROR") or
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
                      "{:>16}\n",
                      "", "", "", "", es_libf, es_asm,
                      es_eager, es_inductor);
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
  SqrtOp op = SqrtOp::Sqrt;
  bool opSet = false;
  constexpr struct option longOptions[]{{"help", 0, nullptr, 'h'},
                                        {"verbose", 0, nullptr, 'v'},
                                        {"quiet", 0, nullptr, 'q'},
                                        {"color", 0, nullptr, 'c'},
                                        {"op", 1, nullptr, 'o'},
                                        {"dump-inputs", 1, nullptr, 'd'},
                                        {"torchinductor", 1, nullptr, 'y'},
                                        {"torcheager", 1, nullptr, 't'},
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
    case 'o': {
      std::string s = optarg;
      if (s == "sqrt") {
        op = SqrtOp::Sqrt;
        opSet = true;
      } else if (s == "rsqrt") {
        op = SqrtOp::Rsqrt;
        opSet = true;
      } else {
        std::cerr << "sqrtrsq_test: --op must be 'sqrt' or 'rsqrt'\n";
        return 1;
      }
      break;
    }
    case 'd': {
      dumpFile = optarg;
      break;
    }
    case 'h': {
      std::cout << "sqrtrsq_test --op sqrt|rsqrt"
                   " [--verbose]"
                   " [--quiet]"
                   " [--color]"
                   " [--dump-inputs filename]"
                   " [--torcheager torcheagersqrt.bin]"
                   " [--torchinductor torchinductorsqrt.bin]"
                   "\n\n"
                   "Run with:\n"
                   "  ./math/sqrtrsq_test --dump-inputs ./roottest.in\n"
                   "  ../bin/torchunary.py --op sqrt --file ./roottest.in\n"
                   "  ../bin/torchunary.py --op rsqrt --file ./roottest.in\n"
                   "  ./math/sqrtrsq_test --op sqrt --torchinductor torchinductorsqrt.bin --torcheager torcheagersqrt.bin --verbose --quiet --color | less -R\n"
                   "  ./math/sqrtrsq_test --op rsqrt --torchinductor torchinductorrsqrt.bin --torcheager torcheagerrsqrt.bin --verbose --quiet --color | less -R\n"
                   "\n"
                   "\t--op sqrt|rsqrt.  Select sqrt or rsqrt operation (required)\n"
                   "\t--dump-inputs filename.  Write input values as binary "
                   "floats to file (x0,x1,x2,...)\n"
                   "\t--verbose.  Show hex values, even if not mismatches\n"
                   "\t--quiet.  Suppress non-matching output\n"
                   "\t--color.  Highlight mismatches in color\n"
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
      std::cerr << "sqrtrsq_test: unknown option\n";
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
    for (size_t i = 0; i < SqrtRsqTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&kCases[i].x), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input values to " << dumpFile << std::endl;
    return 0;
  }

  if (!opSet) {
    std::cerr << "sqrtrsq_test: --op sqrt|rsqrt is required\n";
    return 1;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, SqrtRsqTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, SqrtRsqTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  SqrtRsqTester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(SqrtRsqTester)));
  new (tester) SqrtRsqTester();
  tester->reset();

  dim3 blockSize(SqrtRsqTester::N);
  dim3 gridSize(1);

  if (op == SqrtOp::Sqrt) {
    testKernelSqrtf<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());

    testKernelCustomSqrt<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    testKernelRsqrtf<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());

    testKernelCustomRsqrt<<<gridSize, blockSize>>>(tester);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  tester->displayResults(op,
                         torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  CUDA_CHECK(cudaFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
