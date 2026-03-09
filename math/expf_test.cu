//
// The original LLIR vExpInRegFp32, was supposed to lower an e^x operation,
// but used v_exp_f32, which is a 2^x operation.
//
// This code implements a v_exp_f32 based computation of e^x, scaling
// the inputs by log_2(e).  It compares the output to std::expf, and
// optionally to torch.exp outputs of the same computations.
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

// clang-format off
// PTX inline asm implementation.
inline float __device__ expf_custom(float x) {
  float result, tmp_f;
  int tmp_i;

  // The number 252 is 2 × 126, where 126 is the magnitude of the minimum normal
  // exponent in IEEE 754 single precision (exponent bias 127, minimum normal
  // exponent = −126).
  //
  // The useful range of (x ⋅ log_2 e) for single-precision
  // exp(x) is approximately [−126,+126], (beyond that, the result is 0 or ∞). The
  // goal is to map this range into [0,1] so the hardware saturation clamp
  // (cvt.sat) can handle overflow and underflow in a single branchless
  // instruction.
  __asm__ __volatile__(
      // exp(x) = 2^(x * log2(e)), decomposed as 2^n * 2^frac
      // Step 1: map x into [0,1] range with overflow/underflow clamp
      "fma.rn.f32 %0, %3, 0f3BBB989D, 0f3F000000;\n\t"
      // %0 = x * (log2(e)/252) + 0.5
      "cvt.sat.f32.f32 %0, %0;\n\t"
      // %0 = clamp(%0, 0.0, 1.0)
      // Step 2: magic-number rounding — extract integer exponent n
      "fma.rm.f32 %0, %0, 0f437C0000, 0f4B400001;\n\t"
      // %0 = floor(%0 * 252.0 + 12582913.0)
      // low bits of mantissa now encode n+1
      // Step 3: Cody-Waite range reduction for fractional part
      "add.f32 %1, %0, 0fCB40007F;\n\t"
      // %1 = %0 - 12583039.0 = (n - 126)
      "neg.f32 %1, %1;\n\t"
      // %1 = -(n - 126) = (126 - n)
      "fma.rn.f32 %1, %3, 0f3FB8AA3B, %1;\n\t"
      // %1 = x * log2(e)_hi + (126 - n)
      "fma.rn.f32 %1, %3, 0f32A57060, %1;\n\t"
      // %1 += x * log2(e)_lo — low-order Cody-Waite correction
      // %1 now holds the fractional part f of x*log2(e)
      // Step 4: construct 2^(n-126) via IEEE bit manipulation
      "mov.b32 %2, %0;\n\t"
      // %2 = reinterpret magic-rounded float as int32
      "shl.b32 %2, %2, 23;\n\t"
      // %2 <<= 23 — shift n into IEEE 754 exponent field
      "mov.b32 %0, %2;\n\t"
      // %0 = reinterpret as float = 2^(n-126)
      // Step 5: hardware exp2 of fractional part, then combine
      "ex2.approx.ftz.f32 %1, %1;\n\t"
      // %1 = 2^f — hardware fast approximation, denorms flushed to zero
      "mul.f32 %0, %1, %0;\n\t"
      // %0 = 2^f * 2^(n-126) = 2^(x*log2(e)) = exp(x)
      : "=&f"(result),  // %0 — early clobber: written before %3 is fully consumed
        "=&f"(tmp_f),   // %1 — early clobber: written before %3 is fully consumed
        "=r"(tmp_i)     // %2 — no early clobber needed (different class, and written after last %3 read)
      : "f"(x)          // %3 — input: x, read in multiple instructions throughout
      );

  return result;
}
// clang-format on

#define CUSTOM_EXPF expf_custom

// ---------------------------------------------------------------------------
// Test-case table
// ---------------------------------------------------------------------------
struct ExpCase {
  float x;
  const char *label;
};

// Keep in sync with ExpTester::N below.
static constexpr ExpCase kCases[] = {
    // Simple integer values
    {0.0f, "0"},             //  0
    {1.0f, "1"},             //  1
    {2.0f, "2"},             //  2
    {3.0f, "3"},             //  3
    {-1.0f, "-1"},           //  4
    {-2.0f, "-2"},           //  5

    // ln(2)
    {0.693147180559945309417f, "ln(2)"}, //  6

    // Fractional values
    {0.5f, "0.5"},           //  7
    {1.5f, "1.5"},           //  8
    {-0.5f, "-0.5"},         //  9

    // Larger values
    {10.0f, "10"},           // 10
    {-10.0f, "-10"},         // 11
    {5.0f, "5"},             // 12
    {-5.0f, "-5"},           // 13

    // Very small values
    {0.01f, "0.01"},         // 14
    {0.001f, "0.001"},       // 15
    {-0.01f, "-0.01"},       // 16

    // Special values
    {std::numeric_limits<float>::infinity(), "+inf"},      // 17
    {-std::numeric_limits<float>::infinity(), "-inf"},     // 18
    {std::numeric_limits<float>::quiet_NaN(), "NaN"},      // 19
    {0.0000000001f, "0.0000000001"},      // 20
    {-0.0f, "-0"},           // 21

    // Values near overflow/underflow boundaries
    {88.0f, "88 (e^x ovfl)"},    // 22
    {-88.0f, "-88 (e^x unfl)"},  // 23
    {127.0f, "127 (2^x ovfl)"},  // 24
    {-126.0f, "-126 (2^x unfl)"},// 25

    // Additional test values
    {3.14159265f, "pi"},     // 26
    {2.71828183f, "e"},      // 27
    {0.25f, "0.25"},         // 28
    {4.0f, "4"},             // 29
    {-3.0f, "-3"},           // 30
    {7.5f, "7.5"},           // 31

    // -----------------------------------------------------------------------
    // Subnormal (denormalized) inputs
    // -----------------------------------------------------------------------
    {1.401298e-45f, "min subnorm"},         // 32: smallest positive subnormal
    {5.877472e-39f, "mid subnorm"},         // 33: mid-range subnormal
    {1.1754942e-38f, "max subnorm"},        // 34: largest subnormal (just below FLT_MIN)
    {-1.401298e-45f, "-min subnorm"},       // 35: smallest negative subnormal
    {-1.1754942e-38f, "-max subnorm"},      // 36: largest negative subnormal

    // -----------------------------------------------------------------------
    // FLT_MIN (smallest normal) and FLT_MAX
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::min(), "FLT_MIN"},       // 37: 1.17549435e-38
    {-std::numeric_limits<float>::min(), "-FLT_MIN"},     // 38
    {std::numeric_limits<float>::max(), "FLT_MAX"},       // 39: 3.40282347e+38
    {-std::numeric_limits<float>::max(), "-FLT_MAX"},     // 40: (-> 0)

    // -----------------------------------------------------------------------
    // Precise e^x overflow/underflow boundaries
    // e^88.7228... = FLT_MAX, e^88.7229 overflows
    // e^-87.3365... = FLT_MIN (smallest normal)
    // e^-103.97... underflows to zero (past subnormal range)
    // -----------------------------------------------------------------------
    {88.72f, "88.72 (e^x edge)"},           // 41: just below overflow
    {88.73f, "88.73 (e^x ovfl)"},           // 42: just above overflow
    {89.0f, "89"},                          // 43: safely overflows
    {-87.0f, "-87 (e^x near unfl)"},        // 44: near underflow to subnormal
    {-87.34f, "-87.34 (e^x unfl)"},         // 45: near FLT_MIN boundary
    {-104.0f, "-104 (e^x->0)"},             // 46: underflows past subnormals

    // -----------------------------------------------------------------------
    // Precise 2^x overflow/underflow boundaries (for v_exp_f32 column)
    // 2^128 overflows; 2^-149 is smallest subnormal; 2^-126 is FLT_MIN
    // -----------------------------------------------------------------------
    {128.0f, "128 (2^x ovfl)"},             // 47
    {-149.0f, "-149 (2^x min sub)"},        // 48
    {-150.0f, "-150 (2^x->0)"},             // 49
    {126.0f, "126"},                        // 50
    {-125.0f, "-125"},                      // 51

    // -----------------------------------------------------------------------
    // Near-zero neighbourhood
    // -----------------------------------------------------------------------
    {1e-7f, "1e-7"},          // 52
    {-1e-7f, "-1e-7"},        // 53
    {1e-20f, "1e-20"},        // 54
    {-1e-20f, "-1e-20"},      // 55
    {1e-38f, "1e-38"},        // 56
    {-1e-38f, "-1e-38"},      // 57

    // -----------------------------------------------------------------------
    // Large negative values (well past underflow)
    // -----------------------------------------------------------------------
    {-200.0f, "-200"},        // 58
    {-500.0f, "-500"},        // 59
    {-1000.0f, "-1000"},      // 60

    // -----------------------------------------------------------------------
    // Near-one neighbourhood (e^x ~ 1+x for small x)
    // -----------------------------------------------------------------------
    {1e-5f, "1e-5"},          // 61
    {-1e-5f, "-1e-5"},        // 62
    {1e-3f, "1e-3"},          // 63
    {-1e-3f, "-1e-3"},        // 64
};
static_assert(sizeof(kCases) / sizeof(kCases[0]) == 65,
              "Update ExpTester::N to match kCases length");

class ExpTester {
public:
  static constexpr size_t N = 65;
  float input[N];
  float output_vexp[N];      // v_exp_f32 results (2^x)
  float output_exp[N];       // CUSTOM_EXPF results (e^x via mul+exp)
  float output_cuda_exp[N];  // expf() results (CUDA library)
  float output_cuda_fexp[N]; // __expf() results (CUDA fast approx)

  __host__ ExpTester() {
    for (size_t i = 0; i < N; i++) {
      input[i] = kCases[i].x;
    }
  }

  void __host__ reset() {
    std::memset(output_vexp, 0xff, sizeof(output_vexp));
    std::memset(output_exp, 0xff, sizeof(output_exp));
    std::memset(output_cuda_exp, 0xff, sizeof(output_cuda_exp));
    std::memset(output_cuda_fexp, 0xff, sizeof(output_cuda_fexp));
  }

  void __host__ displayResults(const float *torchinductor,
                               const float *torcheager) const;
};

__global__ void testKernelExp(ExpTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ExpTester::N) {
    self->output_exp[idx] = CUSTOM_EXPF(self->input[idx]);
  }
}

__global__ void testKernelCudaExp(ExpTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ExpTester::N) {
    self->output_cuda_exp[idx] = expf(self->input[idx]);
  }
}

__global__ void testKernelCudaFExp(ExpTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ExpTester::N) {
    self->output_cuda_fexp[idx] = __expf(self->input[idx]); // Fast approximate e^x (lower precision)
  }
}

// -- kernels for just looking at asm listings ------------------------------

__global__ void testKernelOneExp(ExpTester *self) {
  self->output_exp[0] = CUSTOM_EXPF(self->input[0]);
}

__global__ void testKernelOneCudaExp(ExpTester *self) {
  self->output_cuda_exp[0] = expf(self->input[0]);
}

__global__ void testKernelOneCudaFExp(ExpTester *self) {
  self->output_cuda_fexp[0] = __expf(self->input[0]);
}

bool verbose{};
bool useColor{};
bool quiet{};

// ---------------------------------------------------------------------------
// displayResults
// ---------------------------------------------------------------------------
void __host__ ExpTester::displayResults(const float *torchinductor,
                                        const float *torcheager) const {
  // Reference is std::exp (e^x).  Columns: expf, __expf, ASM (CUSTOM_EXPF),
  // and optionally torch-eager, torch-inductor.

  if (useColor) {
    std::cout << RED;
  }
  std::cout << "NATURAL EXPONENTIAL: e^x\n\n\n\n";
  if (useColor) {
    std::cout << RESET;
  }

  std::cout << std::format("{:>4}"     // Idx
                           "{:>16}"    // x
                           "{:>20}"    // Label
                           "{:>16}"    // std::exp
                           "{:>16}"    // expf
                           "{:>16}"    // __expf
                           "{:>16}"    // ASM
                           "{:>16}"    // torch-eager
                           "{:>16}\n", // torch-inductor
                           "Idx", "x", "Label", "std::exp", "expf",
                           "__expf", "ASM(e^x)",
                           torcheager ? "torch-eager" : "",
                           torchinductor ? "torch-inductor" : "");

  std::cout << std::string(175, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float x = input[i];
    float ref = std::exp(x);

    OneResult32 v_expf(ref, output_cuda_exp[i], true, verbose);
    OneResult32 v__expf(ref, output_cuda_fexp[i], true, verbose);
    OneResult32 v_asm(ref, output_exp[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                      torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                         torchinductor != nullptr, verbose);

    uint32_t rbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::string ref_hex = std::format("0x{:08x}", rbits);

    bool allMatch = v_expf.match and v__expf.match and v_asm.match and
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
                               i, x, kCases[i].label, ref, v_expf.value(),
                               v__expf.value(), v_asm.value(),
                               v_eager.value(), v_inductor.value());
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
          "{:>16}"
          "{:>16}\n",
          "", "", "", ref_hex, v_expf.hexValue(), v__expf.hexValue(),
          v_asm.hexValue(), v_eager.hexValue(),
          v_inductor.hexValue());
      std::cout << hexline;

      std::string es_expf = v_expf.errorString();
      std::string es__expf = v__expf.errorString();
      std::string es_asm = v_asm.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();

      const char *color = YELLOW;

      if ((es_expf == "ERROR") or (es__expf == "ERROR") or
          (es_asm == "ERROR") or
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
                      "", "", "", "", es_expf, es__expf, es_asm,
                      es_eager, es_inductor);
      std::cout << matchline;
      if (useColor) {
        std::cout << RESET;
      }
    }
  }
}

int main(int argc, char **argv) {
  int c{};
  const char *dumpFile{};
  const char *torchinductorFile = nullptr;
  const char *torcheagerFile = nullptr;
  constexpr struct option longOptions[]{{"help", 0, nullptr, 'h'},
                                        {"verbose", 0, nullptr, 'v'},
                                        {"quiet", 0, nullptr, 'q'},
                                        {"color", 0, nullptr, 'c'},
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
    case 'd': {
      dumpFile = optarg;
      break;
    }
    case 'h': {
      std::cout << "expf_test"
                   " [--verbose]"
                   " [--quiet]"
                   " [--color]"
                   " [--dump-inputs filename]"
                   " [--torcheager torcheagerexp.bin]"
                   " [--torchinductor torchinductorexp.bin]"
                   "\n\n"
                   "Run with:\n"
                   "  pip3 install torch --index-url "
                   "https://download.pytorch.org/whl/cu126\n"
                   "\n\n"
                   "./math/expf_test --dump-inputs ./exptest.in\n"
                   "../torchexp.py file ./exptest.in\n"
                   "./math/expf_test --torchinductor torchinductor.bin --torcheager torcheager.bin --verbose --quiet --color | less -R\n\n"
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
      std::cerr << "expf_test: unknown option\n";
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
    ExpTester tmp;
    for (size_t i = 0; i < ExpTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&tmp.input[i]), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input values to " << dumpFile << std::endl;
    return 0;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, ExpTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, ExpTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  // Allocate tester object in managed memory
  ExpTester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(ExpTester)));

  new (tester) ExpTester();

  // Launch kernels
  dim3 blockSize(ExpTester::N);
  dim3 gridSize(1);

  tester->reset();

  // Test CUSTOM_EXPF (e^x via mul+exp)
  testKernelExp<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Test CUDA expf()
  testKernelCudaExp<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Test CUDA __expf()
  testKernelCudaFExp<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Display results (computes CPU reference inline)
  tester->displayResults(torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  // Cleanup
  CUDA_CHECK(cudaFree(tester));

  return 0;
}

// vim: et ts=2 sw=2
