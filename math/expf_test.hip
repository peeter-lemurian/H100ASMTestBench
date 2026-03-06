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
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <vector>

#include "readbinary.hpp"
#include "OneResult32.hpp"
#include "colors.hpp"
#include "hipcheck.hpp"

// clang-format off
// Proper e^x implementation using v_mul_f32 + v_exp_f32
inline float __device__ exp_f32_scale_as_input(float x) {
  constexpr float log2e = 1.44269504088896340736f; // log2(e)
  float result;

  // no early clobber required, since %1, %2 (inputs) are not ever read
  // after the first %0 output operation.
  __asm__ __volatile__("v_mul_f32 %0, %1, %2 ;; x * log2(e)\n\t"
                       "v_exp_f32 %0, %0     ;; 2^(x * log2(e))\n\t"
                       : "=v"(result) // %0
                       : "v"(x),       // %1
                         "v"(log2e)    // %2
      );
  return result;
}

inline float __device__ exp_f32_scale_with_move(float x) {
  float result;
  __asm__ __volatile__(
      "s_mov_b32 s45, 0x3fb8aa3b ;; Load log2(e) into scalar\n\t"
      "v_mul_f32 %0, %1, s45     ;; x * log2(e)\n\t"
      "v_exp_f32 %0, %0          ;; 2^(x * log2(e))\n\t"
      : "=v"(result) // %0
      : "v"(x)       // %1
      : "s45");
  return result;
}

inline float __device__ exp_f32_inplace_scale(float x) {
  float result;
  __asm__ __volatile__(
      "v_mul_f32_e32 %0, 0x3fb8aa3b, %1     ;; x * log2(e)\n\t"
      "v_exp_f32_e32 %0, %0                 ;; 2^(x * log2(e))\n\t"
      : "=v"(result) // %0
      : "v"(x)       // %1
      );
  return result;
}

inline float __device__ exp_f32_precise(float x) {
  float result;
  float tmp_n, tmp_r, tmp_lo, tmp_thresh;
  __asm__ __volatile__(
      "                                           ;; --- Load constants into vgprs ---\n\t"
      "v_mov_b32_e32  %1, 0x3fb8aa3b              ;; tmp_n    = log2e_hi\n\t"
      "v_mov_b32_e32  %2, 0x32a5705f              ;; tmp_r    = log2e_lo\n\t"
      "                                           ;; --- Argument reduction ---\n\t"
      "v_mul_f32_e32  %0, %4, %1                  ;; result   = x * log2e_hi\n\t"
      "v_rndne_f32_e32 %3, %0                     ;; tmp_lo   = n = round(result)\n\t"
      "v_fma_f32      %1, %4, %1, -%0             ;; tmp_n    = fma(x, log2e_hi, -result) = hi_err\n\t"
      "v_sub_f32_e32  %0, %0, %3                  ;; result   = frac = result - n\n\t"
      "v_fmac_f32_e32 %1, %4, %2                  ;; tmp_n   += x * log2e_lo\n\t"
      "v_add_f32_e32  %1, %0, %1                  ;; tmp_n    = r = frac + err\n\t"
      "                                           ;; --- Core ---\n\t"
      "v_cvt_i32_f32_e32 %3, %3                   ;; tmp_lo   = (int)n\n\t"
      "v_exp_f32_e32  %1, %1                      ;; tmp_n    = 2^r\n\t"
      "                                           ;; --- Clamp setup interleaved with ldexp ---\n\t"
      "v_mov_b32_e32  %2, 0xc2ce8ed0              ;; tmp_r    = underflow_thresh\n\t"
      "v_cmp_nlt_f32_e32 vcc, %4, %2              ;; vcc = (x >= underflow_thresh)\n\t"
      "v_ldexp_f32    %0, %1, %3                  ;; result   = 2^r * 2^n  (fills hazard slot)\n\t"
      "s_nop 0\n\t"
      "v_cndmask_b32_e32 %0, 0, %0, vcc           ;; x < thresh → 0.0\n\t"
      "v_mov_b32_e32  %2, 0x42b17218              ;; tmp_r    = overflow_thresh\n\t"
      "v_cmp_ngt_f32_e32 vcc, %4, %2              ;; vcc = (x <= overflow_thresh)\n\t"
      "v_mov_b32_e32  %2, 0x7f800000              ;; tmp_r    = +inf\n\t"
      "s_nop 1\n\t"
      "v_cndmask_b32_e32 %0, %2, %0, vcc          ;; x > thresh → +inf\n\t"
      : "=&v"(result),                            // %0
        "=&v"(tmp_n),                             // %1
        "=&v"(tmp_r),                             // %2 - reused for lo, thresholds, +inf
        "=&v"(tmp_lo)                             // %3 - n
      : "v"(x)                                    // %4
      : "vcc");
  return result;
}

inline float __device__ exp_f32_precise_stemp(float x) {
  float result;
  float tmp_n, tmp_r, tmp_lo, tmp_thresh;
  uint64_t stemp;
  __asm__ __volatile__(
      "                                           ;; --- Load constants into vgprs ---\n\t"
      "v_mov_b32_e32  %1, 0x3fb8aa3b              ;; tmp_n    = log2e_hi\n\t"
      "v_mov_b32_e32  %2, 0x32a5705f              ;; tmp_r    = log2e_lo\n\t"
      "                                           ;; --- Argument reduction ---\n\t"
      "v_mul_f32_e32  %0, %5, %1                  ;; result   = x * log2e_hi\n\t"
      "v_rndne_f32_e32 %3, %0                     ;; tmp_lo   = n = round(result)\n\t"
      "v_fma_f32      %1, %5, %1, -%0             ;; tmp_n    = fma(x, log2e_hi, -result) = hi_err\n\t"
      "v_sub_f32_e32  %0, %0, %3                  ;; result   = frac = result - n\n\t"
      "v_fmac_f32_e32 %1, %5, %2                  ;; tmp_n   += x * log2e_lo\n\t"
      "v_add_f32_e32  %1, %0, %1                  ;; tmp_n    = r = frac + err\n\t"
      "                                           ;; --- Core ---\n\t"
      "v_cvt_i32_f32_e32 %3, %3                   ;; tmp_lo   = (int)n\n\t"
      "v_exp_f32_e32  %1, %1                      ;; tmp_n    = 2^r\n\t"
      "                                           ;; --- Clamp setup interleaved with ldexp ---\n\t"
      "v_mov_b32_e32  %2, 0xc2ce8ed0              ;; tmp_r    = underflow_thresh\n\t"
      "v_cmp_nlt_f32 %4, %5, %2                   ;; stemp = (x >= underflow_thresh)\n\t"
      "v_ldexp_f32    %0, %1, %3                  ;; result   = 2^r * 2^n  (fills hazard slot)\n\t"
      "s_nop 0\n\t"
      "v_cndmask_b32 %0, 0, %0, %4                ;; x < thresh → 0.0\n\t"
      "v_mov_b32_e32  %2, 0x42b17218              ;; tmp_r    = overflow_thresh\n\t"
      "v_cmp_ngt_f32 %4, %5, %2                   ;; stemp = (x <= overflow_thresh)\n\t"
      "v_mov_b32_e32  %2, 0x7f800000              ;; tmp_r    = +inf\n\t"
      "s_nop 1\n\t"
      "v_cndmask_b32 %0, %2, %0, %4               ;; x > thresh → +inf\n\t"
      : "=&v"(result),                            // %0
        "=&v"(tmp_n),                             // %1
        "=&v"(tmp_r),                             // %2 - reused for lo, thresholds, +inf
        "=&v"(tmp_lo),                            // %3 - n
        "=&s"(stemp)                              // %4
      : "v"(x)                                    // %5
      : /* no clobbers */ );
  return result;
}
// clang-format on

//#define CUSTOM_EXPF exp_f32_scale_as_input
//#define CUSTOM_EXPF exp_f32_scale_with_move
#ifdef DILUVIAN_FAST_MATH
#define CUSTOM_EXPF exp_f32_inplace_scale
#else
//#define CUSTOM_EXPF exp_f32_precise
#define CUSTOM_EXPF exp_f32_precise_stemp
#endif

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
  float output_rocm_exp[N];  // expf() results (ROCm library)
  float output_rocm_fexp[N]; // __expf() results (ROCm library)

  __host__ ExpTester() {
    for (size_t i = 0; i < N; i++) {
      input[i] = kCases[i].x;
    }
  }

  void __host__ reset() {
    std::memset(output_vexp, 0xff, sizeof(output_vexp));
    std::memset(output_exp, 0xff, sizeof(output_exp));
    std::memset(output_rocm_exp, 0xff, sizeof(output_rocm_exp));
    std::memset(output_rocm_fexp, 0xff, sizeof(output_rocm_fexp));
  }

  void __host__ displayResults(const float *torchinductor,
                               const float *torcheager) const;

  __global__ static void testKernelExp(ExpTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_exp[idx] = CUSTOM_EXPF(self->input[idx]);
    }
  }

  __global__ static void testKernelROCmExp(ExpTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_rocm_exp[idx] = expf(self->input[idx]);
    }
  }

  __global__ static void testKernelROCmFExp(ExpTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_rocm_fexp[idx] = __expf(self->input[idx]); // Fast approximate e^x (lower precision)
    }
  }

  // -- kernels for just looking at asm listings ------------------------------

  __global__ static void testKernelOneExp(ExpTester *self) {
    self->output_exp[0] = CUSTOM_EXPF(self->input[0]);
  }

  __global__ static void testKernelOneROCmExp(ExpTester *self) {
    self->output_rocm_exp[0] = expf(self->input[0]);
  }

  __global__ static void testKernelOneROCmFExp(ExpTester *self) {
    self->output_rocm_fexp[0] = __expf(self->input[0]);
  }
};

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

    OneResult32 v_expf(ref, output_rocm_exp[i], true, verbose);
    OneResult32 v__expf(ref, output_rocm_fexp[i], true, verbose);
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
                   "https://download.pytorch.org/whl/rocm6.3\n"
                   "\n\n"
                   "./expf_test --dump-inputs ./exptest.in\n"
                   "../torchexp.py file ./exptest.in\n"
                   "./expf_test --torchinductor torchinductor.bin --torcheager torcheager.bin --verbose --quiet --color | less -R\n\n"
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
  HIP_CHECK(hipMallocManaged(&tester, sizeof(ExpTester)));

  new (tester) ExpTester();

  // Launch kernels
  dim3 blockSize(ExpTester::N);
  dim3 gridSize(1);

  tester->reset();

  // Test CUSTOM_EXPF (e^x via mul+exp)
  hipLaunchKernelGGL(HIP_KERNEL_NAME(ExpTester::testKernelExp), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  // Test ROCm expf()
  hipLaunchKernelGGL(HIP_KERNEL_NAME(ExpTester::testKernelROCmExp), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  // Test ROCm __expf()
  hipLaunchKernelGGL(HIP_KERNEL_NAME(ExpTester::testKernelROCmFExp), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  // Display results (computes CPU reference inline)
  tester->displayResults(torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  // Cleanup
  HIP_CHECK(hipFree(tester));

  return 0;
}

// vim: et ts=2 sw=2
