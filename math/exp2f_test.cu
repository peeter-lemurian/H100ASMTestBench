//
// exp2f_test.cu
//
// Test harness for 2^x on GPU:
//   - Expect use of PTX ex2 (CUSTOM_EXP2F -- ASM version)
//   - CUDA exp2f
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
inline float __device__ exp2_f32(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = 2^x\n\t"
      "ex2.approx.f32 %0, %1;" // %0 = 2^x
      : "=f"(result) // %0
      : "f"(x)       // %1
      );

  return result;
}
// clang-format on

#define CUSTOM_EXP2F exp2_f32

struct Exp2Case {
  float x;
  const char *label;
};

static constexpr Exp2Case kCases[] = {
    {0.0f, "0"},
    {1.0f, "1"},
    {2.0f, "2"},
    {-1.0f, "-1"},
    {-2.0f, "-2"},
    {0.5f, "0.5"},
    {-0.5f, "-0.5"},
    {10.0f, "10"},
    {-10.0f, "-10"},
    {126.0f, "126"},
    {127.0f, "127"},
    {128.0f, "128 (ovfl)"},
    {-126.0f, "-126 (FLT_MIN)"},
    {-149.0f, "-149 (min sub)"},
    {-150.0f, "-150 (->0)"},
    {std::numeric_limits<float>::infinity(), "+inf"},
    {-std::numeric_limits<float>::infinity(), "-inf"},
    {std::numeric_limits<float>::quiet_NaN(), "NaN"},
    {1.401298e-45f, "min subnorm input"},
    {-1.401298e-45f, "-min subnorm input"},
    {std::numeric_limits<float>::min(), "FLT_MIN input"},
    {-std::numeric_limits<float>::min(), "-FLT_MIN input"},
    {std::numeric_limits<float>::max(), "FLT_MAX input"},
    {-std::numeric_limits<float>::max(), "-FLT_MAX input"},
    {3.14159265f, "pi"},
    {2.71828183f, "e"},
    {1e-7f, "1e-7"},
    {-1e-7f, "-1e-7"},
    {1e-20f, "1e-20"},
    {-1e-20f, "-1e-20"},
    {1e-38f, "1e-38"},
    {-1e-38f, "-1e-38"},
};
static constexpr size_t kNumExp2Cases = sizeof(kCases) / sizeof(kCases[0]);

class Exp2Tester {
public:
  static constexpr size_t N = kNumExp2Cases;
  float input[N];
  float output_asm[N];
  float output_cuda_exp2[N];

  __host__ Exp2Tester() {
    for (size_t i = 0; i < N; i++) {
      input[i] = kCases[i].x;
    }
  }

  void __host__ reset() {
    std::memset(output_asm, 0xff, sizeof(output_asm));
    std::memset(output_cuda_exp2, 0xff, sizeof(output_cuda_exp2));
  }

  void __host__ displayResults(const float *torchinductor,
                               const float *torcheager) const;
};

__global__ void testKernelExp2Asm(Exp2Tester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Exp2Tester::N) {
    self->output_asm[idx] = CUSTOM_EXP2F(self->input[idx]);
  }
}

__global__ void testKernelCudaExp2(Exp2Tester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Exp2Tester::N) {
    self->output_cuda_exp2[idx] = exp2f(self->input[idx]);
  }
}

__global__ void testKernelOneExp2Asm(Exp2Tester *self) {
    self->output_asm[0] = CUSTOM_EXP2F(self->input[0]);
}

__global__ void testKernelOneCudaExp2(Exp2Tester *self) {
    self->output_cuda_exp2[0] = exp2f(self->input[0]);
}

bool verbose{};
bool useColor{};
bool quiet{};

void __host__ Exp2Tester::displayResults(const float *torchinductor,
                                         const float *torcheager) const {
  if (useColor) {
    std::cout << RED;
  }
  std::cout << "POW2: exp2f(x)\n\n\n\n";
  if (useColor) {
    std::cout << RESET;
  }

  std::cout << std::format("{:>4}"     // Idx
                           "{:>16}"    // x
                           "{:>22}"    // Label
                           "{:>16}"    // std::exp2
                           "{:>16}"    // exp2f
                           "{:>16}"    // ASM
                           "{:>16}"    // torch-eager
                           "{:>16}\n", // torch-inductor
                           "Idx", "x", "Label", "std::exp2", "exp2f",
                           "ASM(2^x)",
                           torcheager ? "torch-eager" : "",
                           torchinductor ? "torch-inductor" : "");

  std::cout << std::string(162, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float x = input[i];
    float ref = std::exp2(x);

    OneResult32 v_exp2f(ref, output_cuda_exp2[i], true, verbose);
    OneResult32 v_asm(ref, output_asm[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                      torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                         torchinductor != nullptr, verbose);

    uint32_t rbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::string ref_hex = std::format("0x{:08x}", rbits);

    bool allMatch = v_exp2f.match and v_asm.match and
                    v_inductor.match and v_eager.match;

    if (!quiet or !allMatch) {
      std::cout << std::format("{:>4}"
                               "{:>16g}"
                               "{:>22}"
                               "{:>16.6g}"
                               "{}"
                               "{}"
                               "{}"
                               "{}\n",
                               i, x, kCases[i].label, ref, v_exp2f.value(),
                               v_asm.value(), v_eager.value(),
                               v_inductor.value());
    }

    if (!allMatch) {
      std::string hexline = std::format(
          "{:>4}"
          "{:>16}"
          "{:>22}"
          "{:>16}"
          "{:>16}"
          "{:>16}"
          "{:>16}"
          "{:>16}\n",
          "", "", "", ref_hex, v_exp2f.hexValue(), v_asm.hexValue(),
          v_eager.hexValue(), v_inductor.hexValue());
      std::cout << hexline;

      std::string es_exp2f = v_exp2f.errorString();
      std::string es_asm = v_asm.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();

      const char *color = YELLOW;
      if ((es_exp2f == "ERROR") or (es_asm == "ERROR") or
          (es_eager == "ERROR") or
          (es_inductor == "ERROR")) {
        color = RED;
      }

      if (useColor) {
        std::cout << color;
      }

      std::string matchline =
          std::format("{:>4}"
                      "{:>16}"
                      "{:>22}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}"
                      "{:>16}\n",
                      "", "", "", "", es_exp2f, es_asm,
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
    case 'v':
      verbose = true;
      break;
    case 'q':
      quiet = true;
      break;
    case 'c':
      useColor = true;
      break;
    case 'd':
      dumpFile = optarg;
      break;
    case 'h': {
      std::cout << "exp2f_test"
                   " [--verbose]"
                   " [--quiet]"
                   " [--color]"
                   " [--dump-inputs filename]"
                   " [--torcheager torcheagerexp2.bin]"
                   " [--torchinductor torchinductorexp2.bin]"
                   "\n\n"
                   "./math/exp2f_test --dump-inputs ./exp2test.in\n"
                   "../torch/torchunary.py --op exp2 --file ./exp2test.in\n"
                   "./math/exp2f_test --torchinductor torchinductorexp2.bin --torcheager torcheagerexp2.bin --verbose --quiet --color | less -R\n\n"
                   "\t--dump-inputs filename.  Write input values as binary "
                   "floats to file (x0,x1,x2,...)\n"
                   "\t--verbose.  Show hex values, even if not mismatches\n"
                   "\t--quiet.  Suppress non-matching output\n"
                   "\t--color.  Highlight mismatches in color\n"
                   "\t--help.  Show this output and exit\n";
      return 0;
    }
    case 'y':
      torchinductorFile = optarg;
      break;
    case 't':
      torcheagerFile = optarg;
      break;
    default:
      std::cerr << "exp2f_test: unknown option\n";
      return 1;
    }
  }

  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << std::endl;
      return 2;
    }
    Exp2Tester tmp;
    for (size_t i = 0; i < Exp2Tester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&tmp.input[i]), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input values to " << dumpFile << std::endl;
    return 0;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, Exp2Tester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, Exp2Tester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  Exp2Tester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(Exp2Tester)));

  new (tester) Exp2Tester();

  dim3 blockSize(Exp2Tester::N);
  dim3 gridSize(1);

  tester->reset();

  testKernelExp2Asm<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  testKernelCudaExp2<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  tester->displayResults(torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  CUDA_CHECK(cudaFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
