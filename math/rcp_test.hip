//
// rcp_test.hip
//
// Tests reciprocal using v_rcp_f32 on AMD MI300 hardware,
// comparing against 1.0f / x and optional torch reciprocal outputs.
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
inline float __device__ rcp_f32_inplace(float x) {
  float result;
  __asm__ __volatile__(
      "v_rcp_f32_e32 %0, %1 ;; 1/x\n\t"
      : "=v"(result) // %0
      : "v"(x)       // %1
      );
  return result;
}
// clang-format on

#define CUSTOM_RCPF rcp_f32_inplace

struct RcpCase {
  float x;
  const char *label;
};

static constexpr RcpCase kCases[] = {
    {1.0f, "1"},
    {2.0f, "2"},
    {4.0f, "4"},
    {10.0f, "10"},
    {0.5f, "0.5"},
    {0.25f, "0.25"},
    {-1.0f, "-1"},
    {-2.0f, "-2"},
    {-10.0f, "-10"},
    {3.14159265f, "pi"},
    {2.71828183f, "e"},
    {7.0f, "7"},
    {49.0f, "49"},
    {127.0f, "127"},
    {255.0f, "255"},

    {0.0f, "+0"},
    {-0.0f, "-0"},
    {std::numeric_limits<float>::infinity(), "+inf"},
    {-std::numeric_limits<float>::infinity(), "-inf"},
    {std::numeric_limits<float>::quiet_NaN(), "NaN"},

    {std::numeric_limits<float>::min(), "FLT_MIN"},
    {-std::numeric_limits<float>::min(), "-FLT_MIN"},
    {std::numeric_limits<float>::max(), "FLT_MAX"},
    {-std::numeric_limits<float>::max(), "-FLT_MAX"},
    {std::numeric_limits<float>::denorm_min(), "TRUE_MIN"},
    {-std::numeric_limits<float>::denorm_min(), "-TRUE_MIN"},

    {1e-7f, "1e-7"},
    {-1e-7f, "-1e-7"},
    {1e-20f, "1e-20"},
    {-1e-20f, "-1e-20"},
    {1e-38f, "1e-38"},
    {-1e-38f, "-1e-38"},
};

class RcpTester {
public:
  static constexpr size_t N = sizeof(kCases) / sizeof(kCases[0]);
  float input[N];
  float output_asm[N];
  float output_rocm_div[N];

  __host__ RcpTester() {
    for (size_t i = 0; i < N; i++) {
      input[i] = kCases[i].x;
    }
  }

  void __host__ reset() {
    std::memset(output_asm, 0xff, sizeof(output_asm));
    std::memset(output_rocm_div, 0xff, sizeof(output_rocm_div));
  }

  void __host__ displayResults(const float *torchinductor,
                               const float *torcheager) const;

  __global__ static void testKernelRcpAsm(RcpTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_asm[idx] = CUSTOM_RCPF(self->input[idx]);
    }
  }

  __global__ static void testKernelROCmDiv(RcpTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      self->output_rocm_div[idx] = 1.0f / self->input[idx];
    }
  }

  __global__ static void testKernelOneRcpAsm(RcpTester *self) {
    self->output_asm[0] = CUSTOM_RCPF(self->input[0]);
  }

  __global__ static void testKernelOneROCmDiv(RcpTester *self) {
    self->output_rocm_div[0] = 1.0f / self->input[0];
  }
};

bool verbose{};
bool useColor{};
bool quiet{};

void __host__ RcpTester::displayResults(const float *torchinductor,
                                        const float *torcheager) const {
  if (useColor) {
    std::cout << RED;
  }
  std::cout << "REC: 1/x\n\n\n\n";
  if (useColor) {
    std::cout << RESET;
  }

  std::cout << std::format("{:>4}"     // Idx
                           "{:>16}"    // x
                           "{:>20}"    // Label
                           "{:>16}"    // 1/x (ref)
                           "{:>16}"    // 1.0f/x
                           "{:>16}"    // ASM
                           "{:>16}"    // torch-eager
                           "{:>16}\n", // torch-inductor
                           "Idx", "x", "Label", "1/x(ref)", "1.0f/x",
                           "ASM(rcp)",
                           torcheager ? "torch-eager" : "",
                           torchinductor ? "torch-inductor" : "");

  std::cout << std::string(162, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float x = input[i];
    float ref = 1.0f / x;

    OneResult32 v_div(ref, output_rocm_div[i], true, verbose);
    OneResult32 v_asm(ref, output_asm[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                      torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                         torchinductor != nullptr, verbose);

    uint32_t rbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::string ref_hex = std::format("0x{:08x}", rbits);

    bool allMatch = v_div.match and v_asm.match and
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
                               i, x, kCases[i].label, ref, v_div.value(),
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
          "", "", "", ref_hex, v_div.hexValue(), v_asm.hexValue(),
          v_eager.hexValue(), v_inductor.hexValue());
      std::cout << hexline;

      std::string es_div = v_div.errorString();
      std::string es_asm = v_asm.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();

      const char *color = YELLOW;
      if ((es_div == "ERROR") or (es_asm == "ERROR") or
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
                      "", "", "", "", es_div, es_asm,
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
      std::cout << "rcp_test"
                   " [--verbose]"
                   " [--quiet]"
                   " [--color]"
                   " [--dump-inputs filename]"
                   " [--torcheager torcheagerreciprocal.bin]"
                   " [--torchinductor torchinductorreciprocal.bin]"
                   "\n\n"
                   "Run with:\n"
                   "  ./rcp_test --dump-inputs ./rcptest.in\n"
                   "  ../torchunary.py --op reciprocal --file ./rcptest.in\n"
                   "  ./rcp_test --torchinductor torchinductorreciprocal.bin --torcheager torcheagerreciprocal.bin --verbose --quiet --color | less -R\n\n"
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
      std::cerr << "rcp_test: unknown option\n";
      return 1;
    }
  }

  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << std::endl;
      return 2;
    }
    RcpTester tmp;
    for (size_t i = 0; i < RcpTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&tmp.input[i]), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input values to " << dumpFile << std::endl;
    return 0;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, RcpTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, RcpTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  RcpTester *tester;
  HIP_CHECK(hipMallocManaged(&tester, sizeof(RcpTester)));

  new (tester) RcpTester();

  dim3 blockSize(RcpTester::N);
  dim3 gridSize(1);

  tester->reset();

  hipLaunchKernelGGL(HIP_KERNEL_NAME(RcpTester::testKernelRcpAsm), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  hipLaunchKernelGGL(HIP_KERNEL_NAME(RcpTester::testKernelROCmDiv), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  tester->displayResults(torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  HIP_CHECK(hipFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
