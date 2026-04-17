#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <getopt.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "OneResult32.hpp"
#include "colors.hpp"
#include "hipcheck.hpp"
#include "readbinary.hpp"
#include "custom_asm.hpp"

struct TernaryCase {
  float a;
  float b;
  float c;
  const char *alabel;
  const char *blabel;
  const char *clabel;
};

// clang-format off
static constexpr TernaryCase kCases[] = {
  // ---------------------------------------------------------------------
  // Core sanity
  // ---------------------------------------------------------------------
  {0.0f, 0.0f, 0.0f, "+0", "+0", "+0"},
  {-0.0f, +0.0f, -0.0f, "-0", "+0", "-0"},
  {1.0f, 2.0f, 3.0f, "1", "2", "3"},
  {-1.0f, 2.0f, -3.0f, "-1", "2", "-3"},
  {10.0f, -5.0f, 7.0f, "10", "-5", "7"},
  {-10.0f, -5.0f, 7.0f, "-10", "-5", "7"},

  // ---------------------------------------------------------------------
  // Fractions / rounding-sensitive
  // ---------------------------------------------------------------------
  {0.5f, 0.25f, 0.75f, "0.5", "0.25", "0.75"},
  {-0.5f, 0.25f, -0.75f, "-0.5", "0.25", "-0.75"},
  {1.0f / 3.0f, 3.0f, 1.0f, "1/3", "3", "1"},
  {2.5f, -1.5f, 0.5f, "2.5", "-1.5", "0.5"},

  // ---------------------------------------------------------------------
  // Extremes and specials
  // ---------------------------------------------------------------------
  {std::numeric_limits<float>::min(), 2.0f, -2.0f, "FLT_MIN", "2", "-2"},
  {std::numeric_limits<float>::max(), 0.5f, -0.5f, "FLT_MAX", "0.5", "-0.5"},
  {std::numeric_limits<float>::denorm_min(), 4.0f, -4.0f, "TRUE_MIN", "4", "-4"},
  {std::numeric_limits<float>::infinity(), 1.0f, -1.0f, "+inf", "1", "-1"},
  {-std::numeric_limits<float>::infinity(), 1.0f, 2.0f, "-inf", "1", "2"},
  {std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.0f, "qNaN", "1", "0"},
  {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, "1", "qNaN", "0"},
  {1.0f, 0.0f, std::numeric_limits<float>::quiet_NaN(), "1", "0", "qNaN"},

  // ---------------------------------------------------------------------
  // Ternary-op focused probes (mac / clip / sel)
  // ---------------------------------------------------------------------
  {2.0f, 3.0f, 4.0f, "a=2", "b=3", "c=4"},
  {-2.0f, 3.0f, -4.0f, "a=-2", "b=3", "c=-4"},
  {5.0f, 1.0f, -1.0f, "a=5", "u=1", "d=-1"},
  {-5.0f, 1.0f, -1.0f, "a=-5", "u=1", "d=-1"},
  {1.0f, 42.0f, -42.0f, "s=1", "a=42", "b=-42"},
  {0.0f, 42.0f, -42.0f, "s=0", "a=42", "b=-42"},
  {-1.0f, 42.0f, -42.0f, "s=-1", "a=42", "b=-42"},
};
// clang-format on

static constexpr size_t kNumTernaryCases = sizeof(kCases) / sizeof(kCases[0]);

enum class TernaryOp {
  Unknown,
  Mac,
  Clip,
  Sel,
};

static const char *opName(TernaryOp op) {
  switch (op) {
  case TernaryOp::Mac:
    return "mac";
  case TernaryOp::Clip:
    return "clip";
  case TernaryOp::Sel:
    return "sel";
  case TernaryOp::Unknown:
    return "unknown";
  }
  return "?";
}

class TernaryTester {
public:
  static constexpr size_t N = kNumTernaryCases;

  float input_a[N];
  float input_b[N];
  float input_c[N];
  float out_rocm[N];
  float out_custom[N];

  __host__ TernaryTester() {
    for (size_t i = 0; i < N; i++) {
      input_a[i] = kCases[i].a;
      input_b[i] = kCases[i].b;
      input_c[i] = kCases[i].c;
    }
  }

  __host__ void reset() {
    std::memset(out_rocm, 0xff, sizeof(out_rocm));
    std::memset(out_custom, 0xff, sizeof(out_custom));
  }

  // TODO(peeter): Replace these TODO implementations with real ROCm/custom
  // ternary operations for sel.
  __device__ static float todoRocm(float a, float b, float c) {
    (void)a;
    (void)b;
    (void)c;
    return nanf("");
  }
  __device__ static float todoCustom(float a, float b, float c) {
    (void)a;
    (void)b;
    (void)c;
    return nanf("");
  }

  __global__ static void testMacRocm(TernaryTester *self) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
      self->out_rocm[i] = fmaf(self->input_a[i], self->input_b[i], self->input_c[i]);
  }
  __global__ static void testMacCustom(TernaryTester *self) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
      self->out_custom[i] =
          custom_macf(self->input_a[i], self->input_b[i], self->input_c[i]);
  }

  __global__ static void testClipRocm(TernaryTester *self) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
      self->out_rocm[i] = fmaxf(fminf(self->input_a[i], self->input_b[i]), self->input_c[i]);
  }
  __global__ static void testClipCustom(TernaryTester *self) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
      self->out_custom[i] =
          custom_clipf(self->input_a[i], self->input_b[i], self->input_c[i]);
  }

  __global__ static void testSelRocm(TernaryTester *self) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
      self->out_rocm[i] = (self->input_a[i] != 0.0f) ? self->input_b[i] : self->input_c[i];
  }
  __global__ static void testSelCustom(TernaryTester *self) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
      self->out_custom[i] = custom_sel(
          static_cast<uint32_t>(self->input_a[i] != 0.0f),
          self->input_b[i], self->input_c[i]);
  }

  // -- One-value kernels for asm inspection --------------------------------
  __global__ static void testOneMacRocm(TernaryTester *self) {
    self->out_rocm[0] = fmaf(self->input_a[0], self->input_b[0], self->input_c[0]);
  }
  __global__ static void testOneMacCustom(TernaryTester *self) {
    self->out_custom[0] =
        custom_macf(self->input_a[0], self->input_b[0], self->input_c[0]);
  }

  __global__ static void testOneClipRocm(TernaryTester *self) {
    self->out_rocm[0] =
        fmaxf(fminf(self->input_a[0], self->input_b[0]), self->input_c[0]);
  }
  __global__ static void testOneClipCustom(TernaryTester *self) {
    self->out_custom[0] =
        custom_clipf(self->input_a[0], self->input_b[0], self->input_c[0]);
  }

  __global__ static void testOneSelRocm(TernaryTester *self) {
    self->out_rocm[0] =
        (self->input_a[0] != 0.0f) ? self->input_b[0] : self->input_c[0];
  }
  __global__ static void testOneSelCustom(TernaryTester *self) {
    self->out_custom[0] = custom_sel(
        static_cast<uint32_t>(self->input_a[0] != 0.0f), self->input_b[0],
        self->input_c[0]);
  }
};

bool verbose{};
bool useColor{};
bool quiet{};
bool csvOutput{};

static std::string fp32Hex(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return std::format("0x{:08x}", bits);
}

static void displayResults(const TernaryTester &t, TernaryOp op, const float *torcheager,
                           const float *torchinductor) {
  const char *name = opName(op);

  if (csvOutput) {
    std::cout << std::format(
        "op,idx,a,b,c,alabel,blabel,clabel,row,{0}(rocm),{0}(custom),torch_eager,"
        "torch_inductor\n",
        name);

    for (size_t i = 0; i < TernaryTester::N; i++) {
      float a = t.input_a[i];
      float b = t.input_b[i];
      float c = t.input_c[i];
      float ref = t.out_rocm[i];

      OneResultFloat32 v_custom(ref, t.out_custom[i], true, verbose);
      OneResultFloat32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                               torcheager != nullptr, verbose);
      OneResultFloat32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                                  torchinductor != nullptr, verbose);

      bool allMatch = v_custom.match && v_eager.match && v_inductor.match;

      if (!quiet || !allMatch) {
        std::cout << std::format(
            "{},{},{:g},{:g},{:g},\"{}\",\"{}\",\"{}\",VALUE,{:g},{:g},{},{}\n", name,
            i, a, b, c, kCases[i].alabel, kCases[i].blabel, kCases[i].clabel, ref,
            t.out_custom[i], torcheager ? std::format("{:g}", torcheager[i]) : "",
            torchinductor ? std::format("{:g}", torchinductor[i]) : "");
      }
      if (!allMatch) {
        std::cout << std::format("{},{},,,,,,,,HEX,{},{},{},{}\n", name, i, fp32Hex(ref),
                                 v_custom.hexValue(), v_eager.hexValue(),
                                 v_inductor.hexValue());
        std::cout << std::format("{},{},,,,,,,,DIFF,,{},{},{}\n", name, i,
                                 v_custom.errorString(), v_eager.errorString(),
                                 v_inductor.errorString());
      }
    }
    return;
  }

  if (useColor)
    std::cout << RED;
  std::cout << std::format("TRINARY OP: {}\n\n", name);
  if (useColor)
    std::cout << RESET;

  std::cout << std::format("{:>4}{:>14}{:>14}{:>14}{:>10}{:>10}{:>10}{:>16}{:>16}{:>16}{:>16}\n", "Idx",
                           "a", "b", "c", "a lbl", "b lbl", "c lbl",
                           std::format("{}(rocm)", name), std::format("{}(custom)", name),
                           torcheager ? "torch-eager" : "", torchinductor ? "torch-inductor" : "");
  std::cout << std::string(160, '-') << "\n";

  for (size_t i = 0; i < TernaryTester::N; i++) {
    float a = t.input_a[i];
    float b = t.input_b[i];
    float c = t.input_c[i];
    float ref = t.out_rocm[i];

    OneResultFloat32 v_custom(ref, t.out_custom[i], true, verbose);
    OneResultFloat32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                             torcheager != nullptr, verbose);
    OneResultFloat32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                                torchinductor != nullptr, verbose);

    bool allMatch = v_custom.match && v_eager.match && v_inductor.match;

    if (!quiet || !allMatch) {
      std::cout << std::format(
          "{:>4}{:>14g}{:>14g}{:>14g}{:>10}{:>10}{:>10}{:>16.6g}{}{}{}\n", i, a, b, c,
          kCases[i].alabel, kCases[i].blabel, kCases[i].clabel, ref, v_custom.value(),
          v_eager.value(), v_inductor.value());
    }

    if (verbose || !allMatch) {
      std::cout << std::format(
          "{:>4}{:>14}{:>14}{:>14}{:>10}{:>10}{:>10}{:>16}{:>16}{:>16}{:>16}\n", "",
          fp32Hex(a), fp32Hex(b), fp32Hex(c), "", "", "", fp32Hex(ref),
          v_custom.hexValue(), v_eager.hexValue(), v_inductor.hexValue());
    }

    if (!allMatch) {
      const char *color = YELLOW;
      std::string es_custom = v_custom.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();
      if ((es_custom == "ERROR") || (es_eager == "ERROR") ||
          (es_inductor == "ERROR"))
        color = RED;
      if (useColor)
        std::cout << color;

      std::cout << std::format(
          "{:>4}{:>14}{:>14}{:>14}{:>10}{:>10}{:>10}{:>16}{:>16}{:>16}{:>16}\n", "", "",
          "", "", "", "", "", "", es_custom, es_eager, es_inductor);
      if (useColor)
        std::cout << RESET;
    }
  }
}

using KernelFn = void (*)(TernaryTester *);

struct KernelPair {
  KernelFn rocm;
  KernelFn custom;
};

static KernelPair kernelsForOp(TernaryOp op) {
  switch (op) {
  case TernaryOp::Mac:
    return {TernaryTester::testMacRocm, TernaryTester::testMacCustom};
  case TernaryOp::Clip:
    return {TernaryTester::testClipRocm, TernaryTester::testClipCustom};
  case TernaryOp::Sel:
    return {TernaryTester::testSelRocm, TernaryTester::testSelCustom};
  case TernaryOp::Unknown:
    return {nullptr, nullptr};
  }
  return {nullptr, nullptr};
}

static TernaryOp parseOpName(const std::string &name) {
  if (name == "mac")
    return TernaryOp::Mac;
  if (name == "clip")
    return TernaryOp::Clip;
  if (name == "sel")
    return TernaryOp::Sel;
  return TernaryOp::Unknown;
}

int main(int argc, char **argv) {
  int c{};
  const char *dumpFile{};
  const char *torchinductorFile{};
  const char *torcheagerFile{};
  TernaryOp selectedOp = TernaryOp::Unknown;

  enum class Options {
    op,
    mac,
    clip,
    sel,
    help,
    verbose,
    quiet,
    color,
    csv,
    dumpinputs,
    torcheager,
    torchinductor,
  };

  constexpr struct option longOptions[] = {
      {"op", 1, nullptr, (int)Options::op},
      {"mac", 0, nullptr, (int)Options::mac},
      {"clip", 0, nullptr, (int)Options::clip},
      {"sel", 0, nullptr, (int)Options::sel},
      {"help", 0, nullptr, (int)Options::help},
      {"verbose", 0, nullptr, (int)Options::verbose},
      {"quiet", 0, nullptr, (int)Options::quiet},
      {"color", 0, nullptr, (int)Options::color},
      {"csv", 0, nullptr, (int)Options::csv},
      {"dump-inputs", 1, nullptr, (int)Options::dumpinputs},
      {"torcheager", 1, nullptr, (int)Options::torcheager},
      {"torchinductor", 1, nullptr, (int)Options::torchinductor},
      {nullptr, 0, nullptr, 0},
  };

  while (-1 != (c = getopt_long(argc, argv, "", longOptions, nullptr))) {
    switch ((Options)c) {
    case Options::op:
      selectedOp = parseOpName(optarg ? optarg : "");
      break;
    case Options::mac:
      selectedOp = TernaryOp::Mac;
      break;
    case Options::clip:
      selectedOp = TernaryOp::Clip;
      break;
    case Options::sel:
      selectedOp = TernaryOp::Sel;
      break;
    case Options::verbose:
      verbose = true;
      break;
    case Options::quiet:
      quiet = true;
      break;
    case Options::color:
      useColor = true;
      break;
    case Options::csv:
      csvOutput = true;
      break;
    case Options::dumpinputs:
      dumpFile = optarg;
      break;
    case Options::torcheager:
      torcheagerFile = optarg;
      break;
    case Options::torchinductor:
      torchinductorFile = optarg;
      break;
    case Options::help:
      std::cout << "ternary_test"
                   " --[mac|clip|sel]"
                   " [--op name]"
                   " [--verbose]"
                   " [--quiet]"
                   " [--color]"
                   " [--csv]"
                   " [--dump-inputs filename]"
                   " [--torcheager file.bin]"
                   " [--torchinductor file.bin]\n\n"
                   "Run with:\n"
                   "  complexops/ternary_test --dump-inputs ./ternarytest.in\n"
                   "  ../bin/torchternary.py --op mac --file ./ternarytest.in\n"
                   "  complexops/ternary_test --mac --torcheager torcheagermac.bin "
                   "--torchinductor torchinductormac.bin --verbose --quiet --color | less -R\n\n"
                   "\t--OP.         (required) one of: mac, clip, sel\n"
                   "\t--op name.    alternative way to select the operation\n"
                   "\t--verbose.    Show hex values\n"
                   "\t--quiet.      Suppress VALUE rows\n"
                   "\t--color.      Colorize the title\n"
                   "\t--csv.        Emit CSV output\n"
                   "\t--dump-inputs Write interleaved (a,b,c) float triplets as binary to file\n"
                   "\t--torcheager  Binary float file with torch eager results\n"
                   "\t--torchinductor Binary float file with torch inductor results\n"
                   "\t--help.       Show this output and exit\n";
      return 0;
    default:
      std::cerr << "ternary_test: unknown option\n";
      return 1;
    }
  }

  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << std::endl;
      return 2;
    }
    TernaryTester tmp;
    for (size_t i = 0; i < TernaryTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&tmp.input_a[i]), sizeof(float));
      ofs.write(reinterpret_cast<const char *>(&tmp.input_b[i]), sizeof(float));
      ofs.write(reinterpret_cast<const char *>(&tmp.input_c[i]), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote interleaved (a,b,c) input values to " << dumpFile << std::endl;
    return 0;
  }

  if (selectedOp == TernaryOp::Unknown) {
    std::cerr << "ternary_test: --[op] is required\n";
    return 1;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFile<float>(torchinductorFile, TernaryTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFile<float>(torcheagerFile, TernaryTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  TernaryTester *tester;
  HIP_CHECK(hipMallocManaged(&tester, sizeof(TernaryTester)));
  new (tester) TernaryTester();

  constexpr size_t kThreads = 256;
  dim3 blockSize(kThreads);
  dim3 gridSize((TernaryTester::N + kThreads - 1) / kThreads);

  tester->reset();

  auto [rocmKernel, customKernel] = kernelsForOp(selectedOp);
  if (!rocmKernel || !customKernel) {
    std::cerr << "ternary_test: no kernels for selected operation\n";
    HIP_CHECK(hipFree(tester));
    return 1;
  }

  hipLaunchKernelGGL(rocmKernel, gridSize, blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());
  hipLaunchKernelGGL(customKernel, gridSize, blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  displayResults(*tester, selectedOp, torcheagerFile ? torcheagerOut.data() : nullptr,
                 torchinductorFile ? torchinductorOut.data() : nullptr);

  HIP_CHECK(hipFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
