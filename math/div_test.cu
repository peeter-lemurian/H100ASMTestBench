//
// div_test.cu
//
// Tests division using rcp + mul on NVIDIA H100 hardware,
// comparing results against standard division (/) computed on the CPU.
//
// The intent is to:
//   1. Establish ground truth via CUDA's operator/ and fdividef().
//   2. Test hand-coded PTX inline-asm division
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

#include "OneResult32.hpp"
#include "colors.hpp"
#include "cuda_check.hpp"
#include "readbinary.hpp"

// ---------------------------------------------------------------------------
// Custom division: ASM version
// ---------------------------------------------------------------------------
// clang-format off
inline float __device__ custom_div(float a, float b) {
  float result;

  __asm__ __volatile__(
      "// %0 = a/b\n\t"
      "div.rn.f32 %0, %1, %2;"
      : "=f"(result) // %0
      : "f"(a),      // %1
        "f"(b)       // %2
      );

  return result;
}
// clang-format on

#define CUSTOM_DIV custom_div

// ---------------------------------------------------------------------------
// Test-case table
// ---------------------------------------------------------------------------
struct DivCase {
  float a;
  float b;
  const char *label;
};

// Keep in sync with DivTester::N below.
static const DivCase kCases[] = {

    // -----------------------------------------------------------------------
    // Basic exact results
    // -----------------------------------------------------------------------
    {1.0f, 1.0f, "1/1"},             //   0: = 1
    {4.0f, 2.0f, "4/2"},             //   1: = 2
    {10.0f, 5.0f, "10/5"},           //   2: = 2
    {100.0f, 10.0f, "100/10"},       //   3: = 10
    {1.0f, 2.0f, "1/2"},             //   4: = 0.5
    {1.0f, 4.0f, "1/4"},             //   5: = 0.25
    {3.0f, 4.0f, "3/4"},             //   6: = 0.75
    {1.0f, 8.0f, "1/8"},             //   7: = 0.125
    {1.0f, 16.0f, "1/16"},           //   8: = 0.0625
    {16.0f, 4.0f, "16/4"},           //   9: = 4

    // -----------------------------------------------------------------------
    // Self-division (should be 1)
    // -----------------------------------------------------------------------
    {7.0f, 7.0f, "7/7"},             //  10: = 1
    {0.123f, 0.123f, "0.123/0.123"}, //  11: = 1

    // -----------------------------------------------------------------------
    // Division by 1 and trivial scaling
    // -----------------------------------------------------------------------
    {100.0f, 1.0f, "100/1"},         //  12: = 100
    {0.001f, 1.0f, "0.001/1"},       //  13: = 0.001
    {1.0f, 10.0f, "1/10"},           //  14: = 0.1
    {1.0f, 100.0f, "1/100"},         //  15: = 0.01
    {1.0f, 0.5f, "1/0.5"},           //  16: = 2
    {1.0f, 0.1f, "1/0.1"},           //  17: = 10

    // -----------------------------------------------------------------------
    // Non-integer results (denominators with no exact float reciprocal)
    // -----------------------------------------------------------------------
    {1.0f, 3.0f, "1/3"},             //  18: = 0.333...
    {2.0f, 3.0f, "2/3"},             //  19: = 0.666...
    {1.0f, 6.0f, "1/6"},             //  20: = 0.1666...
    {1.0f, 7.0f, "1/7"},             //  21: ≈ 0.142857
    {1.0f, 9.0f, "1/9"},             //  22: = 0.111...
    {1.0f, 11.0f, "1/11"},           //  23: ≈ 0.0909
    {1.0f, 12.0f, "1/12"},           //  24: = 0.0833...
    {1.0f, 13.0f, "1/13"},           //  25: ≈ 0.0769
    {1.0f, 14.0f, "1/14"},           //  26: ≈ 0.0714
    {1.0f, 15.0f, "1/15"},           //  27: = 0.0666...
    {1.0f, 17.0f, "1/17"},           //  28: ≈ 0.0588
    {1.0f, 49.0f, "1/49"},           //  29: ≈ 0.0204
    {1.0f, 51.0f, "1/51"},           //  30: ≈ 0.0196
    {1.0f, 97.0f, "1/97"},           //  31: ≈ 0.0103
    {1.0f, 101.0f, "1/101"},         //  32: ≈ 0.0099
    {1.0f, 127.0f, "1/127"},         //  33: ≈ 0.00787
    {1.0f, 255.0f, "1/255"},         //  34: ≈ 0.00392

    // Non-trivial numerators with non-trivial denominators
    {2.0f, 7.0f, "2/7"},             //  35: ≈ 0.2857
    {3.0f, 7.0f, "3/7"},             //  36: ≈ 0.4286
    {4.0f, 7.0f, "4/7"},             //  37: ≈ 0.5714
    {5.0f, 7.0f, "5/7"},             //  38: ≈ 0.7143
    {6.0f, 7.0f, "6/7"},             //  39: ≈ 0.8571
    {2.0f, 15.0f, "2/15"},           //  40: ≈ 0.1333
    {4.0f, 15.0f, "4/15"},           //  41: ≈ 0.2667
    {7.0f, 3.0f, "7/3"},             //  42: = 2.333...
    {10.0f, 3.0f, "10/3"},           //  43: = 3.333...
    {22.0f, 7.0f, "22/7"},           //  44: ≈ π
    {0.1f, 0.3f, "0.1/0.3"},         //  45: ≈ 0.333

    // Near-power-of-2 denominators (long mantissa reciprocals)
    {1.0f, 1.5f, "1/1.5"},           //  46: = 0.666...
    {1.0f, 2.5f, "1/2.5"},           //  47: = 0.4
    {1.0f, 3.5f, "1/3.5"},           //  48: ≈ 0.2857
    {1.0f, 5.5f, "1/5.5"},           //  49: ≈ 0.1818
    {1.0f, 6.5f, "1/6.5"},           //  50: ≈ 0.1538
    {1.0f, 7.5f, "1/7.5"},           //  51: ≈ 0.1333
    {3.0f, 5.5f, "3/5.5"},           //  52: ≈ 0.5455
    {5.0f, 6.5f, "5/6.5"},           //  53: ≈ 0.7692

    // Large numerator amplifies rcp error (result stays normal)
    {1000.0f, 7.0f, "1000/7"},       //  54: ≈ 142.857
    {1000.0f, 9.0f, "1000/9"},       //  55: ≈ 111.111
    {1000.0f, 11.0f, "1000/11"},     //  56: ≈ 90.909
    {1000.0f, 13.0f, "1000/13"},     //  57: ≈ 76.923
    {123456.0f, 789.0f, "123456/789"}, // 58: ≈ 156.47

    // Low-bit stress: result near float midpoint
    {355.0f, 113.0f, "355/113"},         //  59: ≈ π (better)
    {103993.0f, 33102.0f, "103993/33102"}, // 60: ≈ π (best single-precision)

    // -----------------------------------------------------------------------
    // Negative operands
    // -----------------------------------------------------------------------
    {-1.0f, 3.0f, "-1/3"},           //  61: ≈ -0.333
    {-10.0f, 2.0f, "-10/2"},         //  62: = -5
    {10.0f, -2.0f, "10/-2"},         //  63: = -5
    {-10.0f, -2.0f, "-10/-2"},       //  64: = 5

    // -----------------------------------------------------------------------
    // Zero numerator / zero denominator
    // -----------------------------------------------------------------------
    {0.0f, 1.0f, "0/1"},             //  65: = 0
    {0.0f, 10.0f, "0/10"},           //  66: = 0
    {0.0f, -5.0f, "0/-5"},           //  67: = -0
    {-0.0f, 1.0f, "-0/1"},           //  68: = -0
    {1.0f, 0.0f, "1/0"},             //  69: = +inf
    {-1.0f, 0.0f, "-1/0"},           //  70: = -inf
    {10.0f, 0.0f, "10/0"},           //  71: = +inf
    {0.0f, 0.0f, "0/0"},             //  72: = NaN
    {1.0f, -0.0f, "1/-0"},           //  73: = -inf
    {-1.0f, -0.0f, "-1/-0"},         //  74: = +inf
    {0.0f, -0.0f, "0/-0"},           //  75: = NaN
    {-0.0f, -0.0f, "-0/-0"},         //  76: = NaN

    // -----------------------------------------------------------------------
    // Infinities
    // -----------------------------------------------------------------------
    {1.0f, std::numeric_limits<float>::infinity(), "1/+inf"},            //  77: = 0
    {-1.0f, std::numeric_limits<float>::infinity(), "-1/+inf"},          //  78: = -0
    {1.0f, -std::numeric_limits<float>::infinity(), "1/-inf"},           //  79: = -0
    {-1.0f, -std::numeric_limits<float>::infinity(), "-1/-inf"},         //  80: = +0
    {std::numeric_limits<float>::infinity(), 1.0f, "+inf/1"},            //  81: = +inf
    {-std::numeric_limits<float>::infinity(), 1.0f, "-inf/1"},           //  82: = -inf
    {-std::numeric_limits<float>::infinity(), -1.0f, "-inf/-1"},         //  83: = +inf
    {std::numeric_limits<float>::infinity(), 0.0f, "+inf/0"},            //  84: = +inf
    {-std::numeric_limits<float>::infinity(), 0.0f, "-inf/0"},           //  85: = -inf
    {std::numeric_limits<float>::infinity(), -0.0f, "+inf/-0"},          //  86: = -inf
    {-std::numeric_limits<float>::infinity(), -0.0f, "-inf/-0"},         //  87: = +inf
    {std::numeric_limits<float>::infinity(),
     std::numeric_limits<float>::infinity(), "+inf/+inf"},               //  88: = NaN
    {-std::numeric_limits<float>::infinity(),
     std::numeric_limits<float>::infinity(), "-inf/+inf"},               //  89: = NaN
    {-std::numeric_limits<float>::infinity(),
     -std::numeric_limits<float>::infinity(), "-inf/-inf"},              //  90: = NaN

    // -----------------------------------------------------------------------
    // NaN
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::quiet_NaN(), 1.0f, "NaN/1"},            //  91: = NaN
    {1.0f, std::numeric_limits<float>::quiet_NaN(), "1/NaN"},            //  92: = NaN
    {std::numeric_limits<float>::quiet_NaN(),
     std::numeric_limits<float>::quiet_NaN(), "NaN/NaN"},                //  93: = NaN
    {-std::numeric_limits<float>::quiet_NaN(), 1.0f, "-NaN/1"},          //  94: = NaN
    {1.0f, -std::numeric_limits<float>::quiet_NaN(), "1/-NaN"},          //  95: = NaN
    {std::numeric_limits<float>::signaling_NaN(), 1.0f, "sNaN/1"},       //  96: = NaN
    {1.0f, std::numeric_limits<float>::signaling_NaN(), "1/sNaN"},       //  97: = NaN
    {-std::numeric_limits<float>::signaling_NaN(), 1.0f, "-sNaN/1"},     //  98: = NaN

    // -----------------------------------------------------------------------
    // Near-1 ULP neighbourhood
    // -----------------------------------------------------------------------
    {std::nextafterf(1.0f, 0.0f), 1.0f, "(1-ulp)/1"}, //  99: just below 1
    {std::nextafterf(1.0f, 2.0f), 1.0f, "(1+ulp)/1"}, // 100: just above 1
    {1.0f, std::nextafterf(1.0f, 0.0f), "1/(1-ulp)"}, // 101: just above 1
    {1.0f, std::nextafterf(1.0f, 2.0f), "1/(1+ulp)"}, // 102: just below 1

    // -----------------------------------------------------------------------
    // Subnormals
    // -----------------------------------------------------------------------
    {1.0e-40f, 1.0f, "1e-40/1"},                       // 103: subnormal/normal
    {1.0f, 1.0e-40f, "1/1e-40"},                       // 104: normal/subnormal
    {1.0e-40f, 1.0e-40f, "1e-40/1e-40"},               // 105: sub/sub = 1
    {1.0e-40f, 2.0f, "1e-40/2"},                       // 106: subnormal/2
    {std::numeric_limits<float>::denorm_min(), 1.0f,
     "FLT_TRUE_MIN/1"},                                // 107: smallest sub / 1
    {1.0f, std::numeric_limits<float>::denorm_min(),
     "1/FLT_TRUE_MIN"},                                // 108: 1 / smallest sub
    {std::numeric_limits<float>::denorm_min(),
     std::numeric_limits<float>::denorm_min(),
     "TRUE_MIN/TRUE_MIN"},                             // 109: = 1

    // -----------------------------------------------------------------------
    // FLT_MIN boundary: smallest normal float
    // FLT_MIN = 1.17549e-38 = 0x00800000
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::min(), 1.0f, "FLT_MIN/1"},              // 110
    {1.0f, std::numeric_limits<float>::min(), "1/FLT_MIN"},              // 111

    // FLT_MIN * small integer (still small normals, non-trivial mantissa)
    {std::numeric_limits<float>::min() * 1.5f, 1.0f, "(FLT_MIN*1.5)/1"}, // 112
    {1.0f, std::numeric_limits<float>::min() * 1.5f, "1/(FLT_MIN*1.5)"}, // 113
    {std::numeric_limits<float>::min() * 2.0f, 1.0f, "(FLT_MIN*2)/1"},   // 114
    {1.0f, std::numeric_limits<float>::min() * 2.0f, "1/(FLT_MIN*2)"},   // 115
    {std::numeric_limits<float>::min() * 3.0f, 1.0f, "(FLT_MIN*3)/1"},   // 116
    {1.0f, std::numeric_limits<float>::min() * 3.0f, "1/(FLT_MIN*3)"},   // 117
    {std::numeric_limits<float>::min() * 4.0f, 1.0f, "(FLT_MIN*4)/1"},   // 118
    {1.0f, std::numeric_limits<float>::min() * 4.0f, "1/(FLT_MIN*4)"},   // 119
    {std::numeric_limits<float>::min() * 7.0f, 1.0f, "(FLT_MIN*7)/1"},   // 120
    {1.0f, std::numeric_limits<float>::min() * 7.0f, "1/(FLT_MIN*7)"},   // 121
    {std::numeric_limits<float>::min() * 8.0f, 1.0f, "(FLT_MIN*8)/1"},   // 122
    {1.0f, std::numeric_limits<float>::min() * 8.0f, "1/(FLT_MIN*8)"},   // 123
    {std::numeric_limits<float>::min() * 16.0f, 1.0f, "(FLT_MIN*16)/1"}, // 124
    {1.0f, std::numeric_limits<float>::min() * 16.0f, "1/(FLT_MIN*16)"}, // 125

    // small/small pairs (quotient is normal)
    {std::numeric_limits<float>::min() * 2.0f,
     std::numeric_limits<float>::min(),
     "(FLT_MIN*2)/(FLT_MIN)"},                                           // 126: = 2
    {std::numeric_limits<float>::min() * 6.0f,
     std::numeric_limits<float>::min() * 2.0f,
     "(FLT_MIN*6)/(FLT_MIN*2)"},                                         // 127: = 3
    {std::numeric_limits<float>::min() * 3.0f,
     std::numeric_limits<float>::min() * 7.0f,
     "(FLT_MIN*3)/(FLT_MIN*7)"},                                         // 128: = 3/7

    // FLT_MIN / N: subnormal territory, rcp overflows
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 2.5f,
     "(FLT_MIN/1.5)/(FLT_MIN/2.5)"},                                     // 129: = 5/3
    {std::numeric_limits<float>::min() / 2.0f,
     std::numeric_limits<float>::min() / 2.0f,
     "(FLT_MIN/2)/(FLT_MIN/2)"},                                         // 130: = 1
    {std::numeric_limits<float>::min() / 2.0f,
     std::numeric_limits<float>::min() / 4.0f,
     "(FLT_MIN/2)/(FLT_MIN/4)"},                                         // 131: = 2
    {std::numeric_limits<float>::min() / 3.0f,
     std::numeric_limits<float>::min() / 7.0f,
     "(FLT_MIN/3)/(FLT_MIN/7)"},                                         // 132: = 7/3
    {std::numeric_limits<float>::min() / 4.0f,
     std::numeric_limits<float>::min() / 6.0f,
     "(FLT_MIN/4)/(FLT_MIN/6)"},                                         // 133: = 3/2
    {std::numeric_limits<float>::min() / 5.0f,
     std::numeric_limits<float>::min() / 3.0f,
     "(FLT_MIN/5)/(FLT_MIN/3)"},                                         // 134: = 3/5
    {std::numeric_limits<float>::min() / 7.0f,
     std::numeric_limits<float>::min() / 2.0f,
     "(FLT_MIN/7)/(FLT_MIN/2)"},                                         // 135: = 2/7
    {std::numeric_limits<float>::min() / 8.0f,
     std::numeric_limits<float>::min() / 8.0f,
     "(FLT_MIN/8)/(FLT_MIN/8)"},                                         // 136: = 1
    {std::numeric_limits<float>::min() / 6.0f,
     std::numeric_limits<float>::min() / 16.0f,
     "(FLT_MIN/6)/(FLT_MIN/16)"},                                        // 137: = 8/3
    {std::numeric_limits<float>::min() / 11.0f,
     std::numeric_limits<float>::min() / 13.0f,
     "(FLT_MIN/11)/(FLT_MIN/13)"},                                       // 138: = 13/11
    {std::numeric_limits<float>::min() / 16.0f,
     std::numeric_limits<float>::min() / 32.0f,
     "(FLT_MIN/16)/(FLT_MIN/32)"},                                       // 139: = 2
    {std::numeric_limits<float>::min() / 32.0f,
     std::numeric_limits<float>::min() / 64.0f,
     "(FLT_MIN/32)/(FLT_MIN/64)"},                                       // 140: = 2

    // Probe: a = FLT_MIN/1.5, b grows — how far before rcp(b) overflows?
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f,
     "(a=FLT_MIN/1.5)/(a)"},                                             // 141: = 1
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 2.0f,
     "(a=FLT_MIN/1.5)/(a*2)"},                                           // 142
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 4.0f,
     "(a=FLT_MIN/1.5)/(a*4)"},                                           // 143
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 8.0f,
     "(a=FLT_MIN/1.5)/(a*8)"},                                           // 144
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 16.0f,
     "(a=FLT_MIN/1.5)/(a*16)"},                                          // 145
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 32.0f,
     "(a=FLT_MIN/1.5)/(a*32)"},                                          // 146
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 64.0f,
     "(a=FLT_MIN/1.5)/(a*64)"},                                          // 147
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 128.0f,
     "(a=FLT_MIN/1.5)/(a*128)"},                                         // 148
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 256.0f,
     "(a=FLT_MIN/1.5)/(a*256)"},                                         // 149
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 512.0f,
     "(a=FLT_MIN/1.5)/(a*512)"},                                         // 150
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 1024.0f,
     "(a=FLT_MIN/1.5)/(a*1024)"},                                        // 151
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 3.0f,
     "(a=FLT_MIN/1.5)/(a*3)"},                                           // 152
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 6.0f,
     "(a=FLT_MIN/1.5)/(a*6)"},                                           // 153
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 12.0f,
     "(a=FLT_MIN/1.5)/(a*12)"},                                          // 154
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 24.0f,
     "(a=FLT_MIN/1.5)/(a*24)"},                                          // 155
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 48.0f,
     "(a=FLT_MIN/1.5)/(a*48)"},                                          // 156
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 96.0f,
     "(a=FLT_MIN/1.5)/(a*96)"},                                          // 157
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 192.0f,
     "(a=FLT_MIN/1.5)/(a*192)"},                                         // 158
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 384.0f,
     "(a=FLT_MIN/1.5)/(a*384)"},                                         // 159
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 768.0f,
     "(a=FLT_MIN/1.5)/(a*768)"},                                         // 160
    {std::numeric_limits<float>::min() / 1.5f,
     std::numeric_limits<float>::min() / 1.5f * 1536.0f,
     "(a=FLT_MIN/1.5)/(a*1536)"},                                        // 161

    // -----------------------------------------------------------------------
    // FLT_MAX boundary: largest normal float
    // FLT_MAX = 3.40282e+38 = 0x7f7fffff
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::max(), 1.0f, "FLT_MAX/1"},              // 162
    {1.0f, std::numeric_limits<float>::max(), "1/FLT_MAX"},              // 163
    {std::numeric_limits<float>::max(),
     std::numeric_limits<float>::max(), "FLT_MAX/FLT_MAX"},              // 164: = 1
    {std::numeric_limits<float>::max(),
     std::numeric_limits<float>::min(), "FLT_MAX/FLT_MIN"},              // 165: overflow → inf
    {std::numeric_limits<float>::min(),
     std::numeric_limits<float>::max(), "FLT_MIN/FLT_MAX"},              // 166: underflow → 0

    // FLT_MAX / small integer (large normals, non-trivial mantissa)
    {std::numeric_limits<float>::max() / 1.5f, 1.0f, "(FLT_MAX/1.5)/1"}, // 167
    {1.0f, std::numeric_limits<float>::max() / 1.5f, "1/(FLT_MAX/1.5)"}, // 168
    {std::numeric_limits<float>::max() / 2.0f, 1.0f, "(FLT_MAX/2)/1"},   // 169
    {1.0f, std::numeric_limits<float>::max() / 2.0f, "1/(FLT_MAX/2)"},   // 170
    {std::numeric_limits<float>::max() / 3.0f, 1.0f, "(FLT_MAX/3)/1"},   // 171
    {1.0f, std::numeric_limits<float>::max() / 3.0f, "1/(FLT_MAX/3)"},   // 172
    {std::numeric_limits<float>::max() / 4.0f, 1.0f, "(FLT_MAX/4)/1"},   // 173
    {1.0f, std::numeric_limits<float>::max() / 4.0f, "1/(FLT_MAX/4)"},   // 174
    {std::numeric_limits<float>::max() / 7.0f, 1.0f, "(FLT_MAX/7)/1"},   // 175
    {1.0f, std::numeric_limits<float>::max() / 7.0f, "1/(FLT_MAX/7)"},   // 176
    {std::numeric_limits<float>::max() / 8.0f, 1.0f, "(FLT_MAX/8)/1"},   // 177
    {1.0f, std::numeric_limits<float>::max() / 8.0f, "1/(FLT_MAX/8)"},   // 178
    {std::numeric_limits<float>::max() / 16.0f, 1.0f, "(FLT_MAX/16)/1"}, // 179
    {1.0f, std::numeric_limits<float>::max() / 16.0f, "1/(FLT_MAX/16)"}, // 180

    // large/large pairs (quotient is normal)
    {std::numeric_limits<float>::max() / 2.0f,
     std::numeric_limits<float>::max() / 4.0f,
     "(FLT_MAX/2)/(FLT_MAX/4)"},                                         // 181: = 2
    {std::numeric_limits<float>::max() / 3.0f,
     std::numeric_limits<float>::max() / 7.0f,
     "(FLT_MAX/3)/(FLT_MAX/7)"},                                         // 182: = 7/3
    {std::numeric_limits<float>::max() / 4.0f,
     std::numeric_limits<float>::max() / 6.0f,
     "(FLT_MAX/4)/(FLT_MAX/6)"},                                         // 183: = 3/2
    {std::numeric_limits<float>::max() / 5.0f,
     std::numeric_limits<float>::max() / 3.0f,
     "(FLT_MAX/5)/(FLT_MAX/3)"},                                         // 184: = 3/5
    {std::numeric_limits<float>::max() / 6.0f,
     std::numeric_limits<float>::max() / 16.0f,
     "(FLT_MAX/6)/(FLT_MAX/16)"},                                        // 185: = 8/3
    {std::numeric_limits<float>::max() / 7.0f,
     std::numeric_limits<float>::max() / 2.0f,
     "(FLT_MAX/7)/(FLT_MAX/2)"},                                         // 186: = 2/7
    {std::numeric_limits<float>::max() / 8.0f,
     std::numeric_limits<float>::max() / 8.0f,
     "(FLT_MAX/8)/(FLT_MAX/8)"},                                         // 187: = 1
    {std::numeric_limits<float>::max() / 11.0f,
     std::numeric_limits<float>::max() / 13.0f,
     "(FLT_MAX/11)/(FLT_MAX/13)"},                                       // 188: = 13/11
    {std::numeric_limits<float>::max() / 16.0f,
     std::numeric_limits<float>::max() / 32.0f,
     "(FLT_MAX/16)/(FLT_MAX/32)"},                                       // 189: = 2
    {std::numeric_limits<float>::max() / 32.0f,
     std::numeric_limits<float>::max() / 64.0f,
     "(FLT_MAX/32)/(FLT_MAX/64)"},                                       // 190: = 2
    {std::numeric_limits<float>::max() / 1.5f,
     std::numeric_limits<float>::max() / 2.5f,
     "(FLT_MAX/1.5)/(FLT_MAX/2.5)"},                                     // 191: = 5/3
    {std::numeric_limits<float>::max() / 2.0f,
     std::numeric_limits<float>::max() / 2.0f,
     "(FLT_MAX/2)/(FLT_MAX/2)"},                                         // 192: = 1

    // Probe: a = FLT_MAX/1536, b grows — rcp(b) underflow threshold
    // a ≈ 2.215e+35, pass/fail boundary between b=a*384 and b=a*512
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f,
     "(a=FLT_MAX/1536)/(a)"},                                            // 193: = 1
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 2.0f,
     "(a=FLT_MAX/1536)/(a*2)"},                                          // 194
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 4.0f,
     "(a=FLT_MAX/1536)/(a*4)"},                                          // 195
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 8.0f,
     "(a=FLT_MAX/1536)/(a*8)"},                                          // 196
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 16.0f,
     "(a=FLT_MAX/1536)/(a*16)"},                                         // 197
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 32.0f,
     "(a=FLT_MAX/1536)/(a*32)"},                                         // 198
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 64.0f,
     "(a=FLT_MAX/1536)/(a*64)"},                                         // 199
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 128.0f,
     "(a=FLT_MAX/1536)/(a*128)"},                                        // 200
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 192.0f,
     "(a=FLT_MAX/1536)/(a*192)"},                                        // 201
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 256.0f,
     "(a=FLT_MAX/1536)/(a*256)"},                                        // 202
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 384.0f,
     "(a=FLT_MAX/1536)/(a*384)"},                                        // 203: known pass
    // Zoom in on the pass/fail boundary between a*384 and a*512
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 400.0f,
     "(a=FLT_MAX/1536)/(a*400)"},                                        // 204
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 432.0f,
     "(a=FLT_MAX/1536)/(a*432)"},                                        // 205
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 448.0f,
     "(a=FLT_MAX/1536)/(a*448)"},                                        // 206
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 464.0f,
     "(a=FLT_MAX/1536)/(a*464)"},                                        // 207
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 480.0f,
     "(a=FLT_MAX/1536)/(a*480)"},                                        // 208
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 496.0f,
     "(a=FLT_MAX/1536)/(a*496)"},                                        // 209
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 504.0f,
     "(a=FLT_MAX/1536)/(a*504)"},                                        // 210
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 508.0f,
     "(a=FLT_MAX/1536)/(a*508)"},                                        // 211
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 510.0f,
     "(a=FLT_MAX/1536)/(a*510)"},                                        // 212
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 511.0f,
     "(a=FLT_MAX/1536)/(a*511)"},                                        // 213
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 512.0f,
     "(a=FLT_MAX/1536)/(a*512)"},                                        // 214: known error
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 768.0f,
     "(a=FLT_MAX/1536)/(a*768)"},                                        // 215: known error
    {std::numeric_limits<float>::max() / 1536.0f,
     std::numeric_limits<float>::max() / 1536.0f * 1536.0f,
     "(a=FLT_MAX/1536)/(a*1536)"},                                       // 216: known error

    // -----------------------------------------------------------------------
    // Cross pairs: small/large and large/small
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::min() * 4.0f,
     std::numeric_limits<float>::max() / 4.0f,
     "(FLT_MIN*4)/(FLT_MAX/4)"},                                         // 217: ≈ 0
    {std::numeric_limits<float>::max() / 4.0f,
     std::numeric_limits<float>::min() * 4.0f,
     "(FLT_MAX/4)/(FLT_MIN*4)"},                                         // 218: ≈ inf
    {std::numeric_limits<float>::min() * 3.0f,
     std::numeric_limits<float>::max() / 3.0f,
     "(FLT_MIN*3)/(FLT_MAX/3)"},                                         // 219: ≈ 0
    {std::numeric_limits<float>::max() / 3.0f,
     std::numeric_limits<float>::min() * 3.0f,
     "(FLT_MAX/3)/(FLT_MIN*3)"},                                         // 220: ≈ inf
    {std::numeric_limits<float>::min() / 2.0f,
     std::numeric_limits<float>::max() / 2.0f,
     "(FLT_MIN/2)/(FLT_MAX/2)"},                                         // 221: ≈ 0
    {std::numeric_limits<float>::max() / 2.0f,
     std::numeric_limits<float>::min() / 2.0f,
     "(FLT_MAX/2)/(FLT_MIN/2)"},                                         // 222: ≈ inf
    {std::numeric_limits<float>::min() / 3.0f,
     std::numeric_limits<float>::max() / 7.0f,
     "(FLT_MIN/3)/(FLT_MAX/7)"},                                         // 223: ≈ 0
    {std::numeric_limits<float>::max() / 7.0f,
     std::numeric_limits<float>::min() / 3.0f,
     "(FLT_MAX/7)/(FLT_MIN/3)"},                                         // 224: ≈ inf

    // -----------------------------------------------------------------------
    // Mixed scale: moderate values spanning large exponent range
    // -----------------------------------------------------------------------
    {1e-10f, 1e10f, "1e-10/1e10"},   // 225: = 1e-20
    {1e-20f, 1e20f, "1e-20/1e20"},   // 226: = 1e-40 (subnormal result)
    {1.0f, 1e20f, "1/1e20"},         // 227: = 1e-20
    {1e10f, 1e-10f, "1e10/1e-10"},   // 228: = 1e20
    {1e20f, 1e-10f, "1e20/1e-10"},   // 229: = 1e30
    {1e30f, 1e-5f, "1e30/1e-5"},     // 230: = 1e35
    {3.4e38f, 0.5f, "3.4e38/0.5"},   // 231: near overflow
    {1e-38f, 1e10f, "1e-38/1e10"},   // 232: near underflow
    {1e38f, 2e38f, "1e38/2e38"},     // 233: = 0.5
    {3e38f, 1e38f, "3e38/1e38"},     // 234: = 3
    {2e38f, 2e38f, "2e38/2e38"},     // 235: = 1
};
static_assert(sizeof(kCases) / sizeof(kCases[0]) == 236,
              "Update DivTester::N to match kCases length");

// ---------------------------------------------------------------------------
// Tester class
// ---------------------------------------------------------------------------
class DivTester {
public:
  static constexpr size_t N = 236;

  float input_a[N];
  float input_b[N];
  float output_div[N];        // CUDA operator/
  float output_fdividef[N];   // CUDA fdividef() (fast approx)
  float output_custom_div[N]; // CUSTOM_DIV() -- ASM version.

  __host__ DivTester() {
    for (size_t i = 0; i < N; i++) {
      input_a[i] = kCases[i].a;
      input_b[i] = kCases[i].b;
    }
  }

  __host__ void reset() {
    std::memset(output_div, 0xff, sizeof(output_div));
    std::memset(output_fdividef, 0xff, sizeof(output_fdividef));
    std::memset(output_custom_div, 0xff, sizeof(output_custom_div));
  }

  // -- display ---------------------------------------------------------------

  void __host__ displayResults(const float *torchinductor,
                               const float *torcheager) const;

  void __host__ displayResults() const;
};

// -- kernels ---------------------------------------------------------------

__global__ void testKernelDiv(DivTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < DivTester::N)
    self->output_div[idx] = self->input_a[idx] / self->input_b[idx];
}

__global__ void testKernelFdividef(DivTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < DivTester::N)
    self->output_fdividef[idx] =
        fdividef(self->input_a[idx], self->input_b[idx]);
}

__global__ void testKernelCustomDiv(DivTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < DivTester::N)
    self->output_custom_div[idx] =
        CUSTOM_DIV(self->input_a[idx], self->input_b[idx]);
}

// -- kernels for just looking at asm listings ------------------------------

__global__ void testKernelOneDiv(DivTester *self) {
  self->output_div[0] = self->input_a[0] / self->input_b[0];
}

__global__ void testKernelOneFdividef(DivTester *self) {
  self->output_fdividef[0] = fdividef(self->input_a[0], self->input_b[0]);
}

__global__ void testKernelOneCustomDiv(DivTester *self) {
  self->output_custom_div[0] = CUSTOM_DIV(self->input_a[0], self->input_b[0]);
}

bool verbose{};
bool compact{};
bool useColor{};
int quietLevel{};

// ---------------------------------------------------------------------------
// displayResults
// ---------------------------------------------------------------------------
void __host__ DivTester::displayResults(const float *torchinductor,
                                        const float *torcheager) const {
  if (useColor) {
    std::cout << RED;
  }
  std::cout << "DIV: fdividef(a,b)\n\n\n\n";
  if (useColor) {
    std::cout << RESET;
  }

  std::cout << std::format("{:>4}"     // Idx
                           "{:>16}"    // a
                           "{:>16}"    // b
                           "{:>30}"    // Label
                           "{:>16}"    // std::div (reference)
                           "{:>16}"    // operator/
                           "{:>16}"    // fdividef
                           "{:>16}"    // ASM
                           "{:>16}"    // torch-eager
                           "{:>16}\n", // torch-inductor
                           "Idx", "a", "b", "Label", "a/b (ref)", "operator/",
                           "fdividef", "ASM(div)",
                           torcheager ? "torch-eager" : "",
                           torchinductor ? "torch-inductor" : "");

  std::cout << std::string(152, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float a = input_a[i];
    float b = input_b[i];
    float ref = (float)((double)a / (double)b);

    OneResult32 v_div(ref, output_div[i], true, verbose);
    OneResult32 v_fdiv(ref, output_fdividef[i], true, verbose);
    OneResult32 v_asm(ref, output_custom_div[i], true, verbose);
    OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                      torcheager != nullptr, verbose);
    OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                         torchinductor != nullptr, verbose);

    uint32_t rbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::string ref_hex = std::format("0x{:08x}", rbits);

    bool allMatch = v_div.match and v_fdiv.match and v_asm.match and
                    v_inductor.match and v_eager.match;

    if (!quietLevel or !allMatch) {
      std::cout << std::format("{:>4}"
                               "{:>16g}"
                               "{:>16g}"
                               "{:>30}"
                               "{:>16.6g}"
                               "{}"
                               "{}"
                               "{}"
                               "{}"
                               "{}\n",
                               i, a, b, kCases[i].label, ref, v_div.value(),
                               v_fdiv.value(), v_asm.value(), v_eager.value(),
                               v_inductor.value());
    }

    if (!allMatch) {
      std::string hexline = std::format(
          "{:>4}"
          "{:>16}"
          "{:>16}"
          "{:>30}"
          "{:>16}"
          "{:>16}"
          "{:>16}"
          "{:>16}"
          "{:>16}"
          "{:>16}\n",
          "", "", "", "", ref_hex, v_div.hexValue(), v_fdiv.hexValue(),
          v_asm.hexValue(), v_eager.hexValue(), v_inductor.hexValue());
      std::cout << hexline;

      std::string es_div = v_div.errorString();
      std::string es_fdiv = v_fdiv.errorString();
      std::string es_asm = v_asm.errorString();
      std::string es_eager = v_eager.errorString();
      std::string es_inductor = v_inductor.errorString();

      const char *color = YELLOW;

      if ((es_div == "ERROR") or (es_fdiv == "ERROR") or (es_asm == "ERROR") or
          (es_eager == "ERROR") or (es_inductor == "ERROR")) {
        color = RED;
      }

      if (useColor) {
        std::cout << color;
      }
      std::string matchline = std::format("{:>4}"
                                          "{:>16}"
                                          "{:>16}"
                                          "{:>30}"
                                          "{:>16}"
                                          "{:>16}"
                                          "{:>16}"
                                          "{:>16}"
                                          "{:>16}"
                                          "{:>16}\n",
                                          "", "", "", "", "", es_div, es_fdiv,
                                          es_asm, es_eager, es_inductor);
      std::cout << matchline;
      if (useColor) {
        std::cout << RESET;
      }
    }
  }
}

void __host__ DivTester::displayResults() const {
  std::cout << std::format("{:>16}"    // a
                           "{:>16}"    // b
                           "{:>30}"    // Label
                           "{:>16}"    // ref
                           "{:>16}\n", // ASM
                           "a", "b", "Label", "a/b (ref)", "ASM(div)");
  std::cout << std::string(84, '-') << "\n";

  for (size_t i = 0; i < N; i++) {
    float a = input_a[i];
    float b = input_b[i];
    float ref = (float)((double)a / (double)b);

    OneResult32 v_asm(ref, output_custom_div[i], true, verbose);

    uint32_t rbits;
    uint32_t abits;
    uint32_t bbits;
    std::memcpy(&abits, &a, sizeof(a));
    std::memcpy(&bbits, &b, sizeof(b));
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::string a_hex = std::format("0x{:08x}", abits);
    std::string b_hex = std::format("0x{:08x}", bbits);
    std::string ref_hex = std::format("0x{:08x}", rbits);

    std::string es_asm = v_asm.errorString();

    const char *color = es_asm == "ERROR" ? RED : YELLOW;

    if ((quietLevel > 1) and (es_asm != "ERROR")) {
      continue;
    }

    if (!quietLevel or !v_asm.match) {
      std::cout << std::format("{:>16g}"
                               "{:>16g}"
                               "{:>30}"
                               "{:>16.6g}"
                               "{}\n",
                               a, b, kCases[i].label, ref, v_asm.value());

      std::string hexline =
          std::format("{:>16}"
                      "{:>16}"
                      "{:>30}"
                      "{:>16}"
                      "{:>16}\n",
                      a_hex, b_hex, "", ref_hex, v_asm.hexValue());
      std::cout << hexline;

      if (useColor) {
        std::cout << color;
      }

      std::cout << std::format("{:>16}"
                               "{:>16}"
                               "{:>30}"
                               "{:>16}"
                               "{:>16}\n",
                               "", "", "", "", es_asm);

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
  constexpr struct option longOptions[]{{"help", 0, nullptr, 'h'},
                                        {"verbose", 0, nullptr, 'v'},
                                        {"compact", 0, nullptr, 'C'},
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
      quietLevel++;
      break;
    }
    case 'c': {
      useColor = true;
      break;
    }
    case 'C': {
      compact = true;
      break;
    }
    case 'd': {
      dumpFile = optarg;
      break;
    }
    case 'h': {
      std::cout << "div_test"
                   " [--verbose]"
                   " [--quiet]"
                   " [--compact]"
                   " [--color]"
                   " [--dump-inputs filename]"
                   " [--torcheager torcheagerdiv.bin]"
                   " [--torchinductor torchinductordiv.bin]"
                   "\n\n"
                   "Run with:\n"
                   "  ./math/div_test --dump-inputs ./divtest.in\n"
                   "  ../torch/torchbinary.py --op div --file ./divtest.in\n"
                   "  ./math/div_test --torchinductor torchinductordiv.bin"
                   " --torcheager torcheagerdiv.bin --verbose --quiet --color"
                   " | less -R\n"
                   "\n"
                   "\t--dump-inputs filename.  Write input a,b pairs as binary "
                   "floats to file (a0,b0,a1,b1,...)\n"
                   "\t--verbose.  Show hex values, even if not mismatches\n"
                   "\t--quiet.  Suppress non-matching output.  w/ --compact "
                   "specify twice to omit ULP mismatches\n"
                   "\t--compact.  Show only std::divf vs ASM\n"
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
      std::cerr << "div_test: unknown option\n";
      return 1;
    }
    }
  }

  if (compact) {
    if (torchinductorFile or torcheagerFile) {
      std::cerr << "--compact and --torch* options are not compatible\n";
      return 2;
    }
  }

  // Dump input a,b pairs to binary file if requested
  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << '\n';
      return 2;
    }
    for (size_t i = 0; i < DivTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&kCases[i].a), sizeof(float));
      ofs.write(reinterpret_cast<const char *>(&kCases[i].b), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input a,b pairs to " << dumpFile << '\n';
    return 0;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, DivTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, DivTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  DivTester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(DivTester)));
  new (tester) DivTester();
  tester->reset();

  dim3 blockSize(DivTester::N);
  dim3 gridSize(1);

  testKernelDiv<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  testKernelFdividef<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  testKernelCustomDiv<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  if (compact) {
    tester->displayResults();
  } else {
    tester->displayResults(torchinductorFile ? torchinductorOut.data()
                                             : nullptr,
                           torcheagerFile ? torcheagerOut.data() : nullptr);
  }

  CUDA_CHECK(cudaFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
