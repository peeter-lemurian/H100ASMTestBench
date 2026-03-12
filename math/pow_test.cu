//
// pow_test.cu
//
// Tests the builtin powf() / __powf() on NVIDIA H100 hardware, comparing
// results against std::pow() computed on the CPU.
//
// The intent is to:
//   1. Establish ground truth via CUDA's powf() and __powf().
//   2. Provide a slot (PowTester::testKernelCustomPow) where a hand-coded
//      inline-asm pow(a,b) can be dropped in later.
//
// pow(a,b) = 2^(b * log2(a))   for a > 0
//
// The obvious NVIDIA PTX sequence is, for a != 0, is:
//   lg2.approx.f32  tmp, a        // log2(a)
//   mul.f32         tmp, b, tmp   // b * log2(a)
//   ex2.approx.f32  tmp, tmp      // 2^(b * log2(a))
//
#include <cuda_runtime.h>
#include <getopt.h>
#include <limits>
#include <unistd.h>
#include <cstdint>

#include "readbinary.hpp"
#include "OneResult32.hpp"
#include "colors.hpp"
#include "cuda_check.hpp"

// ---------------------------------------------------------------------------
// Custom pow
// ---------------------------------------------------------------------------
// clang-format off
inline float __device__ custom_pow(float a, float b) {
  float result = pow(a, b); // placeholder for later ASM version.
  return result;
}
// clang-format on

#define CUSTOM_POW custom_pow

// ---------------------------------------------------------------------------
// Test-case table
// ---------------------------------------------------------------------------
struct PowCase {
  float a;
  float b;
  const char *label;
};

static constexpr PowCase kCases[] = {
    // -----------------------------------------------------------------------
    // Basic integer-like cases
    // -----------------------------------------------------------------------
    {2.0f, 0.0f, "2^0"},     //  0: 1
    {2.0f, 1.0f, "2^1"},     //  1: 2
    {2.0f, 2.0f, "2^2"},     //  2: 4
    {2.0f, 3.0f, "2^3"},     //  3: 8
    {2.0f, 10.0f, "2^10"},   //  4: 1024
    {2.0f, -1.0f, "2^(-1)"}, //  5: 0.5
    {2.0f, -3.0f, "2^(-3)"}, //  6: 0.125

    // -----------------------------------------------------------------------
    // Non-integer exponents
    // -----------------------------------------------------------------------
    {2.0f, 0.5f, "2^(0.5)"},        //  7: sqrt(2)
    {4.0f, 0.5f, "4^(0.5)"},        //  8: 2
    {8.0f, 1.0f / 3.0f, "8^(1/3)"}, //  9: 2
    {2.0f, 1.5f, "2^(1.5)"},        // 10: 2*sqrt(2)

    // -----------------------------------------------------------------------
    // Base e
    // -----------------------------------------------------------------------
    {2.718281828f, 1.0f, "e^1"},     // 11: e
    {2.718281828f, 2.0f, "e^2"},     // 12: e^2
    {2.718281828f, 0.5f, "e^(0.5)"}, // 13: sqrt(e)

    // -----------------------------------------------------------------------
    // Base 10
    // -----------------------------------------------------------------------
    {10.0f, 2.0f, "10^2"},     // 14: 100
    {10.0f, 0.5f, "10^(0.5)"}, // 15: sqrt(10)
    {10.0f, -1.0f, "10^(-1)"}, // 16: 0.1

    // -----------------------------------------------------------------------
    // Fractional base
    // -----------------------------------------------------------------------
    {0.5f, 2.0f, "(0.5)^2"},       // 17: 0.25
    {0.5f, -2.0f, "(0.5)^(-2)"},   // 18: 4
    {0.25f, 0.5f, "(0.25)^(0.5)"}, // 19: 0.5

    // -----------------------------------------------------------------------
    // Near-boundary exponents for 2^x overflow/underflow
    // -----------------------------------------------------------------------
    {2.0f, 126.0f, "2^126"},     // 20: near f32 max
    {2.0f, -126.0f, "2^(-126)"}, // 21: near f32 min normal

    // -----------------------------------------------------------------------
    // Near-boundary for e^x (b*log2(a) perspective)
    // -----------------------------------------------------------------------
    {2.718281828f, 88.0f, "e^88"},     // 22: near expf overflow
    {2.718281828f, -88.0f, "e^(-88)"}, // 23: near expf underflow

    // -----------------------------------------------------------------------
    // a = 1: should always be 1.0 for all b including special values
    // -----------------------------------------------------------------------
    {1.0f, 100.0f, "1^100"},                                     // 24: 1
    {1.0f, -100.0f, "1^(-100)"},                                 // 25: 1
    {1.0f, std::numeric_limits<float>::infinity(), "1^(+inf)"},  // 26: 1
    {1.0f, -std::numeric_limits<float>::infinity(), "1^(-inf)"}, // 27: 1
    {1.0f, std::numeric_limits<float>::quiet_NaN(), "1^NaN"},    // 28: 1

    // -----------------------------------------------------------------------
    // Large base, large exponent
    // -----------------------------------------------------------------------
    {3.0f, 5.0f, "3^5"},     // 29: 243
    {5.0f, 4.0f, "5^4"},     // 30: 625
    {10.0f, 10.0f, "10^10"}, // 31: 1e10

    // -----------------------------------------------------------------------
    // Small base, small exponent
    // -----------------------------------------------------------------------
    {0.1f, 0.1f, "(0.1)^(0.1)"}, // 32
    {0.9f, 10.0f, "(0.9)^10"},   // 33
    {1.1f, 10.0f, "(1.1)^10"},   // 34

    // -----------------------------------------------------------------------
    // pow(x, 0) = 1 for ALL x
    // -----------------------------------------------------------------------
    {0.0f, 0.0f, "0^0"},                                         // 35: 1
    {-1.0f, 0.0f, "(-1)^0"},                                     // 36: 1
    {-2.0f, 0.0f, "(-2)^0"},                                     // 37: 1
    {std::numeric_limits<float>::infinity(), 0.0f, "(+inf)^0"},  // 38: 1
    {-std::numeric_limits<float>::infinity(), 0.0f, "(-inf)^0"}, // 39: 1
    {std::numeric_limits<float>::quiet_NaN(), 0.0f, "NaN^0"},    // 40: 1

    // -----------------------------------------------------------------------
    // pow(0, y) cases
    // -----------------------------------------------------------------------
    {0.0f, 1.0f, "0^1"},     // 41: 0    ; log(0)=-inf, exp(b*-inf)=0)
    {0.0f, -1.0f, "0^(-1)"}, // 42: +inf ; exp(-1*-inf)=exp(+inf)=+inf)
    {0.0f, 2.0f, "0^2"},     // 43: 0
    {0.0f, -2.0f, "0^(-2)"}, // 44: +inf
    {0.0f, 0.5f, "0^(0.5)"}, // 45: 0

    // -----------------------------------------------------------------------
    // Negative base, integer exponent — sign must reflect parity of exponent
    // -----------------------------------------------------------------------
    {-2.0f, 1.0f, "(-2)^1"},     // 46: -2
    {-2.0f, 2.0f, "(-2)^2"},     // 47: 4
    {-2.0f, 3.0f, "(-2)^3"},     // 48: -8
    {-2.0f, 4.0f, "(-2)^4"},     // 49: 16
    {-1.0f, 3.0f, "(-1)^3"},     // 50: -1
    {-1.0f, 4.0f, "(-1)^4"},     // 51: 1
    {-2.0f, -1.0f, "(-2)^(-1)"}, // 52: -0.5
    {-2.0f, -2.0f, "(-2)^(-2)"}, // 53: 0.25

    // -----------------------------------------------------------------------
    // Negative base, non-integer exponent — should be NaN
    // CUSTOM_POW returns NaN, which is correct
    // -----------------------------------------------------------------------
    {-2.0f, 0.5f, "(-2)^(0.5)"}, // 54: NaN
    {-2.0f, 1.5f, "(-2)^(1.5)"}, // 55: NaN
    {-2.0f, 0.1f, "(-2)^(0.1)"}, // 56: NaN

    // -----------------------------------------------------------------------
    // pow(-1, ±inf) = 1 (special IEEE case)
    // -----------------------------------------------------------------------
    {-1.0f, std::numeric_limits<float>::infinity(), "(-1)^(+inf)"},  // 57: 1
    {-1.0f, -std::numeric_limits<float>::infinity(), "(-1)^(-inf)"}, // 58: 1

    // -----------------------------------------------------------------------
    // pow(±inf, y) cases
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::infinity(), 2.0f, "(+inf)^2"},      // 59: +inf
    {std::numeric_limits<float>::infinity(), -1.0f, "(+inf)^(-1)"},  // 60: 0
    {std::numeric_limits<float>::infinity(), 0.5f, "(+inf)^(0.5)"},  // 61: +inf
    {-std::numeric_limits<float>::infinity(), 2.0f, "(-inf)^2"},     // 62: +inf
    {-std::numeric_limits<float>::infinity(), 3.0f, "(-inf)^3"},     // 63: -inf
    {-std::numeric_limits<float>::infinity(), -1.0f, "(-inf)^(-1)"}, // 64: -0
    {-std::numeric_limits<float>::infinity(), -2.0f, "(-inf)^(-2)"}, // 65: +0

    // -----------------------------------------------------------------------
    // NaN propagation — pow(NaN, y) and pow(x, NaN) should return NaN
    // except pow(x, 0) and pow(1, NaN) already covered above
    // -----------------------------------------------------------------------
    {std::numeric_limits<float>::quiet_NaN(), 2.0f, "NaN^2"},     // 66: NaN
    {std::numeric_limits<float>::quiet_NaN(), 0.5f, "NaN^(0.5)"}, // 67: NaN
    {2.0f, std::numeric_limits<float>::quiet_NaN(), "2^NaN"},     // 68: NaN

    // -----------------------------------------------------------------------
    // Subnormal base — behavior depends on DAZ mode
    // -----------------------------------------------------------------------
    {1.175494e-38f, 1.0f,
     "(1.175494e-38)^1"}, // 69: identity, tests subnormal log
    {1.175494e-38f, 2.0f,
     "(1.175494e-38)^2"}, // 70: subnormal^2 -> underflow to 0
    {0.0f, std::numeric_limits<float>::infinity(), "0^(+inf)"},      // 71
    {0.0f, -std::numeric_limits<float>::infinity(), "0^(-inf)"},     // 72
    {-0.0f, std::numeric_limits<float>::infinity(), "(-0)^(+inf)"},  // 73
    {-0.0f, -std::numeric_limits<float>::infinity(), "(-0)^(-inf)"}, // 74
    {std::numeric_limits<float>::infinity(),
     std::numeric_limits<float>::infinity(), "(+inf)^(+inf)"}, // 75
    {std::numeric_limits<float>::infinity(),
     -std::numeric_limits<float>::infinity(), "(+inf)^(-inf)"}, // 76
    {-std::numeric_limits<float>::infinity(),
     std::numeric_limits<float>::infinity(), "(-inf)^(+inf)"}, // 77
    {-std::numeric_limits<float>::infinity(),
     -std::numeric_limits<float>::infinity(), "(-inf)^(-inf)"},      // 78
    {0.0f, -0.5f, "0^(-0.5)"},                                       // 79
    {-0.0f, -0.5f, "(-0)^(-0.5)"},                                   // 80
    {-0.0f, 2.0f, "(-0)^2"},                                         // 81
    {-0.0f, 3.0f, "(-0)^3"},                                         // 82
    {-0.0f, 0.5f, "(-0)^(0.5)"},                                     // 83
    {-2.0f, std::numeric_limits<float>::infinity(), "(-2)^(+inf)"},  // 84
    {-2.0f, -std::numeric_limits<float>::infinity(), "(-2)^(-inf)"}, // 85
    {std::numeric_limits<float>::quiet_NaN(),
     std::numeric_limits<float>::quiet_NaN(), "NaN^NaN"},         // 86
    {0.0f, std::numeric_limits<float>::quiet_NaN(), "0^NaN"},     // 87
    {-std::numeric_limits<float>::quiet_NaN(), 0.0f, "(-NaN)^0"}, // 88

    // -----------------------------------------------------------------------
    // Near-zero base neighbourhood (base ≈ 0)
    // -----------------------------------------------------------------------
    {1e-5f, 1.0f, "(1e-5)^1"},       // 89
    {1e-5f, 2.0f, "(1e-5)^2"},       // 90
    {1e-5f, 0.5f, "(1e-5)^(0.5)"},   // 91
    {1e-5f, -1.0f, "(1e-5)^(-1)"},   // 92
    {0.01f, 1.0f, "(0.01)^1"},       // 93
    {0.01f, 2.0f, "(0.01)^2"},       // 94
    {0.01f, 0.5f, "(0.01)^(0.5)"},   // 95
    {0.01f, -1.0f, "(0.01)^(-1)"},   // 96
    {-0.01f, 1.0f, "(-0.01)^1"},     // 97
    {-0.01f, 2.0f, "(-0.01)^2"},     // 98
    {-0.01f, 3.0f, "(-0.01)^3"},     // 99
    {-0.01f, -1.0f, "(-0.01)^(-1)"}, // 100
    {-0.01f, -2.0f, "(-0.01)^(-2)"}, // 101
    {1e-9f, 1.0f, "(1e-9)^1"},       // 102
    {1e-9f, 2.0f, "(1e-9)^2"},       // 103
    {1e-9f, 0.5f, "(1e-9)^(0.5)"},   // 104
    {1e-9f, -1.0f, "(1e-9)^(-1)"},   // 105
    {-1e-9f, 1.0f, "(-1e-9)^1"},     // 106
    {-1e-9f, 2.0f, "(-1e-9)^2"},     // 107
    {-1e-9f, 3.0f, "(-1e-9)^3"},     // 108
    {-1e-9f, -1.0f, "(-1e-9)^(-1)"}, // 109
    {-1e-9f, -2.0f, "(-1e-9)^(-2)"}, // 110

    // -----------------------------------------------------------------------
    // Near-one base neighbourhood (base ≈ 1)
    // -----------------------------------------------------------------------
    {1.00000001f, 1.0f, "(1+1e-8)^1"},         // 111
    {1.00000001f, 2.0f, "(1+1e-8)^2"},         // 112
    {1.00000001f, 100.0f, "(1+1e-8)^100"},     // 113
    {1.00000001f, -100.0f, "(1+1e-8)^(-100)"}, // 114
    {1.00000001f, 1e6f, "(1+1e-8)^1e6"},       // 115
    {1.00000001f, -1e6f, "(1+1e-8)^(-1e6)"},   // 116
    {0.9999999f, 1.0f, "(1-1e-7)^1"},          // 117
    {0.9999999f, 2.0f, "(1-1e-7)^2"},          // 118
    {0.9999999f, 100.0f, "(1-1e-7)^100"},      // 119
    {0.9999999f, -100.0f, "(1-1e-7)^(-100)"},  // 120
    {0.9999999f, 1e6f, "(1-1e-7)^1e6"},        // 121
    {0.9999999f, -1e6f, "(1-1e-7)^(-1e6)"},    // 122

    // -----------------------------------------------------------------------
    // Near-zero exponent neighbourhood (exponent ≈ 0)
    // -----------------------------------------------------------------------
    {2.0f, 1e-5f, "2^(1e-5)"},       // 123
    {2.0f, -1e-5f, "2^(-1e-5)"},     // 124
    {2.0f, 0.01f, "2^(0.01)"},       // 125
    {2.0f, -0.01f, "2^(-0.01)"},     // 126
    {2.0f, 1e-9f, "2^(1e-9)"},       // 127
    {2.0f, -1e-9f, "2^(-1e-9)"},     // 128
    {10.0f, 1e-5f, "10^(1e-5)"},     // 129
    {10.0f, -1e-5f, "10^(-1e-5)"},   // 130
    {10.0f, 0.01f, "10^(0.01)"},     // 131
    {10.0f, -0.01f, "10^(-0.01)"},   // 132
    {0.5f, 1e-5f, "(0.5)^(1e-5)"},   // 133
    {0.5f, -1e-5f, "(0.5)^(-1e-5)"}, // 134
    {0.5f, 0.01f, "(0.5)^(0.01)"},   // 135
    {0.5f, -0.01f, "(0.5)^(-0.01)"}, // 136
    {0.0f, 1e-5f, "0^(1e-5)"},       // 137
    {0.0f, 0.01f, "0^(0.01)"},       // 138
    {-0.0f, 1e-5f, "(-0)^(1e-5)"},   // 139
    {-0.0f, 0.01f, "(-0)^(0.01)"},   // 140

    // -----------------------------------------------------------------------
    // Near-one exponent neighbourhood (exponent ≈ 1)
    // -----------------------------------------------------------------------
    {2.0f, 1.00000001f, "2^(1+1e-8)"},     // 141
    {2.0f, 0.9999999f, "2^(1-1e-7)"},      // 142
    {10.0f, 1.00000001f, "10^(1+1e-8)"},   // 143
    {10.0f, 0.9999999f, "10^(1-1e-7)"},    // 144
    {0.5f, 1.00000001f, "(0.5)^(1+1e-8)"}, // 145
    {0.5f, 0.9999999f, "(0.5)^(1-1e-7)"},  // 146
    {100.0f, 1.00000001f, "100^(1+1e-8)"}, // 147
    {100.0f, 0.9999999f, "100^(1-1e-7)"},  // 148
    {-2.0f, 1.00000001f, "(-2)^(1+1e-8)"}, // 149: NaN (non-integer exp)
    {-2.0f, 0.9999999f, "(-2)^(1-1e-7)"},  // 150: NaN (non-integer exp)

    // -----------------------------------------------------------------------
    // Cross-neighbourhood: near-zero base × near-zero exponent
    // -----------------------------------------------------------------------
    {1e-5f, 1e-5f, "(1e-5)^(1e-5)"},   // 151
    {1e-5f, -1e-5f, "(1e-5)^(-1e-5)"}, // 152
    {0.01f, 0.01f, "(0.01)^(0.01)"},   // 153
    {0.01f, -0.01f, "(0.01)^(-0.01)"}, // 154
    {1e-9f, 1e-9f, "(1e-9)^(1e-9)"},   // 155
    {1e-9f, -1e-9f, "(1e-9)^(-1e-9)"}, // 156

    // -----------------------------------------------------------------------
    // Cross-neighbourhood: near-one base × near-one exponent
    // -----------------------------------------------------------------------
    {1.00000001f, 1.00000001f, "(1+1e-8)^(1+1e-8)"}, // 157
    {1.00000001f, 0.9999999f, "(1+1e-8)^(1-1e-7)"},  // 158
    {0.9999999f, 1.00000001f, "(1-1e-7)^(1+1e-8)"},  // 159
    {0.9999999f, 0.9999999f, "(1-1e-7)^(1-1e-7)"},   // 160

    // -----------------------------------------------------------------------
    // Subnormal bases with near-one exponents
    // (subnormals: magnitude < 1.175494e-38 = 0x00800000)
    // -----------------------------------------------------------------------
    // Smallest positive subnormal: ~1.4e-45
    {1.401298e-45f, 0.9999999f, "min_subnorm^(1-1e-7)"},  // 161
    {1.401298e-45f, 1.00000001f, "min_subnorm^(1+1e-8)"}, // 162
    {1.401298e-45f, 0.99f, "min_subnorm^0.99"},           // 163
    {1.401298e-45f, 1.01f, "min_subnorm^1.01"},           // 164
    {1.401298e-45f, 0.5f, "min_subnorm^0.5"},             // 165
    {1.401298e-45f, 2.0f, "min_subnorm^2"},               // 166

    // Mid-range subnormal: ~5.9e-39
    {5.877472e-39f, 0.9999999f, "mid_subnorm^(1-1e-7)"},  // 167
    {5.877472e-39f, 1.00000001f, "mid_subnorm^(1+1e-8)"}, // 168
    {5.877472e-39f, 0.99f, "mid_subnorm^0.99"},           // 169
    {5.877472e-39f, 1.01f, "mid_subnorm^1.01"},           // 170
    {5.877472e-39f, 0.5f, "mid_subnorm^0.5"},             // 171
    {5.877472e-39f, 2.0f, "mid_subnorm^2"},               // 172

    // Largest subnormal: just below min normal (~1.1754942e-38)
    {1.1754942e-38f, 0.9999999f, "max_subnorm^(1-1e-7)"},  // 173
    {1.1754942e-38f, 1.00000001f, "max_subnorm^(1+1e-8)"}, // 174
    {1.1754942e-38f, 0.99f, "max_subnorm^0.99"},           // 175
    {1.1754942e-38f, 1.01f, "max_subnorm^1.01"},           // 176
    {1.1754942e-38f, 0.5f, "max_subnorm^0.5"},             // 177
    {1.1754942e-38f, 2.0f, "max_subnorm^2"},               // 178

    // The existing subnormal case value (min normal = 1.175494e-38)
    {1.175494e-38f, 0.9999999f, "min_normal^(1-1e-7)"},  // 179
    {1.175494e-38f, 1.00000001f, "min_normal^(1+1e-8)"}, // 180
    {1.175494e-38f, 0.99f, "min_normal^0.99"},           // 181
    {1.175494e-38f, 1.01f, "min_normal^1.01"},           // 182
    {1.175494e-38f, 0.5f, "min_normal^0.5"},             // 183
    {1.175494e-38f, -1.0f, "min_normal^(-1)"},           // 184

    // -----------------------------------------------------------------------
    // Small normal bases (smallest exponent, but not subnormal)
    // with near-one exponents
    // -----------------------------------------------------------------------
    // Just above min normal
    {1.175495e-38f, 0.9999999f, "(min_normal+eps)^(1-1e-7)"},  // 185
    {1.175495e-38f, 1.00000001f, "(min_normal+eps)^(1+1e-8)"}, // 186
    {1.175495e-38f, 0.99f, "(min_normal+eps)^0.99"},           // 187
    {1.175495e-38f, 1.01f, "(min_normal+eps)^1.01"},           // 188

    // 2x min normal
    {2.350989e-38f, 0.9999999f, "(2*min_normal)^(1-1e-7)"},  // 189
    {2.350989e-38f, 1.00000001f, "(2*min_normal)^(1+1e-8)"}, // 190
    {2.350989e-38f, 0.99f, "(2*min_normal)^0.99"},           // 191
    {2.350989e-38f, 1.01f, "(2*min_normal)^1.01"},           // 192

    // 1e-37 (small normal, a bit above min normal)
    {1e-37f, 0.9999999f, "(1e-37)^(1-1e-7)"},  // 193
    {1e-37f, 1.00000001f, "(1e-37)^(1+1e-8)"}, // 194
    {1e-37f, 0.99f, "(1e-37)^0.99"},           // 195
    {1e-37f, 1.01f, "(1e-37)^1.01"},           // 196

    // 1e-30 (small normal, well above subnormal range)
    {1e-30f, 0.9999999f, "(1e-30)^(1-1e-7)"},  // 197
    {1e-30f, 1.00000001f, "(1e-30)^(1+1e-8)"}, // 198
    {1e-30f, 0.99f, "(1e-30)^0.99"},           // 199
    {1e-30f, 1.01f, "(1e-30)^1.01"},           // 200

    // -----------------------------------------------------------------------
    // Small-normal power-of-two sweep (increasing by powers of two)
    // -----------------------------------------------------------------------
    {0x1p-126f, 1.0f, "2^-126 ^1"},   // 201
    {0x1p-126f, 0.5f, "2^-126 ^0.5"}, // 202
    {0x1p-126f, 2.0f, "2^-126 ^2"},   // 203
    {0x1p-126f, -1.0f, "2^-126 ^-1"}, // 204

    {0x1p-124f, 1.0f, "2^-124 ^1"},   // 205
    {0x1p-124f, 0.5f, "2^-124 ^0.5"}, // 206
    {0x1p-124f, 2.0f, "2^-124 ^2"},   // 207
    {0x1p-124f, -1.0f, "2^-124 ^-1"}, // 208

    {0x1p-122f, 1.0f, "2^-122 ^1"},   // 209
    {0x1p-122f, 0.5f, "2^-122 ^0.5"}, // 210
    {0x1p-122f, 2.0f, "2^-122 ^2"},   // 211
    {0x1p-122f, -1.0f, "2^-122 ^-1"}, // 212

    {0x1p-120f, 1.0f, "2^-120 ^1"},   // 213
    {0x1p-120f, 0.5f, "2^-120 ^0.5"}, // 214
    {0x1p-120f, 2.0f, "2^-120 ^2"},   // 215
    {0x1p-120f, -1.0f, "2^-120 ^-1"}, // 216

    {0x1p-118f, 1.0f, "2^-118 ^1"},   // 217
    {0x1p-118f, 0.5f, "2^-118 ^0.5"}, // 218
    {0x1p-118f, 2.0f, "2^-118 ^2"},   // 219
    {0x1p-118f, -1.0f, "2^-118 ^-1"}, // 220

    {0x1p-116f, 1.0f, "2^-116 ^1"},   // 221
    {0x1p-116f, 0.5f, "2^-116 ^0.5"}, // 222
    {0x1p-116f, 2.0f, "2^-116 ^2"},   // 223
    {0x1p-116f, -1.0f, "2^-116 ^-1"}, // 224

    {0x1p-114f, 1.0f, "2^-114 ^1"},   // 225
    {0x1p-114f, 0.5f, "2^-114 ^0.5"}, // 226
    {0x1p-114f, 2.0f, "2^-114 ^2"},   // 227
    {0x1p-114f, -1.0f, "2^-114 ^-1"}, // 228

    // -----------------------------------------------------------------------
    // Negative small-normal power-of-two sweep (sign/parity stress)
    // -----------------------------------------------------------------------
    {-0x1p-126f, 1.0f, "(-2^-126)^1"},   // 229
    {-0x1p-126f, 2.0f, "(-2^-126)^2"},   // 230
    {-0x1p-126f, 3.0f, "(-2^-126)^3"},   // 231
    {-0x1p-126f, -1.0f, "(-2^-126)^-1"}, // 232
    {-0x1p-126f, -2.0f, "(-2^-126)^-2"}, // 233

    {-0x1p-122f, 1.0f, "(-2^-122)^1"},   // 234
    {-0x1p-122f, 2.0f, "(-2^-122)^2"},   // 235
    {-0x1p-122f, 3.0f, "(-2^-122)^3"},   // 236
    {-0x1p-122f, -1.0f, "(-2^-122)^-1"}, // 237
    {-0x1p-122f, -2.0f, "(-2^-122)^-2"}, // 238

    {-0x1p-118f, 1.0f, "(-2^-118)^1"},   // 239
    {-0x1p-118f, 2.0f, "(-2^-118)^2"},   // 240
    {-0x1p-118f, 3.0f, "(-2^-118)^3"},   // 241
    {-0x1p-118f, -1.0f, "(-2^-118)^-1"}, // 242
    {-0x1p-118f, -2.0f, "(-2^-118)^-2"}, // 243

    {-0x1p-114f, 1.0f, "(-2^-114)^1"},   // 244
    {-0x1p-114f, 2.0f, "(-2^-114)^2"},   // 245
    {-0x1p-114f, 3.0f, "(-2^-114)^3"},   // 246
    {-0x1p-114f, -1.0f, "(-2^-114)^-1"}, // 247
    {-0x1p-114f, -2.0f, "(-2^-114)^-2"}, // 248

    // -----------------------------------------------------------------------
    // Additional zero/signed-zero and boundary exponent cases
    // -----------------------------------------------------------------------
    {0.0f, -0.0f, "0^(-0)"},                                      // 249
    {-0.0f, -0.0f, "(-0)^(-0)"},                                  // 250
    {-0.0f, std::numeric_limits<float>::quiet_NaN(), "(-0)^NaN"}, // 251
    {0.0f, 1.401298e-45f, "0^(+min_subnorm)"},                    // 252
    {0.0f, -1.401298e-45f, "0^(-min_subnorm)"},                   // 253
    {-0.0f, 1.401298e-45f, "(-0)^(+min_subnorm)"},                // 254
    {-0.0f, -1.401298e-45f, "(-0)^(-min_subnorm)"},               // 255
    {-0.0f, 16777215.0f, "(-0)^(odd_2^24-1)"},                    // 256
    {-0.0f, 16777216.0f, "(-0)^(even_2^24)"},                     // 257
    {-0.0f, -16777215.0f, "(-0)^(-odd_2^24-1)"},                  // 258
    {-0.0f, -16777216.0f, "(-0)^(-even_2^24)"},                   // 259

};
static constexpr size_t kNumPowCases = sizeof(kCases) / sizeof(kCases[0]);

bool verbose{};
bool quiet{};
bool useColor{};

// ---------------------------------------------------------------------------
// Tester class
// ---------------------------------------------------------------------------
class PowTester {
public:
  static constexpr size_t N = kNumPowCases;

  float input_a[N];
  float input_b[N];
  float output_powf[N];      // CUDA powf()
  float output_fast_powf[N]; // CUDA __powf() (lower precision)
  float output_custom[N];    // CUSTOM_POW() (hand-coded later)

  __host__ PowTester() {
    for (size_t i = 0; i < N; i++) {
      input_a[i] = kCases[i].a;
      input_b[i] = kCases[i].b;
    }
  }

  __host__ void reset() {
    auto fill = [](float *arr, size_t n) {
      std::memset(arr, 0xff, n * sizeof(float));
    };
    fill(output_powf, N);
    fill(output_fast_powf, N);
    fill(output_custom, N);
  }

  // -- display ---------------------------------------------------------------

  __host__ void displayResults(const float *cuda_powf, const float *cuda___powf,
                               const float *ASM, const float *torchinductor,
                               const float *torcheager) const {

    if (useColor) {
      std::cout << RED;
    }
    std::cout << "POW: powf(a,b)\n\n\n\n";
    if (useColor) {
      std::cout << RESET;
    }

    std::cout << std::format("{:>4}"     // Idx
                             "{:>25}"    // a
                             "{:>25}"    // b
                             "{:>25}"    // Label
                             "{:>16}"    // std::powf
                             "{:>16}"    // powf
                             "{:>16}"    // __powf
                             "{:>16}"    // ASM
                             "{:>16}"    // torchinductor
                             "{:>16}\n", // torcheager
                             "Idx", "a", "b", "Label", "std::powf", "powf",
                             "__powf", "ASM", torcheager ? "torch-eager" : "",
                             torchinductor ? "torch-inductor" : "");

    std::cout << std::string(175, '-') << "\n";

    for (size_t i = 0; i < N; i++) {
      float a = input_a[i];
      float b = input_b[i];
      float ref = powf(a, b);

      OneResult32 v_powf(ref, cuda_powf[i], true, verbose);
      OneResult32 v__powf(ref, cuda___powf[i], true, verbose);
      OneResult32 v_asm(ref, ASM[i], true, verbose);
      OneResult32 v_eager(ref, torcheager ? torcheager[i] : 0.0f,
                        torcheager != nullptr, verbose);
      OneResult32 v_inductor(ref, torchinductor ? torchinductor[i] : 0.0f,
                           torchinductor != nullptr, verbose);

      uint32_t rbits;
      std::memcpy(&rbits, &ref, sizeof(ref));
      std::string ref_hex = std::format("0x{:08x}", rbits);

      bool allMatch = v_powf.match and v__powf.match and v_asm.match and
                      v_inductor.match and v_eager.match;

      if (!quiet or !allMatch)
      {
        std::cout << std::format("{:>4}"
                                 "{:>25}"
                                 "{:>25}"
                                 "{:>25}"
                                 "{:>16.6g}"
                                 "{}"
                                 "{}"
                                 "{}"
                                 "{}"
                                 "{}\n",
                                 i, a, b, kCases[i].label, ref, v_powf.value(),
                                 v__powf.value(), v_asm.value(), v_eager.value(),
                                 v_inductor.value());
      }

      if (!allMatch) {
        std::string hexline = std::format(
            "{:>4}"
            "{:>25}"
            "{:>25}"
            "{:>25}"
            "{:>16}"
            "{:>16}"
            "{:>16}"
            "{:>16}"
            "{:>16}"
            "{:>16}\n",
            "", "", "", "", ref_hex, v_powf.hexValue(), v__powf.hexValue(),
            v_asm.hexValue(), v_eager.hexValue(), v_inductor.hexValue());
        std::cout << hexline;

        std::string es_v_powf = v_powf.errorString();
        std::string es_v__powf = v__powf.errorString();
        std::string es_v_asm = v_asm.errorString();
        std::string es_v_eager = v_eager.errorString();
        std::string es_v_inductor = v_inductor.errorString();

        const char *color = YELLOW;

        if ((es_v_powf == "ERROR") or (es_v__powf == "ERROR") or
            (es_v_asm == "ERROR") or (es_v_eager == "ERROR") or
            (es_v_inductor == "ERROR")) {
          color = RED;
        }

        if (useColor) {
          std::cout << color;
        }
        std::string matchline =
            std::format("{:>4}"
                        "{:>25}"
                        "{:>25}"
                        "{:>25}"
                        "{:>16}"
                        "{:>16}"
                        "{:>16}"
                        "{:>16}"
                        "{:>16}"
                        "{:>16}\n",
                        "", "", "", "", "", es_v_powf, es_v__powf, es_v_asm,
                        es_v_eager, es_v_inductor);
        std::cout << matchline;
        if (useColor) {
          std::cout << RESET;
        }
      }
    }
  }
};

// -- kernels ---------------------------------------------------------------

__global__ void testKernelPowf(PowTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < PowTester::N)
    self->output_powf[idx] = powf(self->input_a[idx], self->input_b[idx]);
}

__global__ void testKernelFastPowf(PowTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < PowTester::N)
    self->output_fast_powf[idx] =
        __powf(self->input_a[idx], self->input_b[idx]);
}

__global__ void testKernelCustomPow(PowTester *self) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < PowTester::N)
    self->output_custom[idx] =
        CUSTOM_POW(self->input_a[idx], self->input_b[idx]);
}

// -- kernels for just looking at asm listings ------------------------------

__global__ void testKernelOnePowf(PowTester *self) {
  self->output_powf[0] = powf(self->input_a[0], self->input_b[0]);
}

__global__ void testKernelOneFastPowf(PowTester *self) {
  self->output_fast_powf[0] = __powf(self->input_a[0], self->input_b[0]);
}

__global__ void testKernelOneCustomPow(PowTester *self) {
  self->output_custom[0] = CUSTOM_POW(self->input_a[0], self->input_b[0]);
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
                                        {"quiet", 0, nullptr, 'q'},
                                        {"color", 0, nullptr, 'c'},
                                        {"err", 1, nullptr, 'e'},
                                        {"precision", 1, nullptr, 'p'},
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
      std::cout << "pow_test"
                   " [--verbose]"
                   " [--quiet]"
                   " [--color]"
                   " [--dump-inputs filename]"
                   " [--torcheager torcheagerpow.bin]"
                   " [--torchinductor torchinductorpow.bin]"
                   "\n\n"
                   "Run with:\n"
                   "  pip3 install torch --index-url "
                   "https://download.pytorch.org/whl/cu126\n"
                   "  ./math/pow_test --dump-inputs ./powtest.in\n"
                   "  ../torchpow.py file ./powtest.in\n"
                   "  ./math/pow_test --torchinductor torchinductorpow.bin "
                   "--torcheager torcheagerpow.bin --verbose --quiet --color | less -R\n"
                   "\n\n"
                   "\t--dump-inputs filename.  Write input a,b pairs as binary "
                   "floats to file (a0,b0,a1,b1,...)\n"
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
      std::cerr << "pow_test: unknown option\n";
      return 1;
    }
    }
  }

  // Dump input a,b pairs to binary file if requested
  if (dumpFile) {
    std::ofstream ofs(dumpFile, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file for writing: " << dumpFile << std::endl;
      return 2;
    }
    for (size_t i = 0; i < PowTester::N; ++i) {
      ofs.write(reinterpret_cast<const char *>(&kCases[i].a), sizeof(float));
      ofs.write(reinterpret_cast<const char *>(&kCases[i].b), sizeof(float));
    }
    ofs.close();
    std::cout << "Wrote input a,b pairs to " << dumpFile << std::endl;
    return 0;
  }

  std::vector<float> torchinductorOut, torcheagerOut;
  if (torchinductorFile) {
    torchinductorOut = readBinaryFloatFile(torchinductorFile, PowTester::N);
    if (torchinductorOut.empty())
      torchinductorFile = nullptr;
  }
  if (torcheagerFile) {
    torcheagerOut = readBinaryFloatFile(torcheagerFile, PowTester::N);
    if (torcheagerOut.empty())
      torcheagerFile = nullptr;
  }

  PowTester *tester;
  CUDA_CHECK(cudaMallocManaged(&tester, sizeof(PowTester)));
  new (tester) PowTester();
  tester->reset();

  dim3 blockSize(PowTester::N);
  dim3 gridSize(1);

  testKernelPowf<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  testKernelFastPowf<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  testKernelCustomPow<<<gridSize, blockSize>>>(tester);
  CUDA_CHECK(cudaDeviceSynchronize());

  tester->displayResults(tester->output_powf, tester->output_fast_powf,
                         tester->output_custom,
                         torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  CUDA_CHECK(cudaFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
