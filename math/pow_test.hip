//
// pow_test.hip
//
// Tests the builtin powf() / __powf() on AMD MI300 hardware, comparing
// results against std::pow() computed on the CPU.
//
// The intent is to:
//   1. Establish ground truth via ROCm's powf() and __powf().
//   2. Provide a slot (PowTester::testKernelCustomPow) where a hand-coded
//      inline-asm pow(a,b) can be dropped in later.
//
// pow(a,b) = 2^(b * log2(a))   for a > 0
//
// The obvious AMD GCN/CDNA sequence is, for a != 0, is:
//   v_log_f32   tmp, a        // log2(a)
//   v_mul_f32   tmp, b, tmp   // b * log2(a)
//   v_exp_f32   tmp, tmp      // 2^(b * log2(a))
//
// Compile:
//   hipcc -O2 -std=c++20 pow_test.hip -o pow_test
//
#include <hip/hip_runtime.h>
#include <getopt.h>
#include <limits>
#include <unistd.h>
#include <cstdint>

#include "readbinary.hpp"
#include "OneResult32.hpp"
#include "colors.hpp"
#include "hipcheck.hpp"

// ---------------------------------------------------------------------------
// Custom pow
// ---------------------------------------------------------------------------
// clang-format off
inline float __device__ custom_pow_too_simple(float a, float b) {
  float result;
  // early clobber required since not all inputs read before first write to %0
  __asm__ __volatile__(
      "v_log_f32_e32 %0, %1\n\t"     // result  = log2(a) -- trans op
      "s_nop 1\n\t"                  // consume the 1-cycle v_log trans hazard
      "v_mul_f32_e32 %0, %2, %0\n\t" // result  = b * log2(a)
      "v_exp_f32_e32 %0, %0\n\t"     // result  = 2^result  = a^b
      : "=&v"(result) // %0
      : "v"(a),        // %1
        "v"(b)         // %2
      );
  return result;
}

// A version with hardcoded s[2:3] clobber, handling the 0^0 special case.
inline float __device__ custom_pow_s23(float a, float b) {
  float result;
  // early clobber required since not all inputs read before first write to %0
  __asm__ __volatile__(
      "v_log_f32_e32 %0, %1\n\t"               // result = log2(a)
      "v_cmp_eq_f32_e32 vcc, 0, %1\n\t"        // vcc = (a == 0)
      "v_mul_f32_e32 %0, %2, %0\n\t"           // result = b * log2(a)
      "v_exp_f32_e32 %0, %0\n\t"               // result = 2^(b*log2(a))
      "v_cmp_eq_f32_e64 s[2:3], 0, %2\n\t"     // s[2:3] = (b == 0)
      "s_and_b64 vcc, vcc, s[2:3]\n\t"         // vcc = (a == 0) && (b == 0)
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc\n\t" // if (a==0 && b==0) result=1.0
                                               // else result
      : "=&v"(result) // %0
      : "v"(a),        // %1
        "v"(b)         // %2
      : "vcc", "s2", "s3");
  return result;
}

// avoiding the hardcoded s[2:3] clobber, also with the 0^0 special case
inline float __device__ custom_pow_0to0(float a, float b) {
  float result;
  uint64_t b_is_zero;
  // Notes:
  // - early clobber for result required since not all inputs read before first
  // write to %0
  // - b_is_zero shouldn't need an early clobber, since a,b,result are all
  // vector register ins/outs and cannot possibly alias b_is_zero
  // - v_log_f32_e32 has a TRANS hazard: the v_cmp_eq_f32_e32 that follows it
  // serves as the
  //   required 1-cycle VALU gap before v_mul_f32_e32 consumes the result
  // - v_exp_f32_e32 has a TRANS hazard: the v_cmp_eq_f32_e64 that follows it
  // serves as the
  //   required 1-cycle VALU gap before v_cndmask_b32_e64 consumes the result
  __asm__ __volatile__(
      "v_log_f32_e32 %0, %2\n\t"               // result = log2(a)
      "v_cmp_eq_f32_e32 vcc, 0, %2\n\t"        // vcc = (a == 0)
      "v_mul_f32_e32 %0, %3, %0\n\t"           // result = b * log2(a)
      "v_exp_f32_e32 %0, %0\n\t"               // result = 2^(b*log2(a))
      "v_cmp_eq_f32_e64 %1, 0, %3\n\t"         // b_is_zero = (b == 0)
      "s_and_b64 vcc, vcc, %1\n\t"             // vcc = (a == 0) && (b == 0)
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc\n\t" // if (a==0 && b==0) result=1.0
                                               // else result
      : "=&v"(result),                         // %0
        "=s"(b_is_zero) // %1 (an aligned scalar register pair)
      : "v"(a),         // %2
        "v"(b)          // %3
      : "vcc");
  return result;
}

// try to generalize 0^0 special case to x^0
inline float __device__ custom_pow_xto0(float a, float b) {
  float result;
  uint64_t b_is_zero;
  // Notes:
  // - early clobber for result required since not all inputs read before first
  // write to %0
  // - b_is_zero shouldn't need an early clobber, since a,b,result are all
  // vector register ins/outs and cannot possibly alias b_is_zero
  // - v_log_f32_e32 has a TRANS hazard: the v_cmp_eq_f32_e32 that follows it
  // serves as the
  //   required 1-cycle VALU gap before v_mul_f32_e32 consumes the result
  // - v_exp_f32_e32 has a TRANS hazard: the nop that follows it seres as the
  //   required 1-cycle VALU gap before v_cndmask_b32_e64 consumes the result
  __asm__ __volatile__(
      "v_log_f32_e32 %0, %1\n\t"        // result = log2(a)
      "v_cmp_eq_f32_e32 vcc, 0, %2\n\t" // vcc = (b == 0)
      "v_mul_f32_e32 %0, %2, %0\n\t"    // result = b * log2(a)
      "v_exp_f32_e32 %0, %0\n\t"        // result = 2^(b*log2(a))
      "s_nop 1\n\t" // consume the 1-cycle v_exp trans hazard
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc\n\t" // if (b==0) result=1.0
                                               // else result
      : "=&v"(result)                          // %0
      : "v"(a),                                // %1
        "v"(b)                                 // %2
      : "vcc");
  return result;
}

// generalize x^0 to handle x^0 or 1^x
//
// Still gets all of the following wrong:
//        a^b       std::powf            powf          __powf             ASM     torch-eager  torch-inductor
// ----------------------------------------------------------------------------------------------------------
//      e^-88     6.05462e-39     6.05462e-39     6.05462e-39               0     6.05462e-39     6.05461e-39
//     (-2)^1              -2              -2              -2            -nan              -2              -2
//     (-2)^2               4               4               4            -nan               4               4
//     (-2)^3              -8              -8              -8            -nan              -8              -8
//     (-2)^4              16              16              16            -nan              16              16
//     (-1)^3              -1              -1              -1            -nan              -1              -1
//     (-1)^4               1               1               1            -nan               1               1
//  (-2)^(-1)            -0.5            -0.5            -0.5            -nan            -0.5            -0.5
//  (-2)^(-2)            0.25            0.25            0.25            -nan            0.25            0.25
//   (-2)^0.5            -nan             nan             nan            -nan             nan             nan
//   (-2)^1.5            -nan             nan             nan            -nan             nan             nan
//   (-2)^0.1            -nan             nan             nan            -nan             nan             nan
//  (-1)^+inf               1               1               1            -nan               1               1
//  (-1)^-inf               1               1               1            -nan               1               1
//   (-inf)^2             inf             inf             inf            -nan             inf             inf
//   (-inf)^3            -inf            -inf            -inf            -nan            -inf            -inf
//  (-inf)^-1              -0              -0              -0            -nan              -0              -0
//  (-inf)^-2               0               0               0            -nan               0               0
//  subnorm^1     1.17549e-38     1.17549e-38     1.17549e-38               0     1.17549e-38     1.17549e-38
// clang-format on
inline float __device__ custom_pow_better_but_still_wrong(float a, float b) {
  float result;
  uint64_t b_is_zero;
  // Notes:
  // - early clobber for result required since not all inputs read before first
  // write to %0
  // - b_is_zero shouldn't need an early clobber, since a,b,result are all
  // vector register ins/outs and cannot possibly alias b_is_zero
  // - v_log_f32_e32 has a TRANS hazard: the v_cmp_eq_f32_e32 that follows it
  // serves as the
  //   required 1-cycle VALU gap before v_mul_f32_e32 consumes the result
  // - v_exp_f32_e32 has a TRANS hazard: the v_cmp_eq_f32_e64 that follows it
  // serves as the
  //   required 1-cycle VALU gap before v_cndmask_b32_e64 consumes the result
  __asm__ __volatile__(
      "v_log_f32_e32 %0, %2\n\t"               // result = log2(a)
      "v_cmp_eq_f32_e32 vcc, 1.0, %2\n\t"      // vcc = (a == 1)
      "v_mul_f32_e32 %0, %3, %0\n\t"           // result = b * log2(a)
      "v_exp_f32_e32 %0, %0\n\t"               // result = 2^(b*log2(a))
      "v_cmp_eq_f32_e64 %1, 0, %3\n\t"         // b_is_zero = (b == 0)
      "s_or_b64 vcc, vcc, %1\n\t"              // vcc = (a == 1) or (b == 0)
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc\n\t" // if (a==0 && b==0) result=1.0
                                               // else result
      : "=&v"(result),                         // %0
        "=s"(b_is_zero) // %1 (an aligned scalar register pair)
      : "v"(a),         // %2
        "v"(b)          // %3
      : "vcc");
  return result;
}

inline float __device__ custom_pow_v1(float a, float b) {
  // For negative a: compute |a|^b, then fix the sign.
  // Sign is negative iff b is an odd integer; if b is not an integer the
  // log/exp path produces NaN naturally (log of negative = NaN), so we only
  // apply the sign fixup when b is an odd integer.
  float abs_a = __builtin_fabsf(a);
  float trunc_b = __builtin_truncf(b);
  bool b_is_integer = (trunc_b == b);
  bool b_is_odd_integer =
      b_is_integer && (__builtin_truncf(b * 0.5f) != b * 0.5f);
  bool a_is_negative = __builtin_signbit(a);

  float sign_factor;
  if (a_is_negative) {
    if (!b_is_integer)
      sign_factor = __builtin_nanf(""); // negative base, non-integer exp -> NaN
    else if (b_is_odd_integer)
      sign_factor = -1.0f;
    else
      sign_factor = 1.0f;
  } else {
    sign_factor = 1.0f;
  }

  float result;
  uint64_t b_is_zero;
  // Notes:
  // - We feed abs_a into the asm so the log/exp path always sees a positive
  // base.
  //   For negative a with non-integer b, abs_a^b is still NaN-free, but the
  //   multiplication by sign_factor (which is +1 in that case) gives the wrong
  //   answer — however the C standard says the result is NaN for negative base
  //   with non-integer exponent, so we need one more fixup for that case.
  //   We handle it by letting the host compute isnan and overriding if needed,
  //   OR by accepting that non-integer negative base -> NaN is already the
  //   mathematically correct answer and sign_factor will be +1 there anyway.
  // - early clobber for result required since not all inputs read before first
  // write to %0
  // - b_is_zero doesn't need early clobber: a, b, result are vector registers,
  //   b_is_zero is a scalar pair — no possible aliasing
  // - v_log_f32_e32 TRANS hazard: v_cmp_eq_f32_e32 serves as the required
  //   1-cycle VALU gap before v_mul_f32_e32 consumes the result
  // - v_exp_f32_e32 TRANS hazard: v_cmp_eq_f32_e64 serves as the required
  //   1-cycle VALU gap before v_cndmask_b32_e64 consumes the result
  __asm__ __volatile__(
      "v_log_f32_e32 %0, %2\n\t"               // result = log2(|a|)
      "v_cmp_eq_f32_e32 vcc, 1.0, %2\n\t"      // vcc = (|a| == 1)
      "v_mul_f32_e32 %0, %3, %0\n\t"           // result = b * log2(|a|)
      "v_exp_f32_e32 %0, %0\n\t"               // result = 2^(b*log2(|a|))
      "v_cmp_eq_f32_e64 %1, 0, %3\n\t"         // b_is_zero = (b == 0)
      "s_or_b64 vcc, vcc, %1\n\t"              // vcc = (|a| == 1) || (b == 0)
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc\n\t" // if so, result = 1.0
      "v_mul_f32_e32 %0, %4, %0\n\t" // apply sign: result *= sign_factor
      : "=&v"(result),               // %0
        "=s"(b_is_zero)              // %1 (aligned scalar register pair)
      : "v"(abs_a),                  // %2
        "v"(b),                      // %3
        "v"(sign_factor)             // %4
      : "vcc");
  return result;
}

inline float __device__ custom_pow_v2(float a, float b) {
  float result;
  uint64_t b_is_zero;

  float trunc_b = __builtin_truncf(b);
  float half_b = __builtin_truncf(b * 0.5f);
  bool b_is_integer = (trunc_b == b);
  bool b_is_odd = b_is_integer & (half_b != b * 0.5f);
  bool a_is_negative = __builtin_signbit(a);
  float abs_a = __builtin_fabsf(a);

  float sign_factor;
  if (a_is_negative) {
    if (!b_is_integer)
      sign_factor = __builtin_nanf("");
    else if (b_is_odd)
      sign_factor = -1.0f;
    else
      sign_factor = 1.0f;
  } else {
    sign_factor = 1.0f;
  }

  // Notes:
  // - early clobber for result required since not all inputs read before first
  // write to %0
  // - b_is_zero doesn't need early clobber: a, b, result are vector registers,
  //   b_is_zero is a scalar pair — no possible aliasing
  // - v_log_f32_e32 TRANS hazard: v_cmp_eq_f32_e32 serves as the required
  //   1-cycle VALU gap before v_mul_f32_e32 consumes the result
  // - v_exp_f32_e32 TRANS hazard: v_cmp_eq_f32_e64 serves as the required
  //   1-cycle VALU gap before v_cndmask_b32_e64 consumes the result
  __asm__ __volatile__(
      "v_log_f32_e32 %0, %2\n\t"          // result = log2(|a|)
      "v_cmp_eq_f32_e32 vcc, 1.0, %2\n\t" // vcc = (|a| == 1)        [trans
                                          // hazard gap for log]
      "v_mul_f32_e32 %0, %3, %0\n\t"      // result = b * log2(|a|)
      "v_exp_f32_e32 %0, %0\n\t"          // result = 2^(b*log2(|a|))
      "v_cmp_eq_f32_e64 %1, 0, %3\n\t" // b_is_zero = (b == 0)    [trans hazard
                                       // gap for exp]
      "s_or_b64 vcc, vcc, %1\n\t"      // vcc = (|a| == 1) || (b == 0)
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc\n\t" // if so, result = 1.0
      "v_mul_f32_e32 %0, %4, %0\n\t"           // apply sign_factor
      : "=&v"(result),                         // %0
        "=s"(b_is_zero)                        // %1
      : "v"(abs_a),                            // %2
        "v"(b),                                // %3
        "v"(sign_factor)                       // %4
      : "vcc");
  return result;
}

// clang-format off
// This doesn't correctly handle negative 0 base:
//
//      Idx               a               b           Label       std::powf            powf          __powf             ASM     torch-eager  torch-inductor
//     ------------------------------------------------------------------------------------------------------------------------
//      80              -0            -0.5     (-0)^(-0.5)             inf             inf             inf             nan             inf             inf
//                                                              0x7f800000      0x7f800000      0x7f800000      0x7fc00000      0x7f800000      0x7f800000
//                                                                                                                   ERROR
//      83              -0             0.5      (-0)^(0.5)               0               0               0             nan               0               0
//                                                              0x00000000      0x00000000      0x00000000      0x7fc00000      0x00000000      0x00000000
//
inline float __device__ custom_pow_broken_neg_0(float a, float b) {
  float result, sign_factor, tmp2, tmp3;
  uint64_t stmp64;

  __asm__ __volatile__(
      ";; --- Compute b_is_odd_integer, used to determine sign_factor ---\n\t"
      "v_mul_f32_e64 %3, %6, 0.5                  ;; %3 = b * 0.5\n\t"
      "v_trunc_f32_e32 %4, %6                     ;; %4 = trunc(b)\n\t"
      "v_cmp_eq_f32_e32 vcc, %6, %4               ;; vcc = (b == trunc(b)) = b_is_integer\n\t"
      "v_trunc_f32_e32 %4, %3                     ;; %4 = trunc(b * 0.5)\n\t"
      "v_cmp_neq_f32_e64 %5, %4, %3               ;; %5 = (trunc(b*0.5) != b*0.5) = b_is_odd_when_integer\n\t"
      "s_and_b64 %5, vcc, %5                      ;; %5 = b_is_integer && b_is_odd_when_integer = b_is_odd_integer\n\t"
      ";; --- Compute sign_factor into %2 ---\n\t"
      "v_cndmask_b32_e64 %3, 1.0, -1.0, %5        ;; %3 = b_is_odd_integer ? -1.0 : 1.0\n\t"
      "v_mov_b32_e32 %2, 0x7fc00000               ;; NaN\n\t"
      "v_cndmask_b32_e64 %2, %2, %3, vcc          ;; %2 = b_is_integer ? (b_is_odd ? -1 : 1) : NaN\n\t"
      "v_cmp_gt_i32_e64 %5, 0, %1                 ;; %5 = a_is_negative (sign bit check, no readlane)\n\t"
      "v_and_b32_e32 %1, 0x7fffffff, %1           ;; %1 = |a| (clear sign bit)\n\t"
      "v_cndmask_b32_e64 %2, 1.0, %2, %5          ;; %2 = a_is_negative ? %2 : 1.0 = sign_factor\n\t"
      ";; --- log/exp core ---\n\t"
      "v_log_f32_e32 %0, %1                       ;; %0 = log2(|a|)  [TRANS op]\n\t"
      "v_cmp_eq_f32_e32 vcc, 1.0, %1              ;; vcc = (|a| == 1.0)  [TRANS hazard gap for log2]\n\t"
      "v_mul_f32_e32 %0, %6, %0                   ;; %0 = b * log2(|a|)\n\t"
      "v_exp_f32_e32 %0, %0                       ;; %0 = 2^(b * log2(|a|)) = |a|^b  [TRANS op]\n\t"
      "v_cmp_eq_f32_e64 %5, 0, %6                 ;; %5 = (b == 0)  [TRANS hazard gap for exp]\n\t"
      "s_or_b64 vcc, vcc, %5                      ;; vcc = (|a| == 1.0) || (b == 0)\n\t"
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc         ;; %0 = vcc ? 1.0 : |a|^b\n\t"
      "v_mul_f32_e32 %0, %2, %0                   ;; %0 = sign_factor * |a|^b = result\n\t"
      : "=&v"(result),      // %0
        "+v"(a),            // %1
        "=&v"(sign_factor), // %2
        "=&v"(tmp2),        // %3
        "=&v"(tmp3),        // %4
        "=s"(stmp64)        // %5
      : "v"(b)              // %6
      : "vcc");
  return result;
}

// This version is functionally correct, but uses a pair of scalar register pairs, which is
// potentially desirable to avoid:
//
// - 4 32-bit vector outputs, 2 2x32 scalar register outputs, 1 32-bit in/out, 1 input: 10 total.
// - 24 instructions
inline float __device__ custom_pow_two_scalar_tempregpairs_vcc(float a, float b) {
  float result, sign_factor, tmp2, tmp3;
  uint64_t stmp64, stmp64b;

  __asm__ __volatile__(
      ";; --- Compute b_is_odd_integer, used to determine sign_factor ---\n\t"
      "v_mul_f32_e64 %3, %7, 0.5                  ;; %3 = b * 0.5\n\t"
      "v_trunc_f32_e32 %4, %7                     ;; %4 = trunc(b)\n\t"
      "v_cmp_eq_f32_e32 vcc, %7, %4               ;; vcc = (b == trunc(b)) = b_is_integer\n\t"
      "v_trunc_f32_e32 %4, %3                     ;; %4 = trunc(b * 0.5)\n\t"
      "v_cmp_neq_f32_e64 %5, %4, %3               ;; %5 = (trunc(b*0.5) != b*0.5) = b_is_odd_when_integer\n\t"
      "s_and_b64 %5, vcc, %5                      ;; %5 = b_is_integer && b_is_odd_when_integer = b_is_odd_integer\n\t"
      "s_mov_b64 %6, %5                           ;; %6 = saved b_is_odd_integer\n\t"
      ";; --- Compute sign_factor into %2 ---\n\t"
      "v_cndmask_b32_e64 %3, 1.0, -1.0, %5        ;; %3 = b_is_odd_integer ? -1.0 : 1.0\n\t"
      "v_mov_b32_e32 %2, 0x7fc00000               ;; NaN\n\t"
      "v_cndmask_b32_e64 %2, %2, %3, vcc          ;; %2 = b_is_integer ? (b_is_odd ? -1 : 1) : NaN\n\t"
      "v_cmp_gt_i32_e64 %5, 0, %1                 ;; %5 = a_is_negative (sign bit set)\n\t"
      "v_and_b32_e32 %1, 0x7fffffff, %1           ;; %1 = |a|\n\t"
      "v_cmp_eq_f32_e64 vcc, 0.0, %1              ;; vcc = a_is_zero\n\t"
      "s_andn2_b64 vcc, vcc, %6                   ;; vcc = a_is_zero && !b_is_odd_integer\n\t"
      "s_andn2_b64 %5, %5, vcc                    ;; %5 = a_is_negative && !(a_is_zero && !b_is_odd_integer)\n\t"
      "v_cndmask_b32_e64 %2, 1.0, %2, %5          ;; %2 = a_is_negative ? %2 : 1.0 = sign_factor\n\t"
      ";; --- log/exp core ---\n\t"
      "v_log_f32_e32 %0, %1                       ;; %0 = log2(|a|)  [TRANS op]\n\t"
      "v_cmp_eq_f32_e32 vcc, 1.0, %1              ;; vcc = (|a| == 1.0)  [TRANS hazard gap for log2]\n\t"
      "v_mul_f32_e32 %0, %7, %0                   ;; %0 = b * log2(|a|)\n\t"
      "v_exp_f32_e32 %0, %0                       ;; %0 = 2^(b * log2(|a|)) = |a|^b  [TRANS op]\n\t"
      "v_cmp_eq_f32_e64 %5, 0, %7                 ;; %5 = (b == 0)  [TRANS hazard gap for exp]\n\t"
      "s_or_b64 vcc, vcc, %5                      ;; vcc = (|a| == 1.0) || (b == 0)\n\t"
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc         ;; %0 = vcc ? 1.0 : |a|^b\n\t"
      "v_mul_f32_e32 %0, %2, %0                   ;; %0 = sign_factor * |a|^b = result\n\t"
      : "=&v"(result),      // %0
        "+v"(a),            // %1
        "=&v"(sign_factor), // %2
        "=&v"(tmp2),        // %3
        "=&v"(tmp3),        // %4
        "=s"(stmp64),       // %5
        "=s"(stmp64b)       // %6
      : "v"(b)              // %7
      : "vcc");
  return result;
}

// variant without use of vcc (using an additional scalar temporary register pair.)
inline float __device__ custom_pow_two_scalar_tempregpairs_stemp(float a, float b) {
  float result, sign_factor, tmp2, tmp3;
  uint64_t stmp64, stmp64b, stmp64c;

  __asm__ __volatile__(
      ";; --- Compute b_is_odd_integer, used to determine sign_factor ---\n\t"
      "v_mul_f32_e64 %3, %8, 0.5                  ;; %3 = b * 0.5\n\t"
      "v_trunc_f32_e32 %4, %8                     ;; %4 = trunc(b)\n\t"
      "v_cmp_eq_f32 %7, %8, %4                    ;; stmp64c = (b == trunc(b)) = b_is_integer\n\t"
      "v_trunc_f32_e32 %4, %3                     ;; %4 = trunc(b * 0.5)\n\t"
      "v_cmp_neq_f32_e64 %5, %4, %3               ;; %5 = (trunc(b*0.5) != b*0.5) = b_is_odd_when_integer\n\t"
      "s_and_b64 %5, %7, %5                       ;; %5 = b_is_integer && b_is_odd_when_integer = b_is_odd_integer\n\t"
      "s_mov_b64 %6, %5                           ;; %6 = saved b_is_odd_integer\n\t"
      ";; --- Compute sign_factor into %2 ---\n\t"
      "v_cndmask_b32_e64 %3, 1.0, -1.0, %5        ;; %3 = b_is_odd_integer ? -1.0 : 1.0\n\t"
      "v_mov_b32_e32 %2, 0x7fc00000               ;; NaN\n\t"
      "v_cndmask_b32 %2, %2, %3, %7               ;; %2 = b_is_integer ? (b_is_odd ? -1 : 1) : NaN\n\t"
      "v_cmp_gt_i32_e64 %5, 0, %1                 ;; %5 = a_is_negative (sign bit set)\n\t"
      "v_and_b32_e32 %1, 0x7fffffff, %1           ;; %1 = |a|\n\t"
      "v_cmp_eq_f32 %7, 0.0, %1                   ;; stmp64c = a_is_zero\n\t"
      "s_andn2_b64 %7, %7, %6                     ;; stmp64c = a_is_zero && !b_is_odd_integer\n\t"
      "s_andn2_b64 %5, %5, %7                     ;; %5 = a_is_negative && !(a_is_zero && !b_is_odd_integer)\n\t"
      "v_cndmask_b32_e64 %2, 1.0, %2, %5          ;; %2 = a_is_negative ? %2 : 1.0 = sign_factor\n\t"
      ";; --- log/exp core ---\n\t"
      "v_log_f32_e32 %0, %1                       ;; %0 = log2(|a|)  [TRANS op]\n\t"
      "v_cmp_eq_f32 %7, 1.0, %1                   ;; stmp64c = (|a| == 1.0)  [TRANS hazard gap for log2]\n\t"
      "v_mul_f32_e32 %0, %8, %0                   ;; %0 = b * log2(|a|)\n\t"
      "v_exp_f32_e32 %0, %0                       ;; %0 = 2^(b * log2(|a|)) = |a|^b  [TRANS op]\n\t"
      "v_cmp_eq_f32_e64 %5, 0, %8                 ;; %5 = (b == 0)  [TRANS hazard gap for exp]\n\t"
      "s_or_b64 %7, %7, %5                        ;; stmp64c = (|a| == 1.0) || (b == 0)\n\t"
      "v_cndmask_b32 %0, %0, 1.0, %7              ;; %0 = stmp64c ? 1.0 : |a|^b\n\t"
      "v_mul_f32_e32 %0, %2, %0                   ;; %0 = sign_factor * |a|^b = result\n\t"
      : "=&v"(result),      // %0
        "+v"(a),            // %1
        "=&v"(sign_factor), // %2
        "=&v"(tmp2),        // %3
        "=&v"(tmp3),        // %4
        "=&s"(stmp64),      // %5
        "=&s"(stmp64b),     // %6
        "=&s"(stmp64c)      // %7
      : "v"(b)              // %8
      : /* no clobbers */ );
  return result;
}

// A variant of custom_pow_two_scalar_tempregpairs, using only a single pair of scalar registers
// and everything else in vector registers:
//
// - 6 32-bit vector outputs, 2x 32-bit scalar register, 1 32-bit in/out, 1 32-bit input: 10 total
// - 27 instructions total (+3 relative to custom_pow_two_scalar_tempregpairs)
inline float __device__ custom_pow_one_scalar_tempregpairs(float a, float b) {
  float result, sign_factor, tmp2, tmp3, b_is_odd_vec, a_is_neg_vec;
  uint64_t stmp64;

  __asm__ __volatile__(
      ";; --- Compute b_is_odd_integer ---\n\t"
      "v_mul_f32_e64 %3, %8, 0.5                  ;; %3 = b * 0.5\n\t"
      "v_trunc_f32_e32 %4, %8                     ;; %4 = trunc(b)\n\t"
      "v_cmp_eq_f32_e32 vcc, %8, %4               ;; vcc = b_is_integer\n\t"
      "v_trunc_f32_e32 %4, %3                     ;; %4 = trunc(b * 0.5)\n\t"
      "v_cmp_neq_f32_e64 %5, %4, %3               ;; %5 = b_is_odd_when_integer\n\t"
      "s_and_b64 %5, vcc, %5                      ;; %5 = b_is_odd_integer\n\t"
      "v_cndmask_b32_e64 %6, 0.0, 1.0, %5         ;; %6 = b_is_odd_integer spilled to vector\n\t"
      ";; --- Compute sign_factor into %2 ---\n\t"
      "v_cndmask_b32_e64 %3, 1.0, -1.0, %5        ;; %3 = b_is_odd ? -1.0 : 1.0\n\t"
      "v_mov_b32_e32 %2, 0x7fc00000               ;; NaN\n\t"
      "v_cndmask_b32_e64 %2, %2, %3, vcc          ;; %2 = b_is_integer ? (b_is_odd ? -1 : 1) : NaN\n\t"
      ";; vcc free, %5 free\n\t"
      ";; --- Negative base / -0 handling ---\n\t"
      "v_cmp_gt_i32_e64 %5, 0, %1                 ;; %5 = a_is_negative (raw bits, sign intact)\n\t"
      "v_cndmask_b32_e64 %7, 0.0, 1.0, %5         ;; %7 = a_is_negative spilled to vector\n\t"
      "v_and_b32_e32 %1, 0x7fffffff, %1           ;; %1 = |a|\n\t"
      "v_cmp_eq_f32_e64 %5, 0.0, %1               ;; %5 = a_is_zero\n\t"
      "v_cmp_neq_f32_e64 vcc, 1.0, %6             ;; vcc = !b_is_odd_integer (from vector spill)\n\t"
      "s_and_b64 %5, %5, vcc                      ;; %5 = a_is_zero && !b_is_odd_integer\n\t"
      "v_cmp_eq_f32_e64 vcc, 1.0, %7              ;; vcc = a_is_negative (from vector spill)\n\t"
      "s_andn2_b64 vcc, vcc, %5                   ;; vcc = a_is_negative && !(a_is_zero && !b_is_odd)\n\t"
      "v_cndmask_b32_e64 %2, 1.0, %2, vcc         ;; %2 = sign_factor (1.0 if positive/even-zero base)\n\t"
      ";; --- log/exp core ---\n\t"
      "v_log_f32_e32 %0, %1                       ;; %0 = log2(|a|)  [TRANS op]\n\t"
      "v_cmp_eq_f32_e32 vcc, 1.0, %1              ;; vcc = (|a| == 1.0)  [TRANS hazard gap for log2]\n\t"
      "v_mul_f32_e32 %0, %8, %0                   ;; %0 = b * log2(|a|)\n\t"
      "v_exp_f32_e32 %0, %0                       ;; %0 = 2^(b * log2(|a|)) = |a|^b  [TRANS op]\n\t"
      "v_cmp_eq_f32_e64 %5, 0, %8                 ;; %5 = (b == 0)  [TRANS hazard gap for exp]\n\t"
      "s_or_b64 vcc, vcc, %5                      ;; vcc = (|a| == 1.0) || (b == 0)\n\t"
      "v_cndmask_b32_e64 %0, %0, 1.0, vcc         ;; %0 = vcc ? 1.0 : |a|^b\n\t"
      "v_mul_f32_e32 %0, %2, %0                   ;; %0 = sign_factor * |a|^b = result\n\t"
      : "=&v"(result),       // %0
        "+v"(a),             // %1
        "=&v"(sign_factor),  // %2
        "=&v"(tmp2),         // %3
        "=&v"(tmp3),         // %4
        "=s"(stmp64),        // %5
        "=&v"(b_is_odd_vec), // %6
        "=&v"(a_is_neg_vec)  // %7
      : "v"(b)               // %8
      : "vcc");
  return result;
}
// clang-format on

#ifdef DILUVIAN_FAST_MATH
#define CUSTOM_POW custom_pow_too_simple
#else
//#define CUSTOM_POW custom_pow_two_scalar_tempregpairs_vcc
#define CUSTOM_POW custom_pow_two_scalar_tempregpairs_stemp
#endif
// #define CUSTOM_POW custom_pow_one_scalar_tempregpairs

// ---------------------------------------------------------------------------
// Test-case table
// ---------------------------------------------------------------------------
struct PowCase {
  float a;
  float b;
  const char *label;
};

// Keep in sync with PowTester::N below.
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
};
static_assert(sizeof(kCases) / sizeof(kCases[0]) == 201,
              "Update PowTester::N to match kCases length");

bool verbose{};
bool quiet{};
bool useColor{};

// ---------------------------------------------------------------------------
// Tester class
// ---------------------------------------------------------------------------
class PowTester {
public:
  static constexpr size_t N = 201;

  float input_a[N];
  float input_b[N];
  float output_powf[N];      // ROCm powf()
  float output_fast_powf[N]; // ROCm __powf() (lower precision)
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

  // -- kernels ---------------------------------------------------------------

  __global__ static void testKernelPowf(PowTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
      self->output_powf[idx] = powf(self->input_a[idx], self->input_b[idx]);
  }

  __global__ static void testKernelFastPowf(PowTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
      self->output_fast_powf[idx] =
          __powf(self->input_a[idx], self->input_b[idx]);
  }

  __global__ static void testKernelCustomPow(PowTester *self) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
      self->output_custom[idx] =
          CUSTOM_POW(self->input_a[idx], self->input_b[idx]);
  }

  // -- kernels for just looking at asm listings ------------------------------

  __global__ static void testKernelOnePowf(PowTester *self) {
    self->output_powf[0] = powf(self->input_a[0], self->input_b[0]);
  }

  __global__ static void testKernelOneFastPowf(PowTester *self) {
    self->output_fast_powf[0] = __powf(self->input_a[0], self->input_b[0]);
  }

  __global__ static void testKernelOneCustomPow(PowTester *self) {
    self->output_custom[0] = CUSTOM_POW(self->input_a[0], self->input_b[0]);
  }

  // -- display ---------------------------------------------------------------

  __host__ void displayResults(const float *RCOm_powf, const float *ROCm___powf,
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
      float ref = std::powf(a, b);

      OneResult32 v_powf(ref, RCOm_powf[i], true, verbose);
      OneResult32 v__powf(ref, ROCm___powf[i], true, verbose);
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
                   "https://download.pytorch.org/whl/rocm6.3\n"
                   "  ./pow_test --dump-inputs ./powtest.in\n"
                   "  ../torchpow.py file ./powtest.in\n"
                   "  ./pow_test --torchinductor torchinductorpow.bin "
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
  HIP_CHECK(hipMallocManaged(&tester, sizeof(PowTester)));
  new (tester) PowTester();
  tester->reset();

  dim3 blockSize(PowTester::N);
  dim3 gridSize(1);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(PowTester::testKernelPowf), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  hipLaunchKernelGGL(HIP_KERNEL_NAME(PowTester::testKernelFastPowf), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  hipLaunchKernelGGL(HIP_KERNEL_NAME(PowTester::testKernelCustomPow), gridSize,
                     blockSize, 0, 0, tester);
  HIP_CHECK(hipDeviceSynchronize());

  tester->displayResults(tester->output_powf, tester->output_fast_powf,
                         tester->output_custom,
                         torchinductorFile ? torchinductorOut.data() : nullptr,
                         torcheagerFile ? torcheagerOut.data() : nullptr);

  HIP_CHECK(hipFree(tester));
  return 0;
}

// vim: et ts=2 sw=2
