#pragma once
#include <cmath>
#include <cstdint>

// clang-format off
//
// math/custom_asm.hpp
//
// CUDA port of Mi300xASMTestBench/complexops/custom_asm.hpp.
//
// Structure mirrors the ROCm original.  For each function, one of:
//   (a) PTX inline-asm  — where a PTX equivalent exists and was tested in
//       the corresponding math/*.cu driver (rcp_test, div_test, sqrtrsq_test,
//       exp2f_test, expf_test).
//   (b) CUDA math-library placeholder — for everything else, marked with
//       "// Placeholder: ..." to make future PTX drop-ins easy to find.
//
// Differences from the ROCm version:
//   - All GCN/RDNA inline-asm blocks removed.
//   - __builtin_bit_cast(uint32_t, x) → __float_as_uint(x) / __uint_as_float
//   - __builtin_sinf/cosf → sinf/cosf (CUDA device math)
//   - __builtin_popcount / __builtin_clz / __builtin_ctz →
//       __popc / __clz / (__ffs-based)
//   - sinpif / cospif use CUDA device builtins directly.
//

//----------------------------------------------------------------------------
// Binary Ops

// Placeholder: NautilusOp atan2
__device__ inline float custom_atan2f(float a, float b) {
  return atan2f(a, b);
}

// Placeholder: NautilusOp mod
__device__ inline float custom_fmodf(float a, float b) {
  return fmodf(a, b);
}

// Placeholder: NautilusOp mod (via remainder)
__device__ inline float custom_remainderf(float a, float b) {
  return remainderf(a, b);
}

//----------------------------------------------------------------------------
// Nautilus compound/select stubs.

// Placeholder: NautilusOp fldiv  y = a // b
inline float __device__ custom_fldivf(float a, float b) {
  return floorf(a / b);
}

// Placeholder: NautilusOp cldiv  y = -(-a // b)
inline float __device__ custom_cldivf(float a, float b) {
  return -floorf((-a) / b);
}

//----------------------------------------------------------------------------
// Trig

// Placeholder: NautilusOp sin
inline float __device__ custom_sinf(float x) {
  return sinf(x);
}

// Placeholder: NautilusOp cos
inline float __device__ custom_cosf(float x) {
  return cosf(x);
}

// Placeholder: NautilusOp tan
inline float __device__ custom_tanf(float x) {
  return tanf(x);
}

// Placeholder: NautilusOp acos
inline float __device__ custom_acosf(float x) {
  return acosf(x);
}

// Placeholder: NautilusOp asin
inline float __device__ custom_asinf(float x) {
  return asinf(x);
}

// Placeholder: NautilusOp atan
inline float __device__ custom_atanf(float x) {
  return atanf(x);
}

// Placeholder: NautilusOp sinpif
// CUDA provides sinpif() natively.
inline float __device__ custom_sinpif(float x) {
  return sinpif(x);
}

// Placeholder: NautilusOp cospif
// CUDA provides cospif() natively.
inline float __device__ custom_cospif(float x) {
  return cospif(x);
}

// Placeholder: NautilusOp tanpif
inline float __device__ custom_tanpif(float x) {
  return tanf(3.14159265358979323846f * x);
}

//----------------------------------------------------------------------------
// Misc.

// Placeholder: NautilusOp erf
inline float __device__ custom_erff(float x) {
  return erff(x);
}

// Placeholder: NautilusOp erfc
inline float __device__ custom_erfcf(float x) {
  return erfcf(x);
}

// Placeholder: NautilusOp lgamma
inline float __device__ custom_lgammaf(float x) {
  return lgammaf(x);
}

// Placeholder: NautilusOp tgamma
inline float __device__ custom_tgammaf(float x) {
  return tgammaf(x);
}

//----------------------------------------------------------------------------
// Log and Exponent related

// Placeholder: NautilusOp expm1
inline float __device__ custom_expm1f(float x) {
  return expm1f(x);
}

// Placeholder: NautilusOp log1p
inline float __device__ custom_log1pf(float x) {
  return log1pf(x);
}

// Placeholder: NautilusOp logb
inline float __device__ custom_logbf(float x) {
  return logbf(x);
}

//----------------------------------------------------------------------------
// Hyperbolic trig

// Placeholder: NautilusOp tanh
inline float __device__ custom_tanhf(float x) {
  return tanhf(x);
}

// Placeholder: NautilusOp acosh
inline float __device__ custom_acoshf(float x) {
  return acoshf(x);
}

// Placeholder: NautilusOp asinh
inline float __device__ custom_asinhf(float x) {
  return asinhf(x);
}

// Placeholder: NautilusOp atanh
inline float __device__ custom_atanhf(float x) {
  return atanhf(x);
}

// Placeholder: NautilusOp sinh
inline float __device__ custom_sinhf(float x) {
  return sinhf(x);
}

// Placeholder: NautilusOp cosh
inline float __device__ custom_coshf(float x) {
  return coshf(x);
}

//----------------------------------------------------------------------------
// Integer operations.
//

// NautilusOp (slides): countones y a — count of set bits in a
// PTX: popc.b32
inline uint32_t __device__ custom_popcount32(uint32_t x) {
  return __popc(x);
}

// Placeholder: 64-bit popcount
inline uint32_t __device__ custom_popcount64(uint64_t x) {
  return __popcll(x);
}

// NautilusOp (slides): countzero — count leading zeros in a
// PTX: bfind.u32 (via __clz intrinsic which lowers to clz.b32)
inline uint32_t __device__ custom_clz32(uint32_t x) {
  return __clz(x);
}

// Placeholder: count trailing zeros in a
// CUDA has no direct __ctz; use __ffs (1-indexed find-first-set) minus 1.
// __ffs(0) == 0, so ctz(0) returns (uint32_t)(-1) — same as __builtin_ctz(0) UB.
inline uint32_t __device__ custom_ctz32(uint32_t x) {
  return (uint32_t)(__ffs((int)x) - 1);
}

//----------------------------------------------------------------------------
// Unary ops (except for custom_fdividef, which is used to implement custom_rcp)

// Placeholder: NautilusOp cbrt
// cbrtf via C library; future PTX: log2/exp2 approximation + Newton-Raphson.
inline float __device__ custom_cbrtf(float x) {
  return cbrtf(x);
}

// NautilusOp: log2
// PTX lg2.approx.f32 — ~23-bit accurate, hardware single-cycle on SM_80+.
inline float __device__ custom_log2f(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = log2(%1)\n\t"
      "lg2.approx.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// Placeholder: NautilusOp log10
inline float __device__ custom_log10f(float x) {
  return log10f(x);
}

// Placeholder: NautilusOp ln
inline float __device__ custom_logf(float x) {
  return logf(x);
}

// NautilusOp: floor
// PTX cvt.rmi.f32.f32 — round-to-minus-infinity conversion (floor).
inline float __device__ custom_floorf(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = floor(%1)\n\t"
      "cvt.rmi.f32.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// NautilusOp: trunc
// PTX cvt.rzi.f32.f32 — round-to-zero (truncate toward zero).
inline float __device__ custom_truncf(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = trunc(%1)\n\t"
      "cvt.rzi.f32.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// NautilusOp: ceil
// PTX cvt.rpi.f32.f32 — round-to-plus-infinity (ceiling).
inline float __device__ custom_ceilf(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = ceil(%1)\n\t"
      "cvt.rpi.f32.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// NautilusOp: abs
// PTX abs.f32 — clears sign bit, preserves NaN payload.
inline float __device__ custom_absf(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = |%1|\n\t"
      "abs.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// NautilusOp: nearbyint
// PTX cvt.rni.f32.f32 — round-to-nearest-even (banker's rounding).
inline float __device__ custom_nearbyintf(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = nearbyint(%1)\n\t"
      "cvt.rni.f32.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// Placeholder: NautilusOp round (round-half-away-from-zero)
// CUDA roundf() implements this correctly.
inline float __device__ custom_roundf(float x) {
  return roundf(x);
}

// NautilusOp: exp2
// PTX ex2.approx.f32 — hardware 2^x approximation (~23-bit accurate).
// From math/exp2f_test.cu.
inline float __device__ custom_exp2f(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = 2^%1\n\t"
      "ex2.approx.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// Placeholder: NautilusOp exp10
// CUDA provides exp10f() natively.
inline float __device__ custom_exp10f(float x) {
  return exp10f(x);
}

// NautilusOp: exp  (e^x)
// Full PTX implementation from math/expf_test.cu.
//
// Algorithm: exp(x) = 2^(x * log2(e)), decomposed as 2^n * 2^frac.
//   1. Map x into [0,1] via fma + cvt.sat (handles over/underflow branchlessly).
//   2. Magic-number round to extract integer exponent n.
//   3. Cody-Waite two-constant range reduction to isolate frac part.
//   4. Construct 2^(n-126) by bit-shifting n into the IEEE exponent field.
//   5. ex2.approx.ftz.f32 for 2^frac, then multiply.
//
// Constants:
//   0f3BBB989D = log2(e) / 252    (maps x into [0,1] for saturation)
//   0f3F000000 = 0.5f             (saturation center)
//   0f437C0000 = 252.0f           (scale back to float exponent range)
//   0f4B400001 = 12582913.0f      (magic round-to-integer bias)
//   0fCB40007F = -12583039.0f     (= -(12582913 + 126) for Cody-Waite)
//   0f3FB8AA3B = log2(e)_hi
//   0f32A57060 = log2(e)_lo       (low-order Cody-Waite correction)
inline float __device__ custom_expf(float x) {
  float result, tmp_f;
  int tmp_i;

  __asm__ __volatile__(
      // Step 1: map x into [0,1] range with overflow/underflow clamp
      "fma.rn.f32 %0, %3, 0f3BBB989D, 0f3F000000;\n\t"
      // %0 = x * (log2(e)/252) + 0.5
      "cvt.sat.f32.f32 %0, %0;\n\t"
      // %0 = clamp(%0, 0.0, 1.0)
      // Step 2: magic-number rounding — extract integer exponent n
      "fma.rm.f32 %0, %0, 0f437C0000, 0f4B400001;\n\t"
      // %0 = floor(%0 * 252.0 + 12582913.0); low mantissa bits = n+1
      // Step 3: Cody-Waite range reduction for fractional part
      "add.f32 %1, %0, 0fCB40007F;\n\t"
      // %1 = %0 - 12583039.0 = (n - 126)
      "neg.f32 %1, %1;\n\t"
      // %1 = 126 - n
      "fma.rn.f32 %1, %3, 0f3FB8AA3B, %1;\n\t"
      // %1 = x * log2(e)_hi + (126 - n)
      "fma.rn.f32 %1, %3, 0f32A57060, %1;\n\t"
      // %1 += x * log2(e)_lo  →  frac part of x*log2(e)
      // Step 4: construct 2^(n-126) via IEEE bit manipulation
      "mov.b32 %2, %0;\n\t"
      // %2 = reinterpret magic-rounded float as int32
      "shl.b32 %2, %2, 23;\n\t"
      // %2 <<= 23 — shift n into IEEE 754 exponent field
      "mov.b32 %0, %2;\n\t"
      // %0 = reinterpret as float = 2^(n-126)
      // Step 5: hardware exp2 of fractional part, then combine
      "ex2.approx.ftz.f32 %1, %1;\n\t"
      // %1 = 2^frac  (denorms flushed to zero)
      "mul.f32 %0, %1, %0;\n\t"
      // %0 = 2^frac * 2^(n-126) = exp(x)
      : "=&f"(result),  // %0 — early clobber
        "=&f"(tmp_f),   // %1 — early clobber
        "=r"(tmp_i)     // %2
      : "f"(x)          // %3
      );

  return result;
}

// NautilusOp: sqrt
// PTX sqrt.rn.f32 — correctly-rounded square root (IEEE 754).
// From math/sqrtrsq_test.cu.
inline float __device__ custom_sqrtf(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = sqrt(%1)\n\t"
      "sqrt.rn.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// NautilusOp: rsqrt
// PTX rsqrt.approx.f32 — hardware reciprocal square-root (~23-bit accurate).
// From math/sqrtrsq_test.cu.
inline float __device__ custom_rsqrtf(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = rsqrt(%1)\n\t"
      "rsqrt.approx.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// NautilusOp: div
// PTX div.rn.f32 — correctly-rounded IEEE 754 division.
// From math/div_test.cu.
inline float __device__ custom_fdividef(float a, float b) {
  float result;
  __asm__ __volatile__(
      "// %0 = %1 / %2\n\t"
      "div.rn.f32 %0, %1, %2;"
      : "=f"(result) // %0
      : "f"(a),      // %1
        "f"(b)       // %2
      );
  return result;
}

// NautilusOp: rec (reciprocal)
// PTX rcp.rn.f32 — correctly-rounded reciprocal.
// From math/rcp_test.cu.
inline float __device__ custom_rcp(float x) {
  float result;
  __asm__ __volatile__(
      "// %0 = 1.0f / %1 (round-to-nearest)\n\t"
      "rcp.rn.f32 %0, %1;"
      : "=f"(result) // %0
      : "f"(x)       // %1
      );
  return result;
}

// Placeholder: NautilusOp pow
// Future PTX sketch (positive base only):
//   lg2.approx.f32  tmp, a   // log2(a)
//   mul.f32         tmp, b, tmp
//   ex2.approx.f32  tmp, tmp
// Full special-case handling (negative base, NaN, ±0, ±inf) requires
// the complex sequence documented in Mi300xASMTestBench/complexops/custom_asm.hpp.
inline __device__ float custom_powf(float a, float b) {
  return powf(a, b);
}

//----------------------------------------------------------------------------
// Binary ops:

// Placeholder: NautilusOp copysign
__device__ inline float custom_copysignf(float a, float b) {
  return copysignf(a, b);
}

// Placeholder: NautilusOp max2
// Note: CUDA fmaxf follows IEEE (returns the non-NaN operand when one input
// is NaN).  The ROCm v_max_f32 version first canonicalized NaN inputs, then
// took the maximum — semantics differ for NaN inputs.
__device__ inline float custom_fmaxf(float a, float b) {
  return fmaxf(a, b);
}

// NautilusOp dim  y = max(a - b, 0)
// Implemented directly without library call.
inline float __device__ custom_dimf(float a, float b) {
  return fmaxf(a - b, 0.0f);
}

// Placeholder: NautilusOp min2
// Same NaN-semantics caveat as custom_fmaxf above.
__device__ inline float custom_fminf(float a, float b) {
  return fminf(a, b);
}

// Placeholder: NautilusOp hypot
__device__ inline float custom_hypotf(float a, float b) {
  return hypotf(a, b);
}

// Placeholder: NautilusOp nextafter
__device__ inline float custom_nextafterf(float a, float b) {
  return nextafterf(a, b);
}

// NautilusOp: root  y = b-th root of a
inline float __device__ custom_rootf(float a, float b) {
  return custom_powf(a, 1.0f / b);
}

//----------------------------------------------------------------------------
// Ternary ops:

// NautilusOp: sel  y = s ? a : b
inline float __device__ custom_sel(uint32_t s, float a, float b) {
  return s ? a : b;
}

// NautilusOp: mac  y = a*b + c
inline float __device__ custom_macf(float a, float b, float c) {
  return fmaf(a, b, c);
}

// NautilusOp: clip  y = max(min(a, u), d)
inline float __device__ custom_clipf(float a, float u, float d) {
  return fmaxf(fminf(a, u), d);
}

// clang-format on
