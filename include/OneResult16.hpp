#pragma once

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>

struct OneResult16 {
  __nv_bfloat16 v{};
  bool isValid{};
  bool match{};
  int16_t diff{};
  std::string hexString;

  OneResult16(__nv_bfloat16 ref, __nv_bfloat16 value, bool valid,
              bool verbose)
      : v{value}, isValid{valid} {
    if (!isValid) {
      match = true;
      return;
    }

    float ref32 = __bfloat162float(ref);
    float v32 = __bfloat162float(v);
    match = false;
    if (std::isnan(v32) and std::isnan(ref32)) {
      match = true;
    } else if (std::isinf(v32) and std::isinf(ref32) and (v32 > 0) == (ref32 > 0)) {
      match = true;
    } else if (ref32 == 0.0f and v32 == 0.0f) {
      match = true;
    } else {
      match = (v32 == ref32);
    }

    if (verbose) {
      match = (std::memcmp(&v, &ref, sizeof(v)) == 0);
    }

    uint16_t rbits;
    uint16_t gbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::memcpy(&gbits, &v, sizeof(v));

    diff = (int16_t)gbits - (int16_t)rbits;

    hexString = std::format("0x{:04x}", gbits);
  }

  std::string errorString(int thresh = 256) const {
    std::string rc = "";
    if (isValid) {
      if (std::abs(diff) >= thresh) {
        rc = "ERROR";
      } else if (!match) {
        uint16_t a = std::abs(diff);
        rc = std::format("{}{} ULP", diff > 0 ? "+" : "-", a);
      }
    }

    return rc;
  }

  std::string value() const {
    if (isValid) {
      float v32 = __bfloat162float(v);
      return std::format("{:g}", v32);
    } else {
      return "";
    }
  }

  std::string hexValue() const {
    if (isValid) {
      return hexString;
    } else {
      return "";
    }
  }
};

// vim: et ts=2 sw=2
