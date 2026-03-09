#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>

struct OneResult32 {
  float v{};
  bool isValid{};
  bool match{};
  int64_t diff{};
  std::string hexString;

  OneResult32(float ref, float value, bool valid, bool verbose)
      : v{value}, isValid{valid} {
    if (!isValid) {
      match = true;
      return;
    }

    match = false;
    if (std::isnan(v) and std::isnan(ref)) {
      match = true;
    } else if (std::isinf(v) and std::isinf(ref) and (v > 0) == (ref > 0)) {
      match = true;
    } else if (ref == 0.0f and v == 0.0f) {
      match = true;
    } else {
      match = (v == ref);
    }

    if (verbose) {
      match = (std::memcmp(&v, &ref, 4) == 0);
      if (std::isnan(v) and std::isnan(ref)) {
        match = true;
      }
    }

    uint32_t rbits;
    uint32_t gbits;
    std::memcpy(&rbits, &ref, sizeof(ref));
    std::memcpy(&gbits, &v, sizeof(v));

    diff = (int32_t)gbits - (int32_t)rbits;

    hexString = std::format("0x{:08x}", gbits);
  }

  std::string errorString(int thresh = 256) const {
    std::string rc = "";
    if (isValid) {
      if (std::abs(diff) >= thresh) {
        rc = "ERROR";
      } else if (!match) {
        uint64_t a = std::abs(diff);
        rc = std::format("{}{} ULP", diff > 0 ? "+" : "-", a);
      }
    }

    return rc;
  }

  std::string value() const {
    if (isValid) {
      return std::format("{:>16g}", v);
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
