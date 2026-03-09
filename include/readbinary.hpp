// header only implementation for convienent use in .cu files
#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

// Helper to read binary float file
std::vector<float> readBinaryFloatFile(const char *filename, size_t expected) {
  std::vector<float> data;
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cerr << "Failed to open file for reading: " << filename << '\n';
    return data;
  }
  ifs.seekg(0, std::ios::end);
  size_t sz = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  if ((sz % sizeof(float)) != 0) {
    std::cerr << "File size not a multiple of float size: " << filename << '\n';
    return data;
  }
  size_t n = sz / sizeof(float);
  if (n != expected) {
    std::cerr << "File does not contain expected number of floats: " << filename
              << '\n';
    return data;
  }
  data.resize(n);
  ifs.read(reinterpret_cast<char *>(data.data()), sz);
  return data;
}

// vim: et ts=2 sw=2
