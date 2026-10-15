//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef SAM3_NPZ_LOADER_HPP
#define SAM3_NPZ_LOADER_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// One array of an .npz file (uncompressed zip of .npy members).
// The SAM3 post-processing weight files only contain little-endian '<f4'
// arrays plus one '<i8' scalar (d_proj), all C-order (fortran_order=False).
struct NpzArray {
  std::vector<int64_t> shape;
  std::vector<float> data;  // '<f4'
  int64_t int_scalar = 0;   // '<i8'
  bool is_float = true;

  size_t count() const {
    size_t n = 1;
    for (int64_t d : shape) n *= (size_t)d;
    return n;
  }
};

using NpzMap = std::map<std::string, NpzArray>;

// Loads every .npy member of `npz_path` into `out`, keyed by the member name
// without the ".npy" suffix. Returns false and sets `err` on failure.
bool loadNpz(const std::string& npz_path, NpzMap& out, std::string& err);

#endif  // SAM3_NPZ_LOADER_HPP
