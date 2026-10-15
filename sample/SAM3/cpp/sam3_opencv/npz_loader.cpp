//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "npz_loader.hpp"

#include <miniz.h>

#include <cstring>

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace {

// Parse one .npy payload: "\x93NUMPY" magic + version + header + raw data.
// Supports format 1.x, dtypes '<f4' / '<i8' (little-endian, C-order).
void parseNpy(const std::string& name, const void* buf, size_t size,
              NpzArray& arr, std::string& err) {
  const unsigned char* p = (const unsigned char*)buf;
  if (size < 10 || memcmp(p, "\x93NUMPY", 6) != 0) {
    err = name + ": not a .npy member";
    return;
  }
  const unsigned major = p[6];
  size_t data_off;
  if (major <= 1) {
    const uint16_t hlen = p[8] | (p[9] << 8);
    data_off = 10 + hlen;
  } else {
    if (size < 12) {
      err = name + ": truncated npy header";
      return;
    }
    const uint32_t hlen = p[8] | (p[9] << 8) | (p[10] << 16) |
                          ((uint32_t)p[11] << 24);
    data_off = 12 + hlen;
  }
  if (data_off > size) {
    err = name + ": bad npy header length";
    return;
  }
  const std::string header((const char*)p + (major <= 1 ? 10 : 12),
                           (major <= 1 ? data_off - 10 : data_off - 12));
  if (header.find("'fortran_order': True") != std::string::npos) {
    err = name + ": fortran_order not supported";
    return;
  }

  // descr: little-endian '<f4' or '<i8' (accept '=', '|', '<' prefixes).
  const size_t dq = header.find("'descr'");
  if (dq == std::string::npos) {
    err = name + ": no descr in npy header";
    return;
  }
  const size_t vq1 = header.find('\'', dq + 7);
  const size_t vq2 = header.find('\'', vq1 + 1);
  if (vq1 == std::string::npos || vq2 == std::string::npos ||
      vq2 - vq1 < 3) {
    err = name + ": bad descr in npy header";
    return;
  }
  const std::string dt = header.substr(vq2 - 2, 2);  // "f4" / "i8"
  bool is_float;
  size_t elem;
  if (dt == "f4") {
    is_float = true;
    elem = 4;
  } else if (dt == "i8") {
    is_float = false;
    elem = 8;
  } else {
    err = name + ": unsupported dtype " + header.substr(vq1 + 1, vq2 - vq1 - 1);
    return;
  }

  // shape: the list after "'shape':".
  const size_t sq = header.find("'shape'");
  if (sq == std::string::npos) {
    err = name + ": no shape in npy header";
    return;
  }
  size_t b = header.find('(', sq);
  size_t e = header.find(')', sq);
  std::vector<int64_t> shape;
  {
    std::istringstream ls(header.substr(b + 1, e - b - 1));
    std::string tok;
    while (std::getline(ls, tok, ',')) {
      if (!tok.empty() && tok.back() == ' ') tok.pop_back();
      if (tok.empty()) continue;
      shape.push_back(atoll(tok.c_str()));
    }
  }
  if (shape.empty()) shape.push_back(1);  // numpy scalar ()

  const size_t count = [&] {
    size_t n = 1;
    for (int64_t d : shape) n *= (size_t)d;
    return n;
  }();
  if (data_off + count * elem > size) {
    err = name + ": truncated npy data";
    return;
  }

  arr.shape = shape;
  arr.is_float = is_float;
  if (is_float) {
    arr.data.resize(count);
    memcpy(arr.data.data(), p + data_off, count * 4);
  } else {
    // Only used for the d_proj scalar.
    if (count != 1) {
      err = name + ": i8 arrays must be scalars";
      return;
    }
    memcpy(&arr.int_scalar, p + data_off, 8);
  }
}

}  // namespace

bool loadNpz(const std::string& npz_path, NpzMap& out, std::string& err) {
  mz_zip_archive zip;
  memset(&zip, 0, sizeof(zip));
  if (!mz_zip_reader_init_file(&zip, npz_path.c_str(), 0)) {
    err = "cannot open npz " + npz_path;
    return false;
  }

  bool ok = true;
  const mz_uint num = mz_zip_reader_get_num_files(&zip);
  for (mz_uint i = 0; i < num && ok; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st)) {
      err = "zip stat failed at " + std::to_string(i);
      ok = false;
      break;
    }
    std::string name = st.m_filename;
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) {
      name.erase(name.size() - 4);
    }

    size_t size = 0;
    void* data = mz_zip_reader_extract_to_heap(&zip, i, &size, 0);
    if (!data) {
      err = "extract failed: " + name;
      ok = false;
      break;
    }
    NpzArray arr;
    parseNpy(name, data, size, arr, err);
    mz_free(data);
    if (!err.empty()) {
      ok = false;
      break;
    }
    out[name] = std::move(arr);
  }

  mz_zip_reader_end(&zip);
  return ok;
}
