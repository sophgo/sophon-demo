//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// Shared helper for the pure bmrt+bmcv path (no sail): upload a contiguous
// 8UC1 host image (already resized to the network's static shape) onto a tight
// FORMAT_GRAY device image and run it via bm_inference (bm_wrapper.hpp). The
// output tensor is copied back into the caller-provided host buffer.
//
// Both wechat_qrcode models take a single 1-channel uint8 input (fused
// --fuse_preprocess bmodels: normalization is baked into the TPU graph) and
// produce a single F32 output, so this helper covers the detect stage and the
// sr stage alike.

#ifndef __WECHAT_QRCODE_INFER_HPP_
#define __WECHAT_QRCODE_INFER_HPP_

#include <cstdint>
#include <vector>

#include "bm_wrapper.hpp"

namespace wechat_qrcode {

// Run `net_name` on a gray 8UC1 mat and return the F32 output base pointer
// (points into out_buf). out_buf is resized to the model output byte size by
// the caller. Returns nullptr on any bm API failure.
inline const float *inferGray(void *p_bmrt, bm_handle_t handle,
                              const cv::Mat &gray, const bm_shape_t &in_shape,
                              const char *net_name,
                              std::vector<uint8_t> &out_buf) {
  const int W = gray.cols;
  const int H = gray.rows;

  // FORMAT_GRAY with DATA_TYPE_EXT_1N_BYTE is 1 byte per pixel; an explicit
  // stride == width keeps the device buffer tight (no row padding), so the
  // model reads exactly H*W contiguous bytes.
  int strides[4] = {W, W, W, W};
  bm_image input;
  if (bm_image_create(handle, H, W, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &input,
                      strides) != BM_SUCCESS)
    return nullptr;
  if (bm_image_alloc_contiguous_mem(1, &input) != BM_SUCCESS) {
    bm_image_destroy(input);
    return nullptr;
  }

  void *host = const_cast<uint8_t *>(gray.data);
  bool ok = false;
  if (bm_image_copy_host_to_device(input, &host) == BM_SUCCESS) {
    ok = bm_inference(p_bmrt, &input, out_buf.data(), in_shape, net_name);
  }

  bm_image_free_contiguous_mem(1, &input);
  bm_image_destroy(input);
  return ok ? reinterpret_cast<const float *>(out_buf.data()) : nullptr;
}

} // namespace wechat_qrcode

#endif // __WECHAT_QRCODE_INFER_HPP_