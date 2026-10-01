//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// Device-side bm_image / bm_device_mem_t helpers (shared across BM1684X /
// BM1688 SDKs), used by the device-input detectAndDecode/forward paths.
//
// - wqDestroyImage: bm_image_destroy has a different signature between the
//   BM1688 SDK (BMCV_VERSION_MAJOR==2, pointer-based) and the BM1684X SDK
//   (older, no version macro, pass-by-value). This wrapper reconciles them.
// - wqD2dStrip: copies a device grayscale plane with stride padding
//   (src_stride bytes per row, h rows, w valid columns) row by row into a
//   contiguous device buffer, stripping the padding (like a tight tensor
//   buffer). src_stride == w uses one full copy to avoid per-row API overhead.

#ifndef __WECHAT_QRCODE_BM_UTIL_HPP_
#define __WECHAT_QRCODE_BM_UTIL_HPP_

#include <bmcv_api_ext.h>
#include <bmlib_runtime.h>

#ifndef BMCV_VERSION_MAJOR
#define BMCV_VERSION_MAJOR 1
#endif

namespace wechat_qrcode {
namespace detail {

static inline bm_status_t wqDestroyImage(bm_image img) {
#if BMCV_VERSION_MAJOR >= 2
  return bm_image_destroy(&img); // new SDK: pointer-based
#else
  return bm_image_destroy(img); // old SDK: pass-by-value
#endif
}

static inline bm_status_t wqD2dStrip(bm_handle_t handle, bm_device_mem_t dst,
                                     bm_device_mem_t src, int w, int h,
                                     size_t src_stride) {
  if (src_stride == static_cast<size_t>(w)) {
    return bm_memcpy_d2d_byte(handle, dst, 0, src, 0,
                              static_cast<size_t>(w) * h);
  }
  for (int r = 0; r < h; ++r) {
    bm_status_t st = bm_memcpy_d2d_byte(
        handle, dst, static_cast<size_t>(r) * w, src,
        static_cast<size_t>(r) * src_stride, static_cast<size_t>(w));
    if (st != BM_SUCCESS)
      return st;
  }
  return BM_SUCCESS;
}

} // namespace detail
} // namespace wechat_qrcode

#endif // __WECHAT_QRCODE_BM_UTIL_HPP_