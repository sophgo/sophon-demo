//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// WeChatQRCode QR detection + recognition (pure bmrt + bmcv, no sail).
//
// The pipeline matches OpenCV wechat_qrcode::WeChatQRCode:
//   grayscale -> detect.bmodel (TPU, multi-shape) -> crop + padding ->
//   scale (sr.bmodel on TPU for small crops, else cubic/area) ->
//   zxing decode (CPU) -> coordinate remap -> texts + 4-corner points
//
// The detect/sr networks run on the TPU via raw bmrt + bm_image (bm_wrapper.hpp
// bm_inference); zxing decoding runs on the CPU. A single instance is not
// thread-safe (the reusable output buffers are shared); use one instance per
// thread if concurrency is required.

#ifndef __WECHAT_QRCODE_HPP_
#define __WECHAT_QRCODE_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "bm_wrapper.hpp"

namespace wechat_qrcode {

// Per-stage timing stats (accumulated across calls; for FPS/latency
// measurement)
struct BenchStats {
  double detect_ms = 0.0; // detect stage (inference + pre/post, on TPU)
  double sr_ms = 0.0;     // scale/sr stage (processImageScale)
  double zxing_ms = 0.0;  // zxing decode (CPU)
  size_t detect_calls = 0;
  size_t sr_calls = 0;
  size_t zxing_calls = 0;
};

class WeChatQRCode {
public:
  // Create on a device handle and load both bmodels immediately. Throws
  // std::runtime_error (English) if a model fails to load.
  WeChatQRCode(bm_handle_t handle, const std::string &detect_bmodel,
               const std::string &sr_bmodel);
  ~WeChatQRCode();

  WeChatQRCode(const WeChatQRCode &) = delete;
  WeChatQRCode &operator=(const WeChatQRCode &) = delete;

  // Detect and decode an 8-bit (CV_8U) image (grayscale or color), returning
  // QR texts; corners are emitted through points. Each point is a 4x2 CV_32FC1
  // (4 corners x (x,y), original-image coordinates), aligned with OpenCV.
  // Returns an empty vector when no QR code is found, decoding fails, or the
  // input is not uint8 (points is cleared in sync).
  std::vector<std::string>
  detectAndDecode(const cv::Mat &img, std::vector<cv::Mat> *points = nullptr);
  std::vector<std::string>
  detectAndDecode(const std::string &image_path,
                  std::vector<cv::Mat> *points = nullptr);

  // Device-input overload (pure bmrt+bmcv): img is a device-memory uint8
  // bm_image (BGR/RGB packed/planar or GRAY; VPU/decoder output) and handle is
  // the bm_handle_t that produced it (must be on the same device as this
  // instance's handle). detect runs on-device (VPP csc + resize + inference,
  // no full-frame host<->device copy); only each QR crop is copied back for
  // zxing. Returns the same texts/points as the cv::Mat overload.
  std::vector<std::string>
  detectAndDecode(bm_handle_t handle, const bm_image &img,
                  std::vector<cv::Mat> *points = nullptr);

  // Per-stage timing stats
  void resetBenchStats();
  BenchStats getBenchStats() const;

private:
  class Impl;
  std::unique_ptr<Impl> p_;
};

} // namespace wechat_qrcode

#endif // __WECHAT_QRCODE_HPP_