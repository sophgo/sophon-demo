//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// WeChatQRCode sr (super-resolution) model inference wrapper (pure bmrt+bmcv, no
// sail). Mirrors OpenCV wechat_qrcode's SuperScale::processImageScale.

#ifndef __WECHAT_QRCODE_SUPER_SCALE_HPP_
#define __WECHAT_QRCODE_SUPER_SCALE_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "../bm_wrapper.hpp"

namespace wechat_qrcode {

// sr model has a fixed 224x224 input; the caffe deconv+Crop actual output is
// 447x447
constexpr int kSrInputSize = 224;
constexpr int kSrOutputSize = 447;

class SuperScale {
public:
  SuperScale() = default;
  ~SuperScale();

  // Load sr.bmodel onto the given device. Returns 0 on success, non-zero on
  // failure.
  int init(bm_handle_t handle, const std::string &bmodel_path);

  // Scale a grayscale image by `scale`: 1.0 as-is; 2.0 goes through the sr
  // network when the conditions hold, else cubic; <1.0 uses INTER_AREA.
  // Semantics match OpenCV SuperScale::processImageScale.
  cv::Mat processImageScale(const cv::Mat &src, float scale, bool use_sr,
                            int sr_max_size = 160);

private:
  int superResoutionScale(const cv::Mat &src, cv::Mat &dst);

  bm_handle_t handle_ = nullptr;
  void *p_bmrt_ = nullptr; // bmrt context holding the sr bmodel
  std::string graph_name_;
  bm_shape_t input_shape_{};
  std::vector<uint8_t> out_buf_; // reusable host output buffer (F32)
};

} // namespace wechat_qrcode

#endif // __WECHAT_QRCODE_SUPER_SCALE_HPP_