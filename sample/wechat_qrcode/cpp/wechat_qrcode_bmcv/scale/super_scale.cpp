//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// WeChatQRCode sr (super-resolution) model inference wrapper (pure bmrt+bmcv, no
// sail)

#include "super_scale.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "../infer.hpp"

namespace wechat_qrcode {

namespace {

size_t dtypeBytes(bm_data_type_t t) {
  switch (t) {
  case BM_FLOAT32:
  case BM_INT32:
  case BM_UINT32:
    return 4;
  case BM_FLOAT16:
  case BM_BFLOAT16:
  case BM_INT16:
  case BM_UINT16:
    return 2;
  default:
    return 1;
  }
}

} // namespace

SuperScale::~SuperScale() {
  if (p_bmrt_ != nullptr) {
    bmrt_destroy(p_bmrt_);
    p_bmrt_ = nullptr;
  }
}

int SuperScale::init(bm_handle_t handle, const std::string &bmodel_path) {
  handle_ = handle;
  p_bmrt_ = bmrt_create(handle);
  if (p_bmrt_ == nullptr)
    return -1;
  if (!bmrt_load_bmodel(p_bmrt_, bmodel_path.c_str()))
    return -1;

  const char **net_names = nullptr;
  bmrt_get_network_names(p_bmrt_, &net_names);
  graph_name_ = net_names[0];

  const bm_net_info_t *info =
      bmrt_get_network_info(p_bmrt_, graph_name_.c_str());
  if (info == nullptr)
    return -1;
  if (info->stage_num > 0)
    input_shape_ = info->stages[0].input_shapes[0];

  const size_t out_count = (info->stage_num > 0)
                               ? static_cast<size_t>(bmrt_shape_count(
                                     &info->stages[0].output_shapes[0]))
                               : static_cast<size_t>(kSrOutputSize) *
                                     kSrOutputSize;
  out_buf_.resize(out_count * dtypeBytes(info->output_dtypes[0]));
  return 0;
}

cv::Mat SuperScale::processImageScale(const cv::Mat &src, float scale,
                                      bool use_sr, int sr_max_size) {
  cv::Mat dst = src;
  if (scale == 1.0f) {
    return dst;
  }

  const int width = src.cols;
  const int height = src.rows;
  if (scale == 2.0f) {
    // Route only small crops through the sr network (same sr_max_size=160
    // semantics as OpenCV)
    if (use_sr && p_bmrt_ &&
        static_cast<int>(std::sqrt(static_cast<double>(width) * height)) <
            sr_max_size) {
      if (superResoutionScale(src, dst) == 0) {
        return dst;
      }
    }
    cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_CUBIC);
  } else if (scale < 1.0f) {
    cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_AREA);
  }

  return dst;
}

int SuperScale::superResoutionScale(const cv::Mat &src, cv::Mat &dst) {
  // fixed resize to 224x224; then resize the 447x447 output back to 2x the
  // original crop size.
  cv::Mat input;
  cv::resize(src, input, cv::Size(kSrInputSize, kSrInputSize), 0, 0,
             cv::INTER_CUBIC);

  const float *out_data =
      inferGray(p_bmrt_, handle_, input, input_shape_, graph_name_.c_str(),
                out_buf_);
  if (out_data == nullptr)
    return -1;

  cv::Mat sr_out(kSrOutputSize, kSrOutputSize, CV_8UC1);
  uint8_t *dst_data = sr_out.data;
  const int out_n = kSrOutputSize * kSrOutputSize;
  for (int i = 0; i < out_n; ++i) {
    dst_data[i] = cv::saturate_cast<uint8_t>(out_data[i] * 255.0f);
  }

  cv::resize(sr_out, dst, cv::Size(src.cols * 2, src.rows * 2), 0, 0,
             cv::INTER_CUBIC);
  return 0;
}

} // namespace wechat_qrcode