//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// WeChatQRCode detect-model inference wrapper (pure bmrt + bmcv, no sail)

#include "ssd_detector.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "../bm_util.hpp"
#include "../infer.hpp"

namespace wechat_qrcode {

namespace {

// bytes per element for a bm_data_type_t
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
    return 1; // INT8 / UINT8 / INT4 / UINT4
  }
}

// Parse W/H out of "detect_<W>_<H>"; returns false on failure
bool parseGraphName(const std::string &name, int *w, int *h) {
  const std::string prefix = "detect_";
  if (name.compare(0, prefix.size(), prefix) != 0)
    return false;
  const std::string rest = name.substr(prefix.size());
  const size_t sep = rest.find('_');
  if (sep == std::string::npos)
    return false;
  auto to_int = [](const std::string &s, int *out) -> bool {
    if (s.empty())
      return false;
    for (char c : s) {
      if (!std::isdigit(static_cast<unsigned char>(c)))
        return false;
    }
    *out = std::atoi(s.c_str());
    return *out > 0;
  };
  return to_int(rest.substr(0, sep), w) && to_int(rest.substr(sep + 1), h);
}

} // namespace

SSDDetector::~SSDDetector() {
  if (p_bmrt_ != nullptr) {
    bmrt_destroy(p_bmrt_);
    p_bmrt_ = nullptr;
  }
}

int SSDDetector::init(bm_handle_t handle, const std::string &bmodel_path) {
  handle_ = handle;
  p_bmrt_ = bmrt_create(handle);
  if (p_bmrt_ == nullptr)
    return -1;
  if (!bmrt_load_bmodel(p_bmrt_, bmodel_path.c_str()))
    return -1;

  // Enumerate all sub-networks; the shape-table bmodel has one "detect_<W>_<H>"
  // per static shape. A legacy single-shape bmodel falls back to a fixed graph.
  const char **net_names = nullptr;
  bmrt_get_network_names(p_bmrt_, &net_names);
  const int num = bmrt_get_network_number(p_bmrt_);

  const bm_net_info_t *first = nullptr;
  for (int i = 0; i < num; ++i) {
    const std::string gname(net_names[i]);
    if (first == nullptr)
      first = bmrt_get_network_info(p_bmrt_, gname.c_str());

    int w = 0, h = 0;
    if (!parseGraphName(gname, &w, &h))
      continue;
    DetectGraph g;
    g.name = gname;
    g.w = w;
    g.h = h;
    g.aspect = static_cast<double>(w) / static_cast<double>(h);
    const bm_net_info_t *info = bmrt_get_network_info(p_bmrt_, gname.c_str());
    if (info->stage_num > 0)
      g.input_shape = info->stages[0].input_shapes[0];
    graphs_.push_back(g);
  }

  if (first == nullptr)
    return -1;

  // Fallback graph for single-shape bmodels (no "detect_<W>_<H>" names)
  fallback_.name = first->name;
  fallback_.w = kDetectInputSize;
  fallback_.h = kDetectInputSize;
  fallback_.aspect = 1.0;
  if (first->stage_num > 0)
    fallback_.input_shape = first->stages[0].input_shapes[0];

  // Reusable output buffer sized from the (1,1,100,7) F32 output contract
  const size_t out_count = (first->stage_num > 0)
                               ? static_cast<size_t>(bmrt_shape_count(
                                     &first->stages[0].output_shapes[0]))
                               : static_cast<size_t>(kDetectCandidateNum) * 7;
  const size_t out_bytes =
      out_count * dtypeBytes(first->output_dtypes[0]);
  out_buf_.resize(out_bytes);
  return 0;
}

const DetectGraph *SSDDetector::selectGraph(int target_width,
                                            int target_height) const {
  if (graphs_.empty())
    return &fallback_;

  const double a = static_cast<double>(target_width) /
                   static_cast<double>(std::max(1, target_height));
  const DetectGraph *best = nullptr;
  double best_cost = 1e300;
  for (const auto &g : graphs_) {
    const double r = a / g.aspect;
    const double cost = (r >= 1.0) ? r : (1.0 / r);
    if (cost < best_cost) {
      best_cost = cost;
      best = &g;
    }
  }
  return best;
}

std::vector<cv::Mat> SSDDetector::forward(const cv::Mat &img, int target_width,
                                          int target_height) {
  const int img_w = img.cols;
  const int img_h = img.rows;

  const DetectGraph *g = selectGraph(target_width, target_height);
  const int in_w = (g != nullptr) ? g->w : kDetectInputSize;
  const int in_h = (g != nullptr) ? g->h : kDetectInputSize;
  const std::string graph_name = (g != nullptr) ? g->name : fallback_.name;
  const bm_shape_t in_shape =
      (g != nullptr) ? g->input_shape : fallback_.input_shape;

  // Proportional resize to the selected network's static shape (INTER_CUBIC,
  // the same as OpenCV), then run the fused uint8 model on device
  cv::Mat input;
  cv::resize(img, input, cv::Size(in_w, in_h), 0, 0, cv::INTER_CUBIC);
  const float *out_data =
      inferGray(p_bmrt_, handle_, input, in_shape, graph_name.c_str(), out_buf_);
  if (out_data == nullptr)
    return {};

  return parseOutput(out_data, img_w, img_h);
}

// Device-input forward: resize the (already grayscale) device image on-device,
// strip the VPP row-stride padding D2D into a tight contiguous input, then run
// the fused uint8 model with no host round-trip. Equivalent to the host forward
// (cv::resize INTER_CUBIC + inferGray), except VPP LINEAR replaces CUBIC
// (BM1684X/CV186X vpp exposes no BICUBIC), a sub-pixel corner delta.
std::vector<cv::Mat> SSDDetector::forward(bm_handle_t handle,
                                          const bm_image &gray,
                                          int target_width, int target_height) {
  const int img_w = gray.width;
  const int img_h = gray.height;

  const DetectGraph *g = selectGraph(target_width, target_height);
  const int in_w = (g != nullptr) ? g->w : kDetectInputSize;
  const int in_h = (g != nullptr) ? g->h : kDetectInputSize;
  const std::string graph_name = (g != nullptr) ? g->name : fallback_.name;
  const bm_shape_t in_shape =
      (g != nullptr) ? g->input_shape : fallback_.input_shape;

  // 1) proportional resize to the selected shape's grayscale on-device. VPP
  //    keeps a 64B-aligned row stride (padded), stripped in step 2.
  bm_image resized;
  if (bm_image_create(handle, in_h, in_w, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE,
                      &resized) != BM_SUCCESS)
    return {};
  if (bm_image_alloc_dev_mem(resized) != BM_SUCCESS) {
    detail::wqDestroyImage(resized);
    return {};
  }
  if (bmcv_image_vpp_convert(handle, 1, gray, &resized, nullptr,
                             BMCV_INTER_LINEAR) != BM_SUCCESS) {
    detail::wqDestroyImage(resized);
    return {};
  }

  // 2) tight contiguous input (stride == width, 8UC1) -- the same layout the
  //    host inferGray builds; bm_inference consumes its device mem directly.
  int strides[4] = {in_w, in_w, in_w, in_w};
  bm_image input;
  if (bm_image_create(handle, in_h, in_w, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE,
                      &input, strides) != BM_SUCCESS) {
    detail::wqDestroyImage(resized);
    return {};
  }
  if (bm_image_alloc_contiguous_mem(1, &input) != BM_SUCCESS) {
    detail::wqDestroyImage(input);
    detail::wqDestroyImage(resized);
    return {};
  }

  // 3) strip stride padding D2D (resized -> input), then infer with no host
  //    round-trip
  bm_device_mem_t src_mem[3] = {};
  bm_device_mem_t dst_mem[3] = {};
  const float *out_data = nullptr;
  if (bm_image_get_device_mem(resized, src_mem) == BM_SUCCESS &&
      bm_image_get_device_mem(input, dst_mem) == BM_SUCCESS) {
    const size_t src_stride = src_mem[0].size / static_cast<size_t>(in_h);
    if (detail::wqD2dStrip(handle, dst_mem[0], src_mem[0], in_w, in_h,
                           src_stride) == BM_SUCCESS &&
        bm_inference(p_bmrt_, &input, out_buf_.data(), in_shape,
                     graph_name.c_str())) {
      out_data = reinterpret_cast<const float *>(out_buf_.data());
    }
  }

  bm_image_free_contiguous_mem(1, &input);
  detail::wqDestroyImage(input);
  detail::wqDestroyImage(resized);
  return out_data ? parseOutput(out_data, img_w, img_h)
                  : std::vector<cv::Mat>{};
}

std::vector<cv::Mat> SSDDetector::parseOutput(const float *out_data, int img_w,
                                              int img_h) const {
  std::vector<cv::Mat> point_list;
  for (int row = 0; row < kDetectCandidateNum; ++row) {
    const float *prob = out_data + row * 7;
    // prob[0] unused; prob[1]==1 means QRCode; prob[2] is confidence (safety
    // threshold, see opencv#2877)
    if (prob[1] == 1.0f && prob[2] > 1e-5f) {
      cv::Mat point(4, 2, CV_32FC1);
      const float x0 = std::max(0.0f, std::min(prob[3] * img_w, img_w - 1.0f));
      const float y0 = std::max(0.0f, std::min(prob[4] * img_h, img_h - 1.0f));
      const float x1 = std::max(0.0f, std::min(prob[5] * img_w, img_w - 1.0f));
      const float y1 = std::max(0.0f, std::min(prob[6] * img_h, img_h - 1.0f));

      point.at<float>(0, 0) = x0;
      point.at<float>(0, 1) = y0;
      point.at<float>(1, 0) = x1;
      point.at<float>(1, 1) = y0;
      point.at<float>(2, 0) = x1;
      point.at<float>(2, 1) = y1;
      point.at<float>(3, 0) = x0;
      point.at<float>(3, 1) = y1;
      point_list.push_back(point);
    }
  }
  return point_list;
}

} // namespace wechat_qrcode