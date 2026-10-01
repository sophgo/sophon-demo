//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// WeChatQRCode detect-model inference wrapper (pure bmrt + bmcv, no sail).
//
// Mirrors OpenCV wechat_qrcode's SSDDetector::forward: detect is a
// fully-convolutional SSD-MobileNet whose DetectionOutput layer is baked into
// the bmodel, so here we only parse the (1,1,100,7) F32 result.
//
// Multi-shape (aligned with OpenCV's ~160k-area proportional rescale):
//   OpenCV picks s = min(1, sqrt(160000/(w*h))) and feeds the detect net an
//   aspect-preserving (h*s, w*s) input. tpu-mlir's caffe frontend cannot do
//   dynamic shape, so we compile once per static shape ("shape table") and merge
//   all sub-networks into one bmodel via model_tool --combine. At runtime we
//   pick the closest sub-network by the target aspect ratio, proportionally
//   resize the gray image to that shape (INTER_CUBIC, matching OpenCV) and infer.
//   The DetectionOutput coordinates are normalized, then scaled back by the
//   original width/height, independent of the selected shape.

#ifndef __WECHAT_QRCODE_SSD_DETECTOR_HPP_
#define __WECHAT_QRCODE_SSD_DETECTOR_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "bm_wrapper.hpp"

namespace wechat_qrcode {

// Fallback fixed detect input size (used only for legacy single-shape bmodels)
constexpr int kDetectInputSize = 384;
// detection_output (1,1,100,7): 100 candidate boxes, 7 values each
// [img_id, label, conf, x0, y0, x1, y1], coordinates normalized to [0,1]
constexpr int kDetectCandidateNum = 100;

// One static-shape detect sub-network
struct DetectGraph {
  std::string name; // network name, like detect_<W>_<H>
  int w = 0;        // input image width
  int h = 0;        // input image height
  double aspect = 1.0; // (double)w / h
  bm_shape_t input_shape{}; // model input shape (1,1,H,W), from net info
};

class SSDDetector {
public:
  SSDDetector() = default;
  ~SSDDetector();

  // Load detect.bmodel (a multi-network bmodel) and record the shape table.
  // Returns 0 on success, non-zero on failure.
  int init(bm_handle_t handle, const std::string &bmodel_path);

  // Input grayscale image (8UC1); output the detected QRCode 4-corner point
  // sets (original-image coordinates, each pooint 4x2 CV_32FC1).
  // target_width/height is the proportional-rescale target size (img w/h * s),
  // used to pick the closest static-shape sub-network by aspect ratio.
  std::vector<cv::Mat> forward(const cv::Mat &img, int target_width,
                               int target_height);

  // Device-input forward: gray is a device FORMAT_GRAY bm_image (uint8, e.g.
  // the csc'd view of a device-decoded frame). Resizes on-device (VPP, LINEAR)
  // and strips the stride padding D2D into a tight input buffer, so the fused
  // uint8 model runs with no host->device copy. Returns the same 4-corner sets
  // as the host forward (coordinates scaled back by the original w/h).
  std::vector<cv::Mat> forward(bm_handle_t handle, const bm_image &gray,
                               int target_width, int target_height);

private:
  // Pick the closest sub-network by target aspect ratio (graphs_[0] fallback)
  const DetectGraph *selectGraph(int target_width, int target_height) const;

  // Parse the (1,1,100,7) F32 detection output; coordinates are normalized and
  // scaled back by the original width/height.
  std::vector<cv::Mat> parseOutput(const float *out_data, int img_w,
                                   int img_h) const;

  bm_handle_t handle_ = nullptr;
  void *p_bmrt_ = nullptr; // bmrt context holding the detect bmodels
  std::vector<DetectGraph> graphs_; // multi-shape sub-networks (may be empty)
  DetectGraph fallback_;            // fixed-shape fallback (legacy bmodels)
  std::vector<uint8_t> out_buf_;    // reusable host output buffer (F32)
};

} // namespace wechat_qrcode

#endif // __WECHAT_QRCODE_SSD_DETECTOR_HPP_