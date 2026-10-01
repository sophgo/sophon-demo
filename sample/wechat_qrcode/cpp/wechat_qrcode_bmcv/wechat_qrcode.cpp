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
// This is the orchestrator: it detects candidate QR regions with the detect
// model (TPU), crops each with padding, scales (sr model on TPU for small
// crops), decodes with zxing (CPU), and remaps the corner points back to the
// original image. Logic is a straight port of the host path of the
// sail::wechat_qrcode implementation (sophon-sail/src/wechat_qrcode), rewritten
// to feed the models via raw bmrt instead of sail::Engine.

#include "wechat_qrcode.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "bm_util.hpp"
#include "decodermgr.hpp"
#include "detector/ssd_detector.hpp"
#include "scale/super_scale.hpp"

namespace wechat_qrcode {

namespace {

// Internal result representation. The public API returns vector<string> +
// vector<cv::Mat>; QRResult is only an internal scratch type.
struct QRResult {
  std::string text;
  std::vector<std::array<float, 2>> corners; // 4 corners, original-image coords
};

// Axis-aligned crop box with padding. Matches OpenCV Align::crop
// (rotate90_=false; this pipeline never transposes): take the 4-point bounding
// box, add padding (0.1 ratio, at least min_padding), and clamp to the image.
struct CropBox {
  int x = 0, y = 0, w = 0, h = 0;
};

// Compute the axis-aligned crop box around a detected QR code.
// @param point      4x2 CV_32F corners, in original-image coordinates.
// @param img_w      source image width.
// @param img_h      source image height.
// @param padding_w  extra padding ratio on the box width (e.g. 0.1).
// @param padding_h  extra padding ratio on the box height.
// @param min_padding  minimum padding in pixels (floor).
// @return the expanded box, clamped to [0, img_w-1] x [0, img_h-1].
CropBox computeCropBox(const cv::Mat &point, int img_w, int img_h,
                       float padding_w, float padding_h, int min_padding) {
  const int x0 = static_cast<int>(point.at<float>(0, 0));
  const int y0 = static_cast<int>(point.at<float>(0, 1));
  const int x2 = static_cast<int>(point.at<float>(2, 0));
  const int y2 = static_cast<int>(point.at<float>(2, 1));
  const int width = x2 - x0 + 1;
  const int height = y2 - y0 + 1;
  const int padx = static_cast<int>(std::max(
      padding_w * static_cast<float>(width), static_cast<float>(min_padding)));
  const int pady = static_cast<int>(std::max(
      padding_h * static_cast<float>(height), static_cast<float>(min_padding)));

  CropBox box;
  box.x = std::max(x0 - padx, 0);
  box.y = std::max(y0 - pady, 0);
  const int end_x = std::min(x2 + padx, img_w - 1);
  const int end_y = std::min(y2 + pady, img_h - 1);
  box.w = end_x - box.x + 1;
  box.h = end_y - box.y + 1;
  return box;
}

// Internal results -> OpenCV-style (texts, points). Each point is a 4x2
// CV_32FC1, matching opencv_contrib.
std::vector<std::string> emit(const std::vector<QRResult> &results,
                              std::vector<cv::Mat> *points) {
  std::vector<std::string> texts;
  texts.reserve(results.size());
  if (points) {
    points->clear();
    points->reserve(results.size());
  }
  for (const auto &r : results) {
    texts.push_back(r.text);
    if (points) {
      cv::Mat m(4, 2, CV_32FC1);
      for (int i = 0; i < 4; ++i) {
        m.at<float>(i, 0) = r.corners[i][0];
        m.at<float>(i, 1) = r.corners[i][1];
      }
      points->push_back(m);
    }
  }
  return texts;
}

bool fileExists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

// Copy plane0 of a FORMAT_GRAY device image back to host as an 8UC1 Mat,
// stripping the row-stride alignment padding row by row. This is the only
// D->H copy in the device path, and it is over a QR crop rather than the frame.
cv::Mat grayPlaneToMat(bm_handle_t handle, const bm_image &img) {
  bm_device_mem_t mem[3] = {};
  if (bm_image_get_device_mem(img, mem) != BM_SUCCESS)
    return {};

  const int w = img.width, h = img.height;
  const size_t plane_bytes = static_cast<size_t>(mem[0].size);
  if (plane_bytes < static_cast<size_t>(w) * h)
    return {};
  const int stride = static_cast<int>(plane_bytes / static_cast<size_t>(h));

  std::vector<uint8_t> buf(plane_bytes);
  if (bm_memcpy_d2s(handle, buf.data(), mem[0]) != BM_SUCCESS)
    return {};

  cv::Mat mat(h, w, CV_8UC1);
  for (int r = 0; r < h; ++r) {
    std::memcpy(mat.ptr(r), buf.data() + static_cast<size_t>(r) * stride, w);
  }
  return mat;
}

// Build a device FORMAT_GRAY view for the detect stage (no full-frame D->H):
//   FORMAT_GRAY          -> shallow copy sharing image_private (read-only for
//                           vpp);
//   BGR/RGB packed/planar-> vpp csc to grayscale (allocates; caller destroys).
// On success sets *gray_view and *need_destroy (whether the caller must
// detail::wqDestroyImage the view afterwards).
bool deviceGrayView(bm_handle_t handle, const bm_image &img,
                    bm_image *gray_view, bool *need_destroy) {
  *need_destroy = false;
  switch (img.image_format) {
  case FORMAT_GRAY:
    *gray_view = img; // already grayscale
    return true;
  case FORMAT_BGR_PACKED:
  case FORMAT_RGB_PACKED:
  case FORMAT_BGR_PLANAR:
  case FORMAT_RGB_PLANAR: {
    if (bm_image_create(handle, img.height, img.width, FORMAT_GRAY,
                        DATA_TYPE_EXT_1N_BYTE, gray_view) != BM_SUCCESS)
      return false;
    if (bm_image_alloc_dev_mem(*gray_view) != BM_SUCCESS) {
      detail::wqDestroyImage(*gray_view);
      return false;
    }
    // BGR/RGB -> grayscale csc; vpp exposes no BICUBIC, so LINEAR is used.
    if (bmcv_image_vpp_convert(handle, 1, img, gray_view, nullptr,
                               BMCV_INTER_LINEAR) != BM_SUCCESS) {
      detail::wqDestroyImage(*gray_view);
      return false;
    }
    *need_destroy = true;
    return true;
  }
  default:
    return false; // other formats unsupported on the imread device path
  }
}

} // namespace

class WeChatQRCode::Impl {
public:
  Impl(bm_handle_t handle, const std::string &detect_bmodel,
       const std::string &sr_bmodel)
      : handle_(handle) {
    // Missing model -> readable error (avoid low-level bmrt failures)
    if (!fileExists(detect_bmodel))
      throw std::runtime_error("model not found: \"" + detect_bmodel + "\"");
    if (!fileExists(sr_bmodel))
      throw std::runtime_error("model not found: \"" + sr_bmodel + "\"");

    detector_ = std::make_unique<SSDDetector>();
    if (detector_->init(handle, detect_bmodel) != 0)
      throw std::runtime_error("failed to load detect bmodel: \"" +
                               detect_bmodel + "\"");
    super_resolution_model_ = std::make_unique<SuperScale>();
    if (super_resolution_model_->init(handle, sr_bmodel) != 0)
      throw std::runtime_error("failed to load sr bmodel: \"" + sr_bmodel +
                               "\"");
  }

  std::vector<cv::Mat> detect(const cv::Mat &img);
  std::vector<QRResult> decode(const cv::Mat &img,
                               std::vector<cv::Mat> &candidate_points);
  // Device-path decode: crop each candidate block on-device via VPP (1:1), copy
  // only that crop back to host, then run the shared sr/zxing decodeCrop.
  std::vector<QRResult> decodeDevice(bm_handle_t handle, const bm_image &gray,
                                     std::vector<cv::Mat> &candidate_points);
  int applyDetector(const cv::Mat &img, std::vector<cv::Mat> &points);
  std::vector<float> getScaleList(int width, int height);

  std::unique_ptr<SSDDetector> detector_;
  std::unique_ptr<SuperScale> super_resolution_model_;
  BenchStats stats_;
  bm_handle_t handle_;

private:
  // sr + zxing + coordinate remapping (offset crop_x/crop_y) + dedup for one crop
  void decodeCrop(const cv::Mat &cropped_img, int crop_x, int crop_y,
                  std::vector<QRResult> &decode_results);
};

WeChatQRCode::WeChatQRCode(bm_handle_t handle, const std::string &detect_bmodel,
                           const std::string &sr_bmodel)
    : p_(new Impl(handle, detect_bmodel, sr_bmodel)) {}

WeChatQRCode::~WeChatQRCode() = default;

std::vector<std::string>
WeChatQRCode::detectAndDecode(const std::string &image_path,
                              std::vector<cv::Mat> *points) {
  // Read the stored pixel frame: IGNORE_ORIENTATION keeps raw EXIF pixels
  // (BoofCV GT is defined in the stored frame; auto-rotating EXIF would skew
  // corners on portrait JPEGs and diverge across OpenCV versions).
  cv::Mat img =
      cv::imread(image_path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
  if (img.empty()) {
    std::cerr << "[wechat_qrcode] failed to read image \"" << image_path
              << "\": unsupported format or file not accessible" << std::endl;
    if (points)
      points->clear();
    return {};
  }
  return detectAndDecode(img, points);
}

// Detect QR codes in an image and decode each of them.
// @param img     BGR/GRAY/RGBA 8-bit input; empty or <=20px returns empty.
// @param points  optional out: one 4x2 CV_32F corner matrix per decoded code
//                (original-image coordinates); cleared on failure / no codes.
// @return decoded texts, one entry per successfully decoded code.
std::vector<std::string>
WeChatQRCode::detectAndDecode(const cv::Mat &img,
                              std::vector<cv::Mat> *points) {
  if (img.empty() || img.cols <= 20 || img.rows <= 20) {
    if (points)
      points->clear();
    return {}; // too little image data for a reliable result
  }
  if (img.depth() != CV_8U) {
    if (points)
      points->clear();
    return {}; // only uint8 (8-bit) input is supported
  }

  cv::Mat input_img;
  const int incn = img.channels();
  if (incn == 3 || incn == 4) {
    cv::cvtColor(img, input_img, cv::COLOR_BGR2GRAY);
  } else {
    input_img = img;
  }

  auto candidate_points = p_->detect(input_img);
  auto results = p_->decode(input_img, candidate_points);
  return emit(results, points);
}

// Device-input detectAndDecode (bm_image stays in device memory): build a
// device grayscale view, run detect on-device, then device-crop each QR block
// and copy only it back to host for zxing -- a straight port of the sail
// detectAndDecode(bm_handle_t, bm_image) device pass-through.
// @param handle  bm_handle_t that produced img (same device as this instance).
// @param img     device-memory uint8 BGR/RGB packed/planar or GRAY.
// @param points  optional out (same contract as the cv::Mat overload).
// @return decoded texts, one entry per successfully decoded code.
std::vector<std::string>
WeChatQRCode::detectAndDecode(bm_handle_t handle, const bm_image &img,
                              std::vector<cv::Mat> *points) {
  if (handle == nullptr || bm_get_devid(handle) != bm_get_devid(p_->handle_)) {
    if (points)
      points->clear();
    return {}; // img must live on the same device as the loaded models
  }
  if (img.width <= 20 || img.height <= 20) {
    if (points)
      points->clear();
    return {}; // too little image data (aligned with the host path)
  }
  if (img.data_type != DATA_TYPE_EXT_1N_BYTE) {
    if (points)
      points->clear();
    return {}; // device pass-through supports uint8 only
  }

  bm_image gray_view;
  bool need_destroy_gray = false;
  if (!deviceGrayView(handle, img, &gray_view, &need_destroy_gray)) {
    if (points)
      points->clear();
    return {};
  }

  // keep-aspect target size (same 400x400 rule as applyDetector) to pick the
  // closest static-shape sub-network
  const int img_w = img.width, img_h = img.height;
  const float targetArea = 400.f * 400.f;
  const float tmpScaleFactor =
      std::min(1.f, std::sqrt(targetArea / static_cast<float>(img_w * img_h)));
  const int detect_width = static_cast<int>(img_w * tmpScaleFactor);
  const int detect_height = static_cast<int>(img_h * tmpScaleFactor);

  const auto t0 = std::chrono::high_resolution_clock::now();
  auto candidate_points =
      p_->detector_->forward(handle, gray_view, detect_width, detect_height);
  const auto t1 = std::chrono::high_resolution_clock::now();
  p_->stats_.detect_ms +=
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  p_->stats_.detect_calls++;

  std::vector<QRResult> results;
  if (!candidate_points.empty())
    results = p_->decodeDevice(handle, gray_view, candidate_points);

  if (need_destroy_gray)
    detail::wqDestroyImage(gray_view);
  return emit(results, points);
}

// Run the TPU detect model and return candidate QR corner boxes.
// @param img  single-channel 8-bit gray input.
// @return one 4x2 CV_32F matrix per detected candidate, original-image coords.
std::vector<cv::Mat> WeChatQRCode::Impl::detect(const cv::Mat &img) {
  std::vector<cv::Mat> points;
  applyDetector(img, points);
  return points;
}

int WeChatQRCode::Impl::applyDetector(const cv::Mat &img,
                                      std::vector<cv::Mat> &points) {
  const int img_w = img.cols;
  const int img_h = img.rows;

  // Keep-aspect target size using OpenCV's target area (400x400); the forward
  // pass picks the closest static-shape sub-network by this size's aspect ratio.
  const float targetArea = 400.f * 400.f;
  const float tmpScaleFactor =
      std::min(1.f, std::sqrt(targetArea / static_cast<float>(img_w * img_h)));
  const int detect_width = static_cast<int>(img_w * tmpScaleFactor);
  const int detect_height = static_cast<int>(img_h * tmpScaleFactor);

  const auto t0 = std::chrono::high_resolution_clock::now();
  points = detector_->forward(img, detect_width, detect_height);
  const auto t1 = std::chrono::high_resolution_clock::now();
  stats_.detect_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  stats_.detect_calls++;
  return 0;
}

// heuristic (same as OpenCV)
std::vector<float> WeChatQRCode::Impl::getScaleList(int width, int height) {
  if (width < 320 || height < 320)
    return {1.0, 2.0, 0.5};
  if (width < 640 && height < 640)
    return {1.0, 0.5};
  return {0.5, 1.0};
}

std::vector<QRResult>
WeChatQRCode::Impl::decode(const cv::Mat &img,
                           std::vector<cv::Mat> &candidate_points) {
  std::vector<QRResult> decode_results;
  if (candidate_points.empty()) {
    return decode_results;
  }

  const float padding_w = 0.1f, padding_h = 0.1f;
  const int min_padding = 15;
  for (auto &point : candidate_points) {
    CropBox box =
        computeCropBox(point, img.cols, img.rows, padding_w, padding_h, min_padding);
    cv::Mat cropped_img = img(cv::Rect(box.x, box.y, box.w, box.h)).clone();
    decodeCrop(cropped_img, box.x, box.y, decode_results);
  }

  return decode_results;
}

// Device-path decode: crop each candidate block on-device via VPP (1:1, no
// scaling), strip the stride padding back to host, then run the shared
// sr/zxing decodeCrop (same as the host decode).
std::vector<QRResult>
WeChatQRCode::Impl::decodeDevice(bm_handle_t handle, const bm_image &gray,
                                 std::vector<cv::Mat> &candidate_points) {
  std::vector<QRResult> decode_results;
  const int img_w = gray.width, img_h = gray.height;
  const float padding_w = 0.1f, padding_h = 0.1f;
  const int min_padding = 15;
  for (auto &point : candidate_points) {
    CropBox box =
        computeCropBox(point, img_w, img_h, padding_w, padding_h, min_padding);

    // on-device crop (vpp crop, 1:1); the crop still carries stride padding,
    // stripped row by row by grayPlaneToMat right after
    bm_image crop_dev;
    if (bm_image_create(handle, box.h, box.w, FORMAT_GRAY,
                        DATA_TYPE_EXT_1N_BYTE, &crop_dev) != BM_SUCCESS)
      continue;
    if (bm_image_alloc_dev_mem(crop_dev) != BM_SUCCESS) {
      detail::wqDestroyImage(crop_dev);
      continue;
    }
    bmcv_rect_t roi; // 1684X SDK uses int fields, 1688 unsigned int; assign per
                     // field to avoid signedness ambiguity
    roi.start_x = box.x;
    roi.start_y = box.y;
    roi.crop_w = box.w;
    roi.crop_h = box.h;
    if (bmcv_image_vpp_convert(handle, 1, gray, &crop_dev, &roi,
                               BMCV_INTER_NEAREST) != BM_SUCCESS) {
      detail::wqDestroyImage(crop_dev);
      continue;
    }
    cv::Mat cropped_img = grayPlaneToMat(handle, crop_dev);
    detail::wqDestroyImage(crop_dev);
    if (cropped_img.empty())
      continue;
    decodeCrop(cropped_img, box.x, box.y, decode_results);
  }

  return decode_results;
}

// Decode one cropped QR candidate: super-resolve (sr or cubic/area) over a few
// numeric scales, run the CPU zxing decoder, remap corners back to the original
// image and drop near-identical duplicates (eps=10px, same as OpenCV).
// @param cropped_img    padded crop of one detected candidate (gray 8-bit).
// @param crop_x/crop_y  top-left offset of the crop in the original image.
// @param decode_results out: appended with each newly decoded QRResult.
void WeChatQRCode::Impl::decodeCrop(const cv::Mat &cropped_img, int crop_x,
                                    int crop_y,
                                    std::vector<QRResult> &decode_results) {
  auto scale_list = getScaleList(cropped_img.cols, cropped_img.rows);
  for (auto cur_scale : scale_list) {
    // sr_ms covers the whole processImageScale stage (sr.bmodel or cubic/area)
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat scaled_img =
        super_resolution_model_->processImageScale(cropped_img, cur_scale,
                                                   true /*use_sr*/);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.sr_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.sr_calls++;

    DecoderMgr decodemgr;
    std::vector<std::string> texts;
    std::vector<std::vector<cv::Point2f>> zxing_points;
    t0 = std::chrono::high_resolution_clock::now();
    const int ret = decodemgr.decodeImage(scaled_img, true /*use_nn_detector*/,
                                          texts, zxing_points);
    t1 = std::chrono::high_resolution_clock::now();
    stats_.zxing_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.zxing_calls++;
    if (ret != 0) {
      continue; // this scale failed to decode, try the next one
    }

    for (size_t i = 0; i < zxing_points.size(); ++i) {
      std::vector<cv::Point2f> points_qr = zxing_points[i];
      for (auto &pt : points_qr) {
        pt /= cur_scale; // divide coordinates back to the crop scale
        pt.x += crop_x;  // remap to the original image
        pt.y += crop_y;
      }

      QRResult result;
      result.text = texts[i];
      for (int j = 0; j < 4; ++j) {
        result.corners.push_back({points_qr[j].x, points_qr[j].y});
      }

      // dedup by the four corner coordinates (same as OpenCV, eps=10px)
      const float eps = 10.f;
      bool is_duplicate = false;
      for (const auto &tmp : decode_results) {
        bool same = true;
        for (size_t j = 0; j < 4; ++j) {
          if (std::abs(tmp.corners[j][0] - points_qr[j].x) >= eps ||
              std::abs(tmp.corners[j][1] - points_qr[j].y) >= eps) {
            same = false;
            break;
          }
        }
        if (same) {
          is_duplicate = true;
          break;
        }
      }
      if (!is_duplicate) {
        decode_results.push_back(result);
      }
    }
    break; // decoded for this candidate, stop trying other scales
  }
}

void WeChatQRCode::resetBenchStats() { p_->stats_ = BenchStats(); }

BenchStats WeChatQRCode::getBenchStats() const { return p_->stats_; }

} // namespace wechat_qrcode