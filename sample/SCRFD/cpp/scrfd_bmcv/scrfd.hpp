//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SCRFD_H
#define SCRFD_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"

#include <dirent.h>

#include <algorithm>
#include <cstring>

// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

struct ScrfdBox {
  int x, y, width, height;
  float score;
};

struct cvai_bbox_t {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
};

struct cvai_pts_t {
  float* x;
  float* y;
  int size;
};

struct cvai_face_info_t {
  cvai_bbox_t bbox;
  cvai_pts_t pts;
};

struct cvai_face_t {
  int size;
  int width;
  int height;
  cvai_face_info_t* info;
};

struct anchor_box {
  float x1;
  float y1;
  float x2;
  float y2;
};

struct anchor_cfg {
 public:
  int STRIDE;
  std::vector<int> SCALES;
  int BASE_SIZE;
  std::vector<float> RATIOS;
  int ALLOWED_BORDER;

  anchor_cfg() {
    STRIDE = 0;
    SCALES.clear();
    BASE_SIZE = 0;
    RATIOS.clear();
    ALLOWED_BORDER = 0;
  }
};

using ScrfdBoxVec = std::vector<cvai_face_info_t>;

class Scrfd {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  // configuration
  float m_confThreshold = 0.5;
  float m_nmsThreshold = 0.5;

  int m_net_h, m_net_w;
  int max_batch;
  int min_dim;
  int output_num;
  bmcv_convert_to_attr converto_attr;

  TimeStamp* m_ts;

  std::vector<float> rescale_params;
  std::vector<cvai_face_info_t> res;

 private:
  int pre_process(const std::vector<bm_image>& input);
  int post_process(const std::vector<bm_image>& images,
                   std::vector<ScrfdBoxVec>& detected_boxes);
  std::vector<std::vector<float>> generate_mmdet_base_anchors(
      float base_size, float center_offset, const std::vector<float>& ratios,
      const std::vector<int>& scales);
  std::vector<std::vector<float>> generate_mmdet_grid_anchors(
      int feat_w, int feat_h, int stride,
      std::vector<std::vector<float>>& base_anchors);
  template <typename T>
  void NonMaximumSuppression(std::vector<T>& bboxes, std::vector<T>& bboxes_nms,
                             const float threshold, const char method);
  void clip_boxes(int width, int height, cvai_bbox_t& box);
  std::vector<cvai_face_info_t> outputParser(
      int frame_width, int frame_height,
      const std::vector<float>& rescale_param, int batch_idx);
  bool compute_pad_resize_param(int src_height, int src_width, int dst_height,
                                int dst_width,
                                std::vector<float>& rescale_params);
  float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h,
                                bool* pIsAligWidth);

 public:
  Scrfd(std::shared_ptr<BMNNContext> context);
  virtual ~Scrfd();
  int Init(float confThresh = 0.5, float nmsThresh = 0.5);
  void enableProfile(TimeStamp* ts);
  int batch_size();
  int Detect(const std::vector<bm_image>& images,
             std::vector<ScrfdBoxVec>& boxes);
  void drawPred(std::vector<cvai_face_info_t>& res, cv::Mat& frame);
  void draw_bmcv(bm_handle_t& handle, cvai_pts_t five_point, float conf,
                 int left, int top, int width, int height, bm_image& frame,
                 bool put_text_flag, bool draw_point_flag);
  void readDirectory(const std::string& directory,
                     std::vector<std::string>& files_vector, bool recursive);
};

#endif  //! SCRFD_H