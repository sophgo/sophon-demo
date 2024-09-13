//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef DIRECTMHP_H
#define DIRECTMHP_H

#include <iostream>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

const float PI = 3.14159265;

struct DirectMHPBox {
  float x, y, width, height;
  float pitch, yaw, roll;
  float score;
  int class_id;
};

using DirectMHPBoxVec = std::vector<DirectMHPBox>;

class DirectMHP {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  //configuration
  float m_confThreshold= 0.5;
  float m_nmsThreshold = 0.5;


  std::vector<std::string> m_class_names;
  int m_class_num = 80; // default is coco names
  int m_net_h, m_net_w;
  int max_batch;
  int output_num;
  int min_dim;
  bmcv_convert_to_attr converto_attr;

  TimeStamp *m_ts;

  private:
  int pre_process(const std::vector<bm_image>& images);
  int post_process(const std::vector<bm_image>& images, std::vector<DirectMHPBoxVec>& boxes);
  int post_process_cpu_opt(const std::vector<bm_image> &images, std::vector<DirectMHPBoxVec>& detected_boxes);
  int argmax(float* data, int dsize);
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
  static float sigmoid(float x);
  void NMS(DirectMHPBoxVec &dets, float nmsConfidence);

  public:
  DirectMHP(std::shared_ptr<BMNNContext> context);
  virtual ~DirectMHP();
  int Init(float confThresh=0.5, float nmsThresh=0.5);
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Detect(const std::vector<bm_image>& images, std::vector<DirectMHPBoxVec>& boxes);
  
  void draw_bmcv(bm_handle_t &handle, int classId, float conf, float left, float top, float width, float height, float pitch, float roll, float yaw,bm_image& frame, bool put_text_flag);
};

#endif //!DIRECTMHP_H
