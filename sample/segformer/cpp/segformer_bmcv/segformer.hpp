//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SEGFORMER_H
#define SEGFORMER_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
#include <string>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

class SegFormer
{
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  int m_net_h, m_net_w;
  int max_batch;
  int output_num;
  int min_dim;

  bmcv_convert_to_attr converto_attr;

  TimeStamp *m_ts;

private:
  int pre_process(const std::vector<bm_image> &images);
  int post_process(std::vector<bm_image> &images,std::vector<std::vector<std::vector<int32_t>>> &result_datas);
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
  static float sigmoid(float x);


public:
  SegFormer(std::shared_ptr<BMNNContext> context);
  virtual ~SegFormer();
  int Init();
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Detect(std::vector<bm_image> &images,std::vector<std::vector<std::vector<int32_t>>> &result_datas);

  
  std::string palette;

};

#endif
