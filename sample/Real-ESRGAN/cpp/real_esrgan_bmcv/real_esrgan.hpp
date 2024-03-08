//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef F9766C78_A85E_4E73_9C8B_94C41D63180B
#define F9766C78_A85E_4E73_9C8B_94C41D63180B

#ifndef Real_ESRGAN_H
#define Real_ESRGAN_H

#include <fstream>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0


class Real_ESRGAN {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  //configuration

  bool use_cpu_opt;


  int m_net_h, m_net_w;
  int max_batch;
  int output_num;
  int min_dim;
  int upsample_scale;
  bmcv_convert_to_attr converto_attr;

  TimeStamp *m_ts;

  private:
  int pre_process(const std::vector<bm_image>& images);
  int post_process(const std::vector<bm_image>& images,std::vector<cv::Mat>& output_mats);
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
 
  public:
  Real_ESRGAN(std::shared_ptr<BMNNContext> context);
  virtual ~Real_ESRGAN();
  int Init();
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Detect(const std::vector<bm_image>& images,std::vector<cv::Mat>& output_mats);
  
};

#endif //!Real_ESRGAN_H


#endif /* F9766C78_A85E_4E73_9C8B_94C41D63180B */
