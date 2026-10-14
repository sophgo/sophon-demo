//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef RAM_HPP
#define RAM_HPP

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "bm_wrapper.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1

class RAM {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  // model info
  int m_net_h;
  int m_net_w;
  int max_batch;
  int output_num;
  int class_num;
  bmcv_convert_to_attr converto_attr;

  // tag list and per-class thresholds
  std::vector<std::string> tag_list;
  std::vector<std::string> tag_list_chinese;
  std::vector<float> class_threshold;

  // for profiling
  TimeStamp *ts_ = NULL;

 private:
  int pre_process(std::vector<bm_image> &images);
  int post_process(std::vector<bm_image> &images,
                   std::vector<std::pair<std::string, std::string>> &results);

 public:
  RAM(std::shared_ptr<BMNNContext> context);
  virtual ~RAM();
  int Init(const std::string &tag_list_file,
           const std::string &tag_list_chinese_file,
           const std::string &tag_list_threshold_file);
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Tag(std::vector<bm_image> &input_images,
          std::vector<std::pair<std::string, std::string>> &results);
};

#endif /* RAM_HPP */
