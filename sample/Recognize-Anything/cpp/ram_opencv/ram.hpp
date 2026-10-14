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

class RAM {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::shared_ptr<BMNNTensor>  m_input_tensor;
  std::shared_ptr<BMNNTensor>  m_output_tensor;

  // model info
  int m_net_h;
  int m_net_w;
  int m_num_channels;
  int m_dev_id;
  int max_batch;
  int output_num;
  int class_num;
  float input_scale;
  float output_scale;
  bm_tensor_t input_tensor;

  // tag list and per-class thresholds
  std::vector<std::string> tag_list;
  std::vector<std::string> tag_list_chinese;
  std::vector<float> class_threshold;

  // for profiling
  TimeStamp *ts_ = NULL;

 private:
  float *m_input_float;
  int8_t *m_input_int8;
  int m_input_count;
  cv::Mat m_mean;
  cv::Mat m_std;

  int pre_process(std::vector<cv::Mat> &images);
  int post_process(std::vector<cv::Mat> &images,
                   std::vector<std::pair<std::string, std::string>> &results);
  void pre_process_image(const cv::Mat& img, std::vector<cv::Mat> *input_channels);
  void setStdMean(std::vector<float> &std, std::vector<float> &mean);
  void wrapInputLayer(std::vector<cv::Mat>* input_channels, int batch_id);

 public:
  RAM(std::shared_ptr<BMNNContext> context, int dev_id);
  virtual ~RAM();
  int Init(const std::string &tag_list_file,
           const std::string &tag_list_chinese_file,
           const std::string &tag_list_threshold_file);
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Tag(std::vector<cv::Mat>& input_images,
          std::vector<std::pair<std::string, std::string>> &results);
};

#endif /* RAM_HPP */
