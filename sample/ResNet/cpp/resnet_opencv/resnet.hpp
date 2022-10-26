//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef RESNET_HPP
#define RESNET_HPP

#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "bmruntime_interface.h"
#include "utils.hpp"

class RESNET
{
public:
  RESNET(const std::string bmodel, int dev_id);
  ~RESNET();
  void preForward(const std::vector<cv::Mat> &images);
  void forward();
  void postForward(std::vector<std::pair<int, float>> &results);
  void enableProfile(TimeStamp *ts);
  int batch_size();

private:
  void setStdMean(std::vector<float> &std, std::vector<float> &mean);
  void wrapInputLayer(std::vector<cv::Mat> *input_channels, const int batch_id);
  void preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels);

  // handle of low level device
  bm_handle_t bm_handle_;
  int dev_id_;

  // runtime helper
  const char **net_names_;
  void *p_bmrt_;

  // network input shape
  int batch_size_;
  int num_channels_;
  int class_num_;
  int net_h_;
  int net_w_;

  // network related parameters
  cv::Mat mean_;
  cv::Mat std_;

  // input & output buffers
  bm_tensor_t input_tensor_;
  bm_tensor_t output_tensor_;
  float input_scale_;
  float output_scale_;
  float *input_f32;
  int8_t *input_int8;
  float *output_f32;
  int8_t *output_int8;
  bool int8_flag_;
  bool int8_output_flag;
  int count_per_img;
  // for profiling
  TimeStamp *ts_;
};

#endif /* RESNET_HPP */
