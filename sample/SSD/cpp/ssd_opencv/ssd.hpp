//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef SSD_HPP
#define SSD_HPP

#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "bmruntime_interface.h"
#include "utils.hpp"
struct ObjRect {
  unsigned int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
};

class SSD {
public:
  SSD(const std::string bmodel, int dev_id);
  ~SSD();
  void preForward(const cv::Mat &image);
  void forward();
  void postForward(const cv::Mat &image, std::vector<ObjRect> &detections);
  void enableProfile(TimeStamp *ts);
  bool getPrecision();

private:
  void setMean(std::vector<float> &values);
  void wrapInputLayer(std::vector<cv::Mat>* input_channels);
  void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

  // handle of low level device
  bm_handle_t bm_handle_;
  int dev_id_;

  // runtime helper
  const char **net_names_;
  void *p_bmrt_;

  // network input shape
  int batch_size_;
  int num_channels_;
  cv::Size input_geometry_;

  // network related parameters
  cv::Mat mean_;
  float threshold_;

  // input & output buffers
  bm_tensor_t input_tensor_;
  bm_tensor_t output_tensor_;
  float input_scale;
  float output_scale;
  float *input_f32;
  int8_t *input_int8;
  float *output_;
  bool flag_int8; 
  // for profiling
  TimeStamp *ts_;
};

#endif /* SSD_HPP */
