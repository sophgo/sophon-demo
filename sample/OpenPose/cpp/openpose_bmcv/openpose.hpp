//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef OpenPose_HPP
#define OpenPose_HPP

#include <string>
#include <iostream>
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "bmnn_utils.h"

#define BUFFER_SIZE (1024 * 500)

struct PoseKeyPoints{
  enum EModelType {
      BODY_25 = 0,
      COCO_18 = 1
  };
  std::vector<float> keypoints;
  std::vector<int> shape;
  int width, height;
  EModelType modeltype;
};

class OpenPose {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;
  int m_net_h, m_net_w;
  int max_batch;
  PoseKeyPoints::EModelType m_model_type;
  bmcv_convert_to_attr converto_attr;
  int output_num;
  // cv::Size netInputSize;
  float nms_threshold;
  TimeStamp *ts_ = NULL;
  tpu_kernel_function_t func_id;
public:
  OpenPose(std::shared_ptr<BMNNContext> context);
  ~OpenPose();
  int Init(bool use_tpu_kernel_post);
  int Detect(const std::vector<bm_image>& input_images, std::vector<PoseKeyPoints>& vct_keypoints, std::string& performance_opt);
  void enableProfile(TimeStamp *ts);
  int batch_size();
  PoseKeyPoints::EModelType get_model_type();
private:
  int pre_process(const std::vector<bm_image>& images);
  int post_process(const std::vector<bm_image>& images, std::vector<PoseKeyPoints>& vct_keypoints, std::string& performance_opt);
};

#endif /* OpenPose_HPP */
