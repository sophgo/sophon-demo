//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

struct YoloV5Box {
  float x, y, width, height;
  float score;
  int class_id;
};

using YoloV5BoxVec = std::vector<YoloV5Box>;

class YoloV5 {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  //configuration
  float m_confThreshold= 0.5;
  float m_nmsThreshold = 0.5;
  bool use_cpu_opt;

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
  int post_process(const std::vector<bm_image>& images, std::vector<YoloV5BoxVec>& boxes);
  int post_process_cpu_opt(const std::vector<bm_image> &images, std::vector<YoloV5BoxVec>& detected_boxes);
  int argmax(float* data, int dsize);
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
  static float sigmoid(float x);
  void NMS(YoloV5BoxVec &dets, float nmsConfidence);

  public:
  YoloV5(std::shared_ptr<BMNNContext> context, bool use_cpu_opt=true);
  virtual ~YoloV5();
  int Init(float confThresh=0.5, float nmsThresh=0.5, const std::string& coco_names_file="");
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Detect(const std::vector<bm_image>& images, std::vector<YoloV5BoxVec>& boxes);
  void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
  void draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int right, int bottom, bm_image& frame, bool put_text_flag=false);
};

#endif //!YOLOV5_H
