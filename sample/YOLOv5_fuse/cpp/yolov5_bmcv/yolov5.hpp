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


  std::vector<std::string> m_class_names;
  int m_class_num = 80; // default is coco names
  int m_net_h, m_net_w;
  int max_batch;
  int output_num;
  int min_dim;
  bmcv_convert_to_attr converto_attr;

  TimeStamp *m_ts;

  private:
  int prepare_data(const std::vector<bm_image>& images);
  int get_result(const std::vector<bm_image>& images, std::vector<YoloV5BoxVec>& boxes);
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);

  public:
  YoloV5(std::shared_ptr<BMNNContext> context);
  virtual ~YoloV5();
  int Init(const std::string& coco_names_file="");
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Detect(const std::vector<bm_image>& images, std::vector<YoloV5BoxVec>& boxes);
  void draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int right, int bottom, bm_image& frame, float draw_thresh, bool put_text_flag=false);
};

#endif //!YOLOV5_H
