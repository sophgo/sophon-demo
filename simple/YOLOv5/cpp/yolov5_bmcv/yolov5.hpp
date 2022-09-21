/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/

#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bmnn_utils.h"
#include "utils.hpp"

struct YoloV5Box {
  int x, y, width, height;
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
  float m_objThreshold = 0.5;


  std::vector<std::string> m_class_names;
  int m_class_num = 80; // default is coco names
  int m_net_h, m_net_w;
  int max_batch;

  TimeStamp *m_ts;


  private:
  int pre_process(const std::vector<cv::Mat>& images, int real_batch);
  int post_process(const std::vector<cv::Mat>& images, std::vector<YoloV5BoxVec>& boxes, int real_batch);
  int argmax(float* data, int dsize);
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
  void NMS(YoloV5BoxVec &dets, float nmsConfidence);

  public:
  YoloV5(std::shared_ptr<BMNNContext> context);
  virtual ~YoloV5();

  int Init(float confThresh=0.5, float objThresh=0.5, float nmsThresh=0.5, const std::string& coco_names_file="");
  int Detect(const std::vector<cv::Mat>& images, std::vector<YoloV5BoxVec>& boxes);
  void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
  void enableProfile(TimeStamp *ts);
};





#endif //!YOLOV5_H
