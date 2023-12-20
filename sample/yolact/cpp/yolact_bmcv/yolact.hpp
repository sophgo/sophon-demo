//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef yolact_H
#define yolact_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

// Fix(Mask) #####################################
struct yolactBox {
  int x, y, width, height;
  float score;
  int class_id;
  cv::Mat Mask_prototype;
};
// Fix(Mask) #####################################

using yolactBoxVec = std::vector<yolactBox>;

class Yolact {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;
  std::vector<bm_image> test;
  
  //configuration
  float m_confThreshold= 0.5;
  float m_nmsThreshold = 0.5;
  int m_keep_top_k = 100;

  std::vector<int> conv_ws = {69, 35, 18, 9, 5};
  std::vector<int> conv_hs = {69, 35, 18, 9, 5};
  std::vector<int> scales  = {24, 48, 96, 192, 384};
  std::vector<float> aspect_ratios = {1.0, 0.5, 2.0};
  std::vector<float> variances = {0.1, 0.2};

  std::vector<std::string> m_class_names;
  int m_class_num = 80; // default is coco names
  int m_net_h, m_net_w;
  int max_batch;
  int output_num;
  int min_dim;
  bmcv_convert_to_attr converto_attr, converto_attr1;

  TimeStamp *m_ts;

  private:
  int pre_process(const std::vector<bm_image>& images);
  int post_process(const std::vector<bm_image>& images, std::vector<yolactBoxVec>& boxes);
  std::pair<int, float> argmax(float* data, int dsize, float m_confThreshold);
  yolactBox decode_pos(float x, float y, float w, float h, std::vector<float> prior, int image_width, int image_height);
  std::vector<std::vector<float>> make_prior(int image_width, int image_height);
  //static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
  static float sigmoid(float x);
  void NMS(yolactBoxVec &dets, float nmsConfidence);
  std::pair<int, int> sanitize_coordinates(int _x1, int _x2, int img_size, int padding=0);
  cv::Mat crop_mask(const cv::Mat& masks, std::vector<int> boxes, int padding = 1);

  public:
  Yolact(std::shared_ptr<BMNNContext> context);
  virtual ~Yolact();
  int Init(float confThresh=0.15, float nmsThresh=0.5, int keep_top_k=100, const std::string& coco_names_file="");
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Detect(const std::vector<bm_image>& images, std::vector<yolactBoxVec>& boxes);
  void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
  void draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int right, int bottom, bm_image& frame, bool put_text_flag=false);
  void drawMask(int ClassId, int left, int top, int right, int bottom, int width, int height, cv::Mat Mask_prototype, cv::Mat& frame);
  static bool compareYolactBoxByScore(const yolactBox& box1, const yolactBox& box2);
};

#endif //!yolact_H
