//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef DINOV2_HPP
#define DINOV2_HPP

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"

// DINOv2 (dinov2_vitb14_reg, 224x224) 图像分类（bmcv 硬件前处理版本）。
// 模型输出官方线性探针特征 [B, 1536]，再接官方线性头 (weight[1000,1536], bias[1000])。
// 前处理对齐官方 make_classification_eval_transform() 默认口径：
// Resize(短边=256, VPP 不支持 BICUBIC 故用 LINEAR) -> CenterCrop(224) -> (x/255 - mean) / std，
// 全部走 bmcv（vpp_convert + crop + convert_to）。
// 推理接口直接使用 bm_wrapper.hpp 暴露的 bmruntime 底层 API（bmrt/bm_handle/bm_tensor_t），
// 不再依赖 bmnn_utils.h 的 BMNNContext/BMNNNetwork/BMNNTensor 封装（参考 YOLOv8_plus_det 的 cpp）。
class DINOV2 {
  // bmruntime 底层句柄 / bmodel 上下文 / 网络信息
  bm_handle_t handle = NULL;
  void *bmrt = NULL;
  const bm_net_info_t *netinfo = NULL;
  std::vector<std::string> network_names;
  bm_misc_info misc_info;

  // 网络输入 tensor：device mem 由 bm_image 连续内存供给（每次 pre_process 绑定）
  bm_tensor_t input_tensor;

  std::vector<bm_image> m_cropped_imgs;   // RGB_PLANAR uint8, 224x224（vpp resize + crop 后）
  std::vector<bm_image> m_converto_imgs;  // RGB_PLANAR float/int8, 网络输入

  // model info
  int m_net_h;
  int m_net_w;
  int m_resize_size;  // 官方 eval 的 resize 短边（crop 224 时为 256）
  int max_batch;
  int feature_dim;  // 1536
  int class_num;    // 1000
  std::vector<int> m_batches;  // 各 stage 支持的首维 batch（静态 batch 就近对齐）
  bmcv_convert_to_attr converto_attr;

  // 线性头 (weight[class_num, feature_dim] row-major + bias[class_num])
  std::vector<float> m_head_weight;
  std::vector<float> m_head_bias;

  // for profiling
  TimeStamp *ts_ = NULL;

  int load_head(const std::string &head_file);
  int get_nearest_batch(int real_batch);
  int pre_process(std::vector<bm_image> &images);
  int forward(bm_tensor_t &input_tensor, std::vector<bm_tensor_t> &output_tensors);
  float *get_cpu_data(bm_tensor_t *tensor, float scale);
  int post_process(std::vector<bm_image> &images, std::vector<bm_tensor_t> &output_tensors,
                   std::vector<std::pair<int, float>> &results);

  public:
  DINOV2(const std::string &bmodel_file, const std::string &head_file, int dev_id = 0);
  virtual ~DINOV2();
  void enableProfile(TimeStamp *ts);
  int batch_size();
  bm_handle_t get_handle() { return handle; }
  int Classify(std::vector<bm_image>& input_images, std::vector<std::pair<int, float>>& results);
};

#endif /* DINOV2_HPP */