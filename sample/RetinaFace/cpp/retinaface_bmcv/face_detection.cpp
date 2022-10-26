//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
//  face_detection.cpp
//  NelivaSDK
//
//  Created by Bitmain on 2020/11/5.
//  Copyright © 2020年 AnBaolei. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include "face_detection.hpp"

using namespace std;
using namespace cv;

FaceDetection::FaceDetection(const std::string bmodel_path, int device_id) {
  bmodel_path_ = bmodel_path;
  device_id_ = device_id;
  load_model();
  float mean[3] = {104, 117, 123};
  float input_scale = 1.0;
  input_scale *= net_info_->input_scales[0];
  convert_attr_.alpha_0 = input_scale;
  convert_attr_.beta_0 = -input_scale * mean[0];  //0
  convert_attr_.alpha_1 = input_scale;
  convert_attr_.beta_1 = -input_scale * mean[1];  //0
  convert_attr_.alpha_2 = input_scale;
  convert_attr_.beta_2 = -input_scale * mean[2];  //0
  bm_status_t bm_ret = bm_image_create_batch(bm_handle_,
                              net_h_,
                              net_w_,
                              FORMAT_BGR_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE,
                              resize_bmcv_,
                              batch_size_);
  if (BM_SUCCESS != bm_ret) {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }
  bm_status_t ret = bm_image_create_batch(bm_handle_, net_h_, net_w_,
                        FORMAT_BGR_PLANAR,
                        data_type_,
                        scaled_inputs_, batch_size_);
  if (BM_SUCCESS != ret) {
    std::cerr << "ERROR: bm_image_create_batch failed" << std::endl;
    exit(-1);
  }
  shared_ptr<RetinaFacePostProcess> post_ptr(new RetinaFacePostProcess);
  post_process_ = post_ptr;
}

FaceDetection::~FaceDetection() {
}

bool FaceDetection::run(vector<Mat>& input_imgs,
                               vector<vector<stFaceRect> >& results) {
  
  std::vector<bm_image> input_bm_imgs;
  for (size_t i = 0; i < input_imgs.size(); i++) {
    bm_image bmimg;
    bm_image_from_mat(bm_handle_, input_imgs[i], bmimg);
    input_bm_imgs.push_back(bmimg);
  }
  //std::vector<bm_image> processed_imgs;
  std::vector<float> ratios;
  ratios = preprocess(input_bm_imgs);
  // assert(static_cast<int>(input_imgs.size()) == batch_size_);
  // bmcv_image_convert_to(bm_handle_, batch_size_,
  //            convert_attr_, &processed_imgs[0], scaled_inputs_);
  forward();
  postprocess(results, ratios, input_imgs);
  for (size_t i = 0; i < input_bm_imgs.size(); i++) {
    bm_image_destroy(input_bm_imgs[i]);
    //bm_image_destroy(processed_imgs[i]);
  }
  
  return true;
}

float FaceDetection::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  float ratio_w, ratio_h;
  ratio_h = (float) dst_h / src_h;
  ratio_w = (float) dst_w / src_w;
  if (src_w > src_h) {
    // 宽是长边
    *pIsAligWidth = false;
    ratio = (float) dst_w / src_w;
  } else {
    //高是长边
    *pIsAligWidth = true;
    ratio = (float) dst_h / src_h;
  }

  return ratio;
}

std::vector<float> FaceDetection::preprocess(const std::vector<bm_image>& input_imgs) {
  std::vector<bm_image> processed_imgs;
  std::vector<float> ratios;
  for (size_t i = 0; i < input_imgs.size(); i++) {
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(input_imgs[i].width, input_imgs[i].height, net_w_, net_h_, &isAlignWidth);
    ratios.push_back(ratio);
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 0;
    padding_attr.padding_g = 0;
    padding_attr.padding_r = 0;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
      padding_attr.dst_crop_h = net_h_;
      padding_attr.dst_crop_w = input_imgs[i].width*ratio;
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = 0;
    }else{
      padding_attr.dst_crop_h = input_imgs[i].height*ratio;
      padding_attr.dst_crop_w = net_w_;
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = 0;
    }

    bmcv_rect_t crop_rect{0, 0, input_imgs[i].width, input_imgs[i].height};
    auto ret = bmcv_image_vpp_convert_padding(bm_handle_, 1, input_imgs[i], &resize_bmcv_[i],
        &padding_attr, &crop_rect);
  }
  bmcv_image_convert_to(bm_handle_, batch_size_, convert_attr_, resize_bmcv_, scaled_inputs_);
  
  return ratios;
}

void FaceDetection::postprocess(vector<vector<stFaceRect> >& results, vector<float> ratios, vector<Mat>& input_imgs) {
  for (int i = 0; i < batch_size_; i++) {
    float *preds[output_num_];
    vector<stFaceRect> det_result;
    results.push_back(det_result);
    int img_h = input_imgs[i].rows;
    int img_w = input_imgs[i].cols;
    float ratio_ = ratios[i];
    for (int j = 0; j < output_num_; j++) {
      if (BM_FLOAT32 == net_info_->output_dtypes[j]) {
        preds[j] = reinterpret_cast<float*>(outputs_[j]) + output_sizes_[j] * i;
      } else {
        signed char* int8_ptr = reinterpret_cast<signed char*>(outputs_[j])
                                                       + output_sizes_[j] * i;
        preds[j] = new float[output_sizes_[j]];
        for (int k = 0; k < output_sizes_[j]; k++) {
          preds[j][k] = int8_ptr[k] * net_info_->output_scales[j];
        }
      }
    }
    post_process_->run(*net_info_, preds,
                  results[i], img_h, img_w, ratio_, max_face_count_, score_threshold_);
    for (int j = 0; j < output_num_; j++) {
      if (BM_FLOAT32 != net_info_->output_dtypes[j]) {
        delete []preds[j];
      }
    }
  }
  return;
}

 void FaceDetection::set_max_face_count(int max_face_count) {
   max_face_count_ = max_face_count;
 }
 
 void FaceDetection::set_score_threshold(float score_threshold) {
   score_threshold_ = score_threshold;
 }
