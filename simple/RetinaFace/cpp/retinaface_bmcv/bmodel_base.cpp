//
//  bmodel_base.cpp
//  NelivaSDK
//
//  Created by Bitmain on 2020/11/5.
//  Copyright © 2020年 AnBaolei. All rights reserved.
//

#include "bmodel_base.hpp"

BmodelBase::~BmodelBase() {
  if (batch_size_ >= 1) {
    bm_image_destroy_batch(scaled_inputs_, batch_size_);
    bm_image_destroy_batch(resize_bmcv_, batch_size_);
  }
  if (scaled_inputs_) {
    delete []scaled_inputs_;
    delete []resize_bmcv_;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    if (BM_FLOAT32 == net_info_->output_dtypes[i]) {
      delete [] reinterpret_cast<float*>(outputs_[i]);
    } else {
      delete [] reinterpret_cast<signed char*>(outputs_[i]);
    }
  }
  if (net_names_ != nullptr) {
    free(net_names_);
  }
  if (p_bmrt_ != nullptr) {
    bmrt_destroy(p_bmrt_);
  }
  bm_dev_free(bm_handle_);
}

bm_handle_t BmodelBase::get_bm_handle() {
  return bm_handle_;
}

void BmodelBase::forward() {
  bool res = bm_inference(p_bmrt_,
         scaled_inputs_, outputs_, input_shape_,
             reinterpret_cast<const char*>(net_names_[0]));
  if (!res) {
    std::cout << "ERROR : inference failed!!"<< std::endl;
    exit(1);
  }
  return;
}

void BmodelBase::load_model() {
  bm_dev_request(&bm_handle_, device_id_);
  p_bmrt_ = bmrt_create(bm_handle_);
  bool flag = bmrt_load_bmodel(p_bmrt_, bmodel_path_.c_str());
  if (!flag) {
    std::cout << "ERROR: failed to load bmodel[" << bmodel_path_ << "] " << std::endl;
    exit(-1);
  }
  bmrt_get_network_names(p_bmrt_, &net_names_);
  std::cout << "> Load model " << net_names_[0] << " successfully" << std::endl;
  net_info_ = const_cast<bm_net_info_t*>(bmrt_get_network_info(p_bmrt_, net_names_[0]));
  std::cout << "input scale:" << net_info_->input_scales[0] << std::endl;
  std::cout << "output scale:" << net_info_->output_scales[0] << std::endl;
  std::cout << "input number:" << net_info_->input_num << std::endl;
  std::cout << "output number:" << net_info_->output_num << std::endl;

  if (BM_FLOAT32 == net_info_->input_dtypes[0]) {
    std::cout <<  "fp32 model" << std::endl;
    data_type_ = DATA_TYPE_EXT_FLOAT32;
  } else {
    std::cout <<  "int8 model" << std::endl;
  }
  auto &input_shape = net_info_->stages[0].input_shapes[0];
  int count = static_cast<int>(bmrt_shape_count(&input_shape));
  std::cout << "input count:" << count << std::endl;
  output_num_ = net_info_->output_num;
  for (int i = 0; i < output_num_; i++) {
    auto &output_shape = net_info_->stages[0].output_shapes[i];
    count = static_cast<int>(bmrt_shape_count(&output_shape));
    std::cout << "output " << i << " count:" << count << std::endl;
    if (BM_FLOAT32 == net_info_->output_dtypes[i]) {
      float* out = new float[count];
      outputs_.push_back(out);
    } else {
      signed char* out = new signed char[count];
      outputs_.push_back(out);
    }
    output_sizes_.push_back(count / output_shape.dims[0]);
  }

  batch_size_ = input_shape.dims[0];
  num_channels_ = input_shape.dims[1];
  net_h_ = input_shape.dims[2];
  net_w_ = input_shape.dims[3];
  input_shape_ = {4, {batch_size_, 3, net_h_, net_w_}};
  scaled_inputs_ = new bm_image[batch_size_];
  resize_bmcv_ = new bm_image[batch_size_];
  return;
}

int BmodelBase::batch_size() {
  return batch_size_;
};
