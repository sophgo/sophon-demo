//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include "resnet.hpp"

using namespace std;

RESNET::RESNET(bm_handle_t bm_handle, const string bmodel) : p_bmrt_(nullptr)
{

  bool ret;

  // get device handle
  bm_handle_ = bm_handle;

  // init bmruntime contxt
  p_bmrt_ = bmrt_create(bm_handle_);
  if (NULL == p_bmrt_)
  {
    cout << "ERROR: get handle failed!" << endl;
    exit(1);
  }

  // load bmodel from file
  ret = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!ret)
  {
    cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
    exit(1);
  }

  bmrt_get_network_names(p_bmrt_, &net_names_);

  // get model info by model name
  net_info_ = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  if (NULL == net_info_)
  {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }

  if (BM_FLOAT32 == net_info_->input_dtypes[0])
    input_is_int8_ = false;
  else
    input_is_int8_ = true;

  if (BM_FLOAT32 == net_info_->output_dtypes[0])
    output_is_int8_ = false;
  else
    output_is_int8_ = true;

  bm_shape_t input_shape = net_info_->stages[0].input_shapes[0];
  int input_count = bmrt_shape_count(&input_shape);
  net_h_ = input_shape.dims[2];
  net_w_ = input_shape.dims[3];
  batch_size_ = input_shape.dims[0];
  num_channels_ = input_shape.dims[1];
  input_scale_ = net_info_->input_scales[0];

  // allocate output buffer
  bm_shape_t output_shape = net_info_->stages[0].output_shapes[0];
  int output_count = bmrt_shape_count(&output_shape);
  class_num_ = output_shape.dims[1];
  count_per_img = output_count / batch_size_;
  output_scale_ = net_info_->output_scales[0];
  if (output_is_int8_)
  {
    output_int8 = new int8_t[output_count];
  }
  else
  {
    output_fp32 = new float[output_count];
  }

  // init bm images for storing results of combined operation of resize & crop & split
  bm_status_t bm_ret = bm_image_create_batch(bm_handle_,
                                             net_h_,
                                             net_w_,
                                             FORMAT_RGB_PLANAR,
                                             DATA_TYPE_EXT_1N_BYTE,
                                             resize_bmcv_,
                                             MAX_BATCH);
  if (BM_SUCCESS != bm_ret)
  {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }

  // bm images for storing inference inputs
  bm_image_data_format_ext data_type;
  if (input_is_int8_)
  { // INT8
    data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  else
  { // FP32
    data_type = DATA_TYPE_EXT_FLOAT32;
  }
  bm_ret = bm_image_create_batch(bm_handle_,
                                 net_h_,
                                 net_w_,
                                 FORMAT_RGB_PLANAR,
                                 data_type,
                                 linear_trans_bmcv_,
                                 MAX_BATCH);

  if (BM_SUCCESS != bm_ret)
  {
    cout << "ERROR: bm_image_create_batch failed" << endl;
    exit(1);
  }

  // initialize linear transform parameter
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  linear_trans_param_.alpha_0 = 1 / (255. * std[0]) * input_scale_;
  linear_trans_param_.alpha_1 = 1 / (255. * std[1]) * input_scale_;
  linear_trans_param_.alpha_2 = 1 / (255. * std[2]) * input_scale_;
  linear_trans_param_.beta_0 = (-mean[0] / std[0]) * input_scale_;
  linear_trans_param_.beta_1 = (-mean[1] / std[1]) * input_scale_;
  linear_trans_param_.beta_2 = (-mean[2] / std[2]) * input_scale_;
}

RESNET::~RESNET()
{
  // deinit bm images
  bm_image_destroy_batch(resize_bmcv_, MAX_BATCH);
  bm_image_destroy_batch(linear_trans_bmcv_, MAX_BATCH);

  // free output buffer
  if (output_is_int8_)
  {
    delete[] output_int8;
  }
  else
  {
    delete[] output_fp32;
  }

  // deinit contxt handle
  bmrt_destroy(p_bmrt_);
  free(net_names_);
}

void RESNET::enableProfile(TimeStamp *ts)
{
  ts_ = ts;
}

void RESNET::preForward(vector<bm_image> &input)
{
  LOG_TS(ts_, "resnet pre-process")
  preprocess_bmcv(input);
  LOG_TS(ts_, "resnet pre-process")
}

void RESNET::forward()
{
  LOG_TS(ts_, "resnet inference")
  bool res;
  if (output_is_int8_)
  {
    res = bm_inference(p_bmrt_, linear_trans_bmcv_, (int8_t *)output_int8, input_shape_, net_names_[0]);
  }
  else
  {
    res = bm_inference(p_bmrt_, linear_trans_bmcv_, (float *)output_fp32, input_shape_, net_names_[0]);
  }

  LOG_TS(ts_, "resnet inference")
  if (!res)
  {
    cout << "ERROR : inference failed!!" << endl;
    exit(1);
  }
}

static bool comp(const pair<float, int> &lhs,
                 const pair<float, int> &rhs)
{
  return lhs.first > rhs.first;
}

void RESNET::postForward(vector<bm_image> &input, vector<pair<int, float>> &results)
{
  LOG_TS(ts_, "resnet post-process")
  results.clear();

  for (int i = 0; i < batch_size_; i++)
  {
    float exp_sum = 0;
    for (int j = 0; j < class_num_; j++)
    {
      if (output_is_int8_)
      {
        exp_sum += std::exp(output_int8[count_per_img * i + j] * output_scale_);
      }
      else
      {
        exp_sum += std::exp(output_fp32[count_per_img * i + j]);
      }
    }
    int max_idx = -1;
    float max_score = -1;
    for (int j = 0; j < class_num_; j++)
    {
      float score = 0;
      if (output_is_int8_)
      {
        score = std::exp(output_int8[count_per_img * i + j] * output_scale_) / exp_sum;
      }
      else
      {
        score = std::exp(output_fp32[count_per_img * i + j]) / exp_sum;
      }
      if (max_score < score)
      {
        max_score = score;
        max_idx = j;
      }
    }

#ifdef DEBUG
    cout << max_idx << ": " << max_score << endl;
#endif
    results.push_back({max_idx, max_score});
  }
  LOG_TS(ts_, "resnet post-process")
}

void RESNET::preprocess_bmcv(vector<bm_image> &input)
{

  // set input shape according to input bm images
  input_shape_ = {4, {(int)input.size(), 3, net_h_, net_w_}};

  // resize && split by bmcv
  for (size_t i = 0; i < input.size(); i++)
  {
    LOG_TS(ts_, "resnet pre-process-vpp")
    bmcv_rect_t crop_rect_ = {0, 0, input[i].width, input[i].height};
    bmcv_image_vpp_convert(bm_handle_, 1, input[i], &resize_bmcv_[i], &crop_rect_);
    LOG_TS(ts_, "resnet pre-process-vpp")
  }
  // do linear transform
  LOG_TS(ts_, "linear transform")
  bmcv_image_convert_to(bm_handle_, input.size(), linear_trans_param_, resize_bmcv_, linear_trans_bmcv_);
  LOG_TS(ts_, "linear transform")
}

int RESNET::batch_size()
{
  return batch_size_;
};
