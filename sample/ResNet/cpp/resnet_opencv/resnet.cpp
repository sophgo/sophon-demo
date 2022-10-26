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
#include "utils.hpp"

using namespace std;

RESNET::RESNET(const string bmodel, int dev_id)
{
  // init device id
  dev_id_ = dev_id;

  // create device handle
  bm_status_t ret = bm_dev_request(&bm_handle_, dev_id_);
  if (BM_SUCCESS != ret)
  {
    std::cout << "ERROR: bm_dev_request err=" << ret << std::endl;
    exit(-1);
  }

  // init bmruntime contxt
  p_bmrt_ = bmrt_create(bm_handle_);
  if (NULL == p_bmrt_)
  {
    cout << "ERROR: get handle failed!" << endl;
    exit(1);
  }

  // load bmodel by file
  bool flag = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!flag)
  {
    cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
    exit(-1);
  }

  bmrt_get_network_names(p_bmrt_, &net_names_);
  cout << "> Load model " << net_names_[0] << " successfully" << endl;

  // get model info by model name
  auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  if (NULL == net_info)
  {
    cout << "ERROR: get net-info failed!" << endl;
    exit(1);
  }

  cout << "input scale:" << net_info->input_scales[0] << endl;
  cout << "output scale:" << net_info->output_scales[0] << endl;

  /* get fp32/int8 type, the thresholds may be different */
  if (BM_FLOAT32 == net_info->input_dtypes[0])
  {
    int8_flag_ = false;
    cout << "fp32 input" << endl;
  }
  else
  {
    int8_flag_ = true;
    cout << "int8 input" << endl;
  }
  if (BM_FLOAT32 == net_info->output_dtypes[0])
  {
    int8_output_flag = false;
    cout << "fp32 output" << endl;
  }
  else
  {
    int8_output_flag = true;
    cout << "int8 output" << endl;
  }

#ifdef DEBUG
  bmrt_print_network_info(net_info);
#endif

  bm_shape_t input_shape = net_info->stages[0].input_shapes[0];
  /* attach device_memory for inference input data */
  bmrt_tensor(&input_tensor_, p_bmrt_, net_info->input_dtypes[0], input_shape);
  /* malloc input and output system memory for preprocess data */
  int input_count = bmrt_shape_count(&input_shape);
  net_h_ = input_shape.dims[2];
  net_w_ = input_shape.dims[3];
  batch_size_ = input_shape.dims[0];
  num_channels_ = input_shape.dims[1];
  input_scale_ = net_info->input_scales[0];
  cout << "input count:" << input_count << endl;
  if (int8_flag_)
  {
    input_int8 = new int8_t[input_count];
  }
  else
  {
    input_f32 = new float[input_count];
  }

  bm_shape_t output_shape = net_info->stages[0].output_shapes[0];
  bmrt_tensor(&output_tensor_, p_bmrt_, net_info->output_dtypes[0], output_shape);
  int output_count = bmrt_shape_count(&output_shape);
  class_num_ = output_shape.dims[1];
  count_per_img = output_count / batch_size_;
  output_scale_ = net_info->output_scales[0];
  if (int8_output_flag)
  {
    output_int8 = new int8_t[output_count];
  }
  else
  {
    output_f32 = new float[output_count];
  }



  vector<float> scale_values;
  scale_values.push_back(0.017124753831663668);
  scale_values.push_back(0.01750700280112045);
  scale_values.push_back(0.017429193899782137);

  vector<float> mean_values;
  mean_values.push_back(-2.1179039301310043);
  mean_values.push_back(-2.0357142857142856);
  mean_values.push_back(-1.8044444444444445);
  setStdMean(scale_values, mean_values);

  ts_ = nullptr;
}

RESNET::~RESNET()
{
  if (int8_flag_)
  {
    delete[] input_int8;
  }
  else
  {
    delete[] input_f32;
  }
  if (int8_output_flag)
  {
    delete[] output_int8;
  }
  else
  {
    delete[] output_f32;
  }
  bm_free_device(bm_handle_, input_tensor_.device_mem);
  bm_free_device(bm_handle_, output_tensor_.device_mem);
  free(net_names_);
  bmrt_destroy(p_bmrt_);
  bm_dev_free(bm_handle_);
}

void RESNET::enableProfile(TimeStamp *ts)
{
  ts_ = ts;
}

void RESNET::preForward(const vector<cv::Mat> &images)
{
  LOG_TS(ts_, "resnet pre-process")
  for (int i = 0; i < batch_size_; i++)
  {
    vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels, i);
    preprocess(images[i], &input_channels);
  }
  LOG_TS(ts_, "resnet pre-process")
}

void RESNET::forward()
{
  if (int8_flag_)
  {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_int8));
  }
  else
  {
    bm_memcpy_s2d(bm_handle_, input_tensor_.device_mem, ((void *)input_f32));
  }
  LOG_TS(ts_, "resnet inference")
  bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_names_[0],
                                   &input_tensor_, 1, &output_tensor_, 1, true, false);
  if (!ret)
  {
    cout << "ERROR: Failed to launch network" << net_names_[0] << "inference" << endl;
  }

  // sync, wait for finishing inference
  bm_thread_sync(bm_handle_);
  LOG_TS(ts_, "resnet inference")

  size_t size = bmrt_tensor_bytesize(&output_tensor_);
  if (int8_output_flag)
  {
    bm_memcpy_d2s_partial(bm_handle_, output_int8, output_tensor_.device_mem, size);
  }
  else
  {
    bm_memcpy_d2s_partial(bm_handle_, output_f32, output_tensor_.device_mem, size);
  }
}

static bool comp(const std::pair<float, int> &lhs,
                 const std::pair<float, int> &rhs)
{
  return lhs.first > rhs.first;
}

void RESNET::postForward(vector<pair<int, float>> &results)
{

  LOG_TS(ts_, "resnet post-process")

  results.clear();

  for (int i = 0; i < batch_size_; i++)
  {
    float exp_sum = 0;
    for (int j = 0; j < class_num_; j++)
    {
      if (int8_output_flag)
      {
        exp_sum += std::exp(output_int8[count_per_img * i + j] * output_scale_);
      }
      else
      {
        exp_sum += std::exp(output_f32[count_per_img * i + j]);
      }
    }
    int max_idx = -1;
    float max_score = -1;
    for (int j = 0; j < class_num_; j++)
    {
      float score = 0;
      if (int8_output_flag)
      {
        score = std::exp(output_int8[count_per_img * i + j] * output_scale_) / exp_sum;
      }
      else
      {
        score = std::exp(output_f32[count_per_img * i + j]) / exp_sum;
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

void RESNET::setStdMean(vector<float> &std, vector<float> &mean)
{
  // init mat mean_
  vector<cv::Mat> std_channels;
  vector<cv::Mat> mean_channels;
  for (int i = 0; i < num_channels_; i++)
  {
    /* Extract an individual channel. */
    cv::Mat std_channel(net_h_, net_w_, CV_32FC1, cv::Scalar((float)std[i]), cv::SophonDevice(this->dev_id_));
    std_channels.push_back(std_channel);
    cv::Mat mean_channel(net_h_, net_w_, CV_32FC1, cv::Scalar((float)mean[i]), cv::SophonDevice(this->dev_id_));
    mean_channels.push_back(mean_channel);
  }
  if (int8_flag_)
  {
    std_.create(net_h_, net_w_, CV_8SC3, dev_id_);
    mean_.create(net_h_, net_w_, CV_8SC3, dev_id_);
  }
  else
  {
    std_.create(net_h_, net_w_, CV_32FC3, dev_id_);
    mean_.create(net_h_, net_w_, CV_32FC3, dev_id_);
  }

  cv::merge(std_channels, std_);
  cv::merge(mean_channels, mean_);
  cout << "mean_size:" << mean_.size().height << "," << mean_.size().width << "," << mean_.channels() << endl;
  cout << "std_size:" << std_.size().height << "," << std_.size().width << "," << std_.channels() << endl;
}

void RESNET::wrapInputLayer(std::vector<cv::Mat> *input_channels, int batch_id)
{
  int h = net_h_;
  int w = net_w_;

  // init input_channels
  if (int8_flag_)
  {
    int8_t *channel_base = input_int8;
    channel_base += h * w * num_channels_ * batch_id;
    for (int i = 0; i < num_channels_; i++)
    {
      cv::Mat channel(h, w, CV_8SC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  }
  else
  {
    float *channel_base = input_f32;
    channel_base += h * w * num_channels_ * batch_id;
    for (int i = 0; i < num_channels_; i++)
    {
      cv::Mat channel(h, w, CV_32FC1, channel_base);
      input_channels->push_back(channel);
      channel_base += h * w;
    }
  }
}

void RESNET::preprocess(const cv::Mat &img, vector<cv::Mat> *input_channels)
{
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample = img;
  cv::Mat sample_resized(net_h_, net_w_, CV_8UC3, cv::SophonDevice(dev_id_));
  if (sample.size() != cv::Size(net_w_, net_h_))
  {
    cv::resize(sample, sample_resized, cv::Size(net_w_, net_h_));
  }
  else
  {
    sample_resized = sample;
  }
  cv::Mat sample_resized_rgb(cv::SophonDevice(this->dev_id_));
  cv::cvtColor(sample_resized, sample_resized_rgb, cv::COLOR_BGR2RGB);

  cv::Mat sample_float(cv::SophonDevice(this->dev_id_));
  sample_resized_rgb.convertTo(sample_float, CV_32FC3);

  cv::Mat sample_multiply(cv::SophonDevice(this->dev_id_));
  cv::Mat sample_normalized(cv::SophonDevice(this->dev_id_));
  cv::multiply(sample_float, std_, sample_multiply);
  cv::add(sample_multiply, mean_, sample_normalized);

  // /*note: int8 in convert need mul input_scale*/
  if (int8_flag_)
  {
    cv::Mat sample_int8(cv::SophonDevice(this->dev_id_));
    sample_normalized.convertTo(sample_int8, CV_8SC1, input_scale_);
    cv::split(sample_int8, *input_channels);
  }
  else
  {
    cv::Mat sample_fp32(cv::SophonDevice(this->dev_id_));
    sample_normalized.convertTo(sample_fp32, CV_32FC3, input_scale_);
    cv::split(sample_fp32, *input_channels);
  }
}

int RESNET::batch_size()
{
  return batch_size_;
};