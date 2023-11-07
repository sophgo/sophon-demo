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
#define DUMP_FILE 0

using namespace std;

RESNET::RESNET(std::shared_ptr<BMNNContext> context, int dev_id) : m_bmContext(context), m_dev_id(dev_id) {
  std::cout << "ResNet create bm_context" << std::endl;
}

int RESNET::Init() {
  //1. get network
  m_bmNetwork = m_bmContext->network(0);
  
  //2. get input
  max_batch = m_bmNetwork->maxBatch();
  m_input_tensor = m_bmNetwork->inputTensor(0);
  m_num_channels = m_input_tensor->get_shape()->dims[1];
  m_net_h = m_input_tensor->get_shape()->dims[2];
  m_net_w = m_input_tensor->get_shape()->dims[3];
  m_input_count = bmrt_shape_count(m_input_tensor->get_shape());
  input_scale = m_input_tensor->get_scale();
  if(m_input_tensor->get_dtype() == BM_INT8) {
      m_input_int8 = new int8_t[m_input_count];
  } else {
      m_input_float = new float[m_input_count];
  }

  //3. get output
  m_output_tensor = m_bmNetwork->outputTensor(0);
  output_scale = m_output_tensor->get_scale();
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num > 0);
  class_num = m_bmNetwork->outputTensor(0)->get_shape()->dims[1];

  //4. set mean and scale
  vector<float> scale_values;
  scale_values.push_back(0.017124753831663668);
  scale_values.push_back(0.01750700280112045);
  scale_values.push_back(0.017429193899782137);

  vector<float> mean_values;
  mean_values.push_back(-2.1179039301310043);
  mean_values.push_back(-2.0357142857142856);
  mean_values.push_back(-1.8044444444444445);
  setStdMean(scale_values, mean_values);

  //5. set device mem
  bmrt_tensor(&input_tensor, 
              m_bmContext->bmrt(), 
              m_input_tensor->get_dtype(), 
              *m_input_tensor->get_shape());
  m_input_tensor->set_device_mem(&input_tensor.device_mem);
  return 0;
}

void RESNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void RESNET::setStdMean(vector<float> &std, vector<float> &mean) {
  // init mat m_mean
  vector<cv::Mat> std_channels;
  vector<cv::Mat> mean_channels;
  for (int i = 0; i < m_num_channels; i++) {
    /* Extract an individual channel. */
    cv::Mat std_channel(m_net_h, m_net_w, CV_32FC1, cv::Scalar((float)std[i]), cv::SophonDevice(this->m_dev_id));
    std_channels.push_back(std_channel);
    cv::Mat mean_channel(m_net_h, m_net_w, CV_32FC1, cv::Scalar((float)mean[i]), cv::SophonDevice(this->m_dev_id));
    mean_channels.push_back(mean_channel);
  }
  // Todo: fp16
  if (m_input_tensor->get_dtype() == BM_INT8) {
    m_std.create(m_net_h, m_net_w, CV_8SC3, m_dev_id);
    m_mean.create(m_net_h, m_net_w, CV_8SC3, m_dev_id);
  }
  else {
    m_std.create(m_net_h, m_net_w, CV_32FC3, m_dev_id);
    m_mean.create(m_net_h, m_net_w, CV_32FC3, m_dev_id);
  }

  cv::merge(std_channels, m_std);
  cv::merge(mean_channels, m_mean);
  cout << "mean_size:" << m_mean.size().height << "," << m_mean.size().width << "," << m_mean.channels() << endl;
  cout << "std_size:" << m_std.size().height << "," << m_std.size().width << "," << m_std.channels() << endl;
}

void RESNET::wrapInputLayer(std::vector<cv::Mat>* input_channels, int batch_id) {
  int h = m_net_h;
  int w = m_net_w;

  //init input_channels
  if(m_input_tensor->get_dtype() == BM_INT8) {
    int8_t *channel_base = m_input_int8;
    channel_base += h * w * m_num_channels * batch_id;
    for (int i = 0; i < m_num_channels; i++) {
    cv::Mat channel(h, w, CV_8SC1, channel_base);
    input_channels->push_back(channel);
    channel_base += h * w;
    }
  } else {
    float *channel_base = m_input_float;
    channel_base += h * w * m_num_channels * batch_id;
    for (int i = 0; i < m_num_channels; i++) {
    cv::Mat channel(h, w, CV_32FC1, channel_base);
    input_channels->push_back(channel);
    channel_base += h * w;
    }
  }
}

int RESNET::batch_size() {
  return max_batch;
};

int RESNET::Classify(std::vector<cv::Mat>& input_images, std::vector<std::pair<int, float>>& results) {
  int ret = 0;
  // 1. preprocess
  LOG_TS(ts_, "resnet preprocess");
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  LOG_TS(ts_, "resnet preprocess");

  // 2. forward
  LOG_TS(ts_, "resnet inference");
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  LOG_TS(ts_, "resnet inference");

  // 3. post process
  LOG_TS(ts_, "resnet postprocess");
  ret = post_process(input_images, results);
  CV_Assert(ret == 0);
  LOG_TS(ts_, "resnet postprocess");

  return ret;
}

RESNET::~RESNET() {
  std::cout << "ResNet delete bm_context" << std::endl;  
  bm_free_device(m_bmContext->handle(), input_tensor.device_mem);
  if(m_input_tensor->get_dtype() == BM_INT8) {
      delete [] m_input_int8;
  } else {
      delete [] m_input_float;
  }
}

int RESNET::post_process(vector<cv::Mat> &images, vector<pair<int, float>> &results) {
  results.clear();
  m_output_tensor = m_bmNetwork->outputTensor(0);
  output_scale = m_output_tensor->get_scale();
  float* output_data = (float*)m_output_tensor->get_cpu_data();

  for(unsigned int batch_idx = 0; batch_idx < images.size(); ++ batch_idx) {
    float exp_sum = 0;
    for (int j = 0; j < class_num; j++) {
      exp_sum += std::exp(*(output_data + batch_idx * class_num + j) * output_scale);
    }
    int max_idx = -1;
    float max_score = -1;
    for (int j = 0; j < class_num; j++) {
      float score = 0;
      score = std::exp(*(output_data + batch_idx * class_num + j) * output_scale) / exp_sum;
      if (max_score < score) {
        max_score = score;
        max_idx = j;
      }
    }

#ifdef DEBUG
    cout << max_idx << ": " << max_score << endl;
#endif
    results.push_back({max_idx, max_score});
  }

  return 0;
}

void RESNET::pre_process_image(const cv::Mat& img, std::vector<cv::Mat> *input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample = img;
  cv::Mat sample_resized(m_net_h, m_net_w, CV_8UC3, cv::SophonDevice(m_dev_id));
  if (sample.size() != cv::Size(m_net_w, m_net_h)) {
    cv::resize(sample, sample_resized, cv::Size(m_net_w, m_net_h));
  } else {
    sample_resized = sample;
  }
  cv::Mat sample_resized_rgb(m_net_h, m_net_w, CV_8UC3, cv::SophonDevice(this->m_dev_id));
  cv::cvtColor(sample_resized, sample_resized_rgb, cv::COLOR_BGR2RGB);

  cv::Mat sample_float(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(this->m_dev_id));
  sample_resized_rgb.convertTo(sample_float, CV_32FC3);

  cv::Mat sample_multiply(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(this->m_dev_id));
  cv::Mat sample_normalized(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(this->m_dev_id));
  cv::multiply(sample_float, m_std, sample_multiply);
  cv::add(sample_multiply, m_mean, sample_normalized);

  // // /*note: int8 in convert need mul input_scale*/
  if (m_input_tensor->get_dtype() == BM_INT8) {
    cv::Mat sample_int8(m_net_h, m_net_w, CV_8SC1, cv::SophonDevice(this->m_dev_id));
    sample_normalized.convertTo(sample_int8, CV_8SC1, input_scale);
    cv::split(sample_int8, *input_channels);
  } else {
    cv::Mat sample_fp32(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(this->m_dev_id));
    sample_normalized.convertTo(sample_fp32, CV_32FC3, input_scale);
    cv::split(sample_fp32, *input_channels);
  }
}

int RESNET::pre_process(vector<cv::Mat> &images) {
  //Safety check.
  assert(images.size() <= max_batch);
  
  //1. Preprocess input images in host memory.
  for(int i = 0; i < max_batch; i++){
    std::vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels, i);
    if(i < images.size())
      pre_process_image(images[i], &input_channels);
    else {
      cv::Mat tmp = cv::Mat::zeros(m_net_h, m_net_w, CV_32FC3);
      pre_process_image(tmp, &input_channels);
    }
  }
  //2. Attach to input tensor.
  if(m_input_tensor->get_dtype() == BM_INT8) {
    bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_int8);
  } else {
    bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_float);
  }

  return 0;
}
