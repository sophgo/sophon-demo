//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "ram.hpp"
#define DUMP_FILE 0

using namespace std;

static std::vector<std::string> read_lines(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("cannot open file: " + path);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    lines.push_back(line);
  }
  return lines;
}

RAM::RAM(std::shared_ptr<BMNNContext> context, int dev_id)
    : m_bmContext(context), m_dev_id(dev_id) {
  std::cout << "RAM create bm_context" << std::endl;
}

int RAM::Init(const std::string &tag_list_file,
              const std::string &tag_list_chinese_file,
              const std::string &tag_list_threshold_file) {
  // 1. get network
  m_bmNetwork = m_bmContext->network(0);

  // 2. get input
  max_batch = m_bmNetwork->maxBatch();
  m_input_tensor = m_bmNetwork->inputTensor(0);
  m_num_channels = m_input_tensor->get_shape()->dims[1];
  m_net_h = m_input_tensor->get_shape()->dims[2];
  m_net_w = m_input_tensor->get_shape()->dims[3];
  m_input_count = bmrt_shape_count(m_input_tensor->get_shape());
  input_scale = m_input_tensor->get_scale();
  if (m_input_tensor->get_dtype() == BM_INT8) {
    m_input_int8 = new int8_t[m_input_count];
  } else {
    m_input_float = new float[m_input_count];
  }

  // 3. get output
  m_output_tensor = m_bmNetwork->outputTensor(0);
  output_scale = m_output_tensor->get_scale();
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num > 0);
  class_num = m_bmNetwork->outputTensor(0)->get_shape()->dims[1];

  // 4. set mean and scale
  //    matches torchvision Normalize(mean=[0.485,0.456,0.406],
  //    std=[0.229,0.224,0.225]) after ToTensor (scales by 1/255)
  const std::vector<float> mean_v = {0.485, 0.456, 0.406};
  const std::vector<float> std_v = {0.229, 0.224, 0.225};
  vector<float> scale_values;
  scale_values.push_back(1 / (255. * std_v[0]));
  scale_values.push_back(1 / (255. * std_v[1]));
  scale_values.push_back(1 / (255. * std_v[2]));
  vector<float> mean_values;
  mean_values.push_back(-mean_v[0] / std_v[0]);
  mean_values.push_back(-mean_v[1] / std_v[1]);
  mean_values.push_back(-mean_v[2] / std_v[2]);
  setStdMean(scale_values, mean_values);

  // 5. set device mem
  bmrt_tensor(&input_tensor,
              m_bmContext->bmrt(),
              m_input_tensor->get_dtype(),
              *m_input_tensor->get_shape());
  m_input_tensor->set_device_mem(&input_tensor.device_mem);

  // 6. load tag lists and per-class thresholds
  tag_list = read_lines(tag_list_file);
  tag_list_chinese = read_lines(tag_list_chinese_file);
  class_threshold.assign(class_num, 1.0f);
  std::ifstream tf(tag_list_threshold_file);
  if (tf.is_open()) {
    for (int i = 0; i < class_num; ++i) {
      std::string s;
      if (!std::getline(tf, s)) break;
      if (!s.empty() && s.back() == '\r') s.pop_back();
      try {
        class_threshold[i] = std::stof(s);
      } catch (...) {
        // keep default 1.0 if line is not a float
      }
    }
  }

  return 0;
}

void RAM::enableProfile(TimeStamp *ts) { ts_ = ts; }

void RAM::setStdMean(vector<float> &std, vector<float> &mean) {
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
  } else {
    m_std.create(m_net_h, m_net_w, CV_32FC3, m_dev_id);
    m_mean.create(m_net_h, m_net_w, CV_32FC3, m_dev_id);
  }

  cv::merge(std_channels, m_std);
  cv::merge(mean_channels, m_mean);
  cout << "mean_size:" << m_mean.size().height << "," << m_mean.size().width << "," << m_mean.channels() << endl;
  cout << "std_size:" << m_std.size().height << "," << m_std.size().width << "," << m_std.channels() << endl;
}

void RAM::wrapInputLayer(std::vector<cv::Mat>* input_channels, int batch_id) {
  int h = m_net_h;
  int w = m_net_w;

  //init input_channels
  if (m_input_tensor->get_dtype() == BM_INT8) {
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

int RAM::batch_size() {
  return max_batch;
};

int RAM::Tag(std::vector<cv::Mat>& input_images,
             std::vector<std::pair<std::string, std::string>> &results) {
  int ret = 0;
  // 1. preprocess
  ts_->save("ram preprocess", max_batch);
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  ts_->save("ram preprocess", max_batch);

  // 2. forward
  ts_->save("ram inference", max_batch);
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  ts_->save("ram inference", max_batch);

  // 3. post process
  ts_->save("ram postprocess", max_batch);
  ret = post_process(input_images, results);
  CV_Assert(ret == 0);
  ts_->save("ram postprocess", max_batch);

  return ret;
}

RAM::~RAM() {
  std::cout << "RAM delete bm_context" << std::endl;
  bm_free_device(m_bmContext->handle(), input_tensor.device_mem);
  if (m_input_tensor->get_dtype() == BM_INT8) {
    delete [] m_input_int8;
  } else {
    delete [] m_input_float;
  }
}

int RAM::post_process(
    vector<cv::Mat> &images,
    vector<pair<std::string, std::string>> &results) {
  results.clear();
  m_output_tensor = m_bmNetwork->outputTensor(0);
  float* output_data = (float*)m_output_tensor->get_cpu_data();

  for (unsigned int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
    std::string tag_output;
    std::string tag_output_chinese;
    bool first = true;
    for (int j = 0; j < class_num; j++) {
      float logit = *(output_data + batch_idx * class_num + j);
      float sigmoid = 1.0f / (1.0f + std::exp(-logit));
      if (sigmoid > class_threshold[j] && j < (int)tag_list.size()) {
        if (!first) {
          tag_output += " | ";
          tag_output_chinese += " | ";
        }
        tag_output += tag_list[j];
        tag_output_chinese += tag_list_chinese[j];
        first = false;
      }
    }
    results.push_back({tag_output, tag_output_chinese});
  }

  return 0;
}

void RAM::pre_process_image(const cv::Mat& img, std::vector<cv::Mat> *input_channels) {
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

int RAM::pre_process(vector<cv::Mat> &images) {
  //Safety check.
  assert(images.size() <= max_batch);

  //1. Preprocess input images in host memory.
  for (int i = 0; i < max_batch; i++) {
    std::vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels, i);
    if (i < (int)images.size())
      pre_process_image(images[i], &input_channels);
    else {
      cv::Mat tmp = cv::Mat::zeros(m_net_h, m_net_w, CV_32FC3);
      pre_process_image(tmp, &input_channels);
    }
  }
  //2. Attach to input tensor.
  if (m_input_tensor->get_dtype() == BM_INT8) {
    bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_int8);
  } else {
    bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_float);
  }

  return 0;
}
