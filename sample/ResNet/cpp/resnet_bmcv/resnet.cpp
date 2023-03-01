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
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0

using namespace std;

RESNET::RESNET(std::shared_ptr<BMNNContext> context) : m_bmContext(context) {
  std::cout << "ResNet ctor .." << std::endl;
}

int RESNET::Init() {
  //1. get network
  m_bmNetwork = m_bmContext->network(0);
  
  //2. get input
  max_batch = m_bmNetwork->maxBatch();
  auto tensor = m_bmNetwork->inputTensor(0);
  m_net_h = tensor->get_shape()->dims[2];
  m_net_w = tensor->get_shape()->dims[3];

  //3. get output
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num > 0);
  class_num = m_bmNetwork->outputTensor(0)->get_shape()->dims[1];
  
  //4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for(int i=0; i<max_batch; i++){
    auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8){
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
  assert(BM_SUCCESS == ret);

  // 5.converto
  float input_scale = tensor->get_scale();
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  converto_attr.alpha_0 = 1 / (255. * std[0]) * input_scale;
  converto_attr.alpha_1 = 1 / (255. * std[1]) * input_scale;
  converto_attr.alpha_2 = 1 / (255. * std[2]) * input_scale;
  converto_attr.beta_0 = (-mean[0] / std[0]) * input_scale;
  converto_attr.beta_1 = (-mean[1] / std[1]) * input_scale;
  converto_attr.beta_2 = (-mean[2] / std[2]) * input_scale;

  return 0;
}

void RESNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

int RESNET::batch_size() {
  return max_batch;
};

int RESNET::Classify(std::vector<bm_image>& input_images, std::vector<std::pair<int, float>>& results) {
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
  std::cout << "ResNet dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int RESNET::post_process(vector<bm_image> &images, vector<pair<int, float>> &results) {
  results.clear();
  assert(output_num == 1);
  std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(output_num - 1);
  float* output_data = (float*)outputTensor->get_cpu_data();
  auto output_scale = outputTensor->get_scale();

  for(unsigned int batch_idx = 0; batch_idx < images.size(); ++ batch_idx)
  {
    float exp_sum = 0;
    for (int j = 0; j < class_num; j++)
    {
      exp_sum += std::exp(*(output_data + batch_idx * class_num + j) * output_scale);
    }
    int max_idx = -1;
    float max_score = -1;
    for (int j = 0; j < class_num; j++)
    {
      float score = 0;
      score = std::exp(*(output_data + batch_idx * class_num + j) * output_scale) / exp_sum;
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

  return 0;
}

int RESNET::pre_process(vector<bm_image> &images) {
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();
  
  // 1.1 resize image
  int ret = 0;
  for(int i = 0; i < image_n; ++i)
  {
    bm_image image1 = images[i];
    bm_image image_aligned;
    bool need_copy = image1.width & (64-1);
    if(need_copy){
      int stride1[3], stride2[3];
      bm_image_get_stride(image1, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_bmContext->handle(), image1.height, image1.width,
          image1.image_format, image1.data_type, &image_aligned, stride2);

      bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;
      bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
    } else {
      image_aligned = image1;
    }
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height, m_net_w, m_net_h, &isAlignWidth);
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
      padding_attr.dst_crop_h = images[i].height*ratio;
      padding_attr.dst_crop_w = m_net_w;

      int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
      padding_attr.dst_crop_sty = ty1;
      padding_attr.dst_crop_stx = 0;
    } else{
      padding_attr.dst_crop_h = m_net_h;
      padding_attr.dst_crop_w = images[i].width*ratio;

      int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
    auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
        &padding_attr, &crop_rect);
#else
    auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i], &m_resized_imgs[i]);
#endif
    assert(BM_SUCCESS == ret);

#if DUMP_FILE
    cv::Mat resized_img;
    cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
    std::string fname = cv::format("resized_img_%d.jpg", i);
    cv::imwrite(fname, resized_img);
#endif
    // bm_image_destroy(image1);
    if(need_copy) bm_image_destroy(image_aligned);
  }

  // 1.2 converto
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);

  // 1.3 attach to tensor
  if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
  return 0;
}

float RESNET::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth) {
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w){
    *pIsAligWidth = true;
    ratio = r_w;
  }
  else{
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}
