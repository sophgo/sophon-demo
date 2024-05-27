//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "real_esrgan.hpp"
#include <fstream>
#include <vector>
#include <string>
#include<cmath>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1
using namespace std;

Real_ESRGAN::Real_ESRGAN(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
  std::cout << "Real_ESRGAN ctor .." << std::endl;
}

Real_ESRGAN::~Real_ESRGAN() {
  std::cout << "Real_ESRGAN dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int Real_ESRGAN::Init()
{
  
  //1. get network
  m_bmNetwork = m_bmContext->network(0);
  
  //2. get input
  max_batch = m_bmNetwork->maxBatch();
  auto tensor = m_bmNetwork->inputTensor(0);
  m_net_h = tensor->get_shape()->dims[2];
  m_net_w = tensor->get_shape()->dims[3];
  cout<<"m_net_h:"<<m_net_h<<"    "<<"m_net_w "<<m_net_w<<endl;
  //3. get output
  output_num = m_bmNetwork->outputTensorNum();
  cout<<"output_num:"<<output_num<<endl;
  assert(output_num == 1 || output_num == 3);
  auto output_tensor=m_bmNetwork->outputTensor(0);
  min_dim = output_tensor->get_shape()->num_dims;
  cout<<"min_dim:"<<min_dim<<endl;
  int m_net_out_put_h = output_tensor->get_shape()->dims[2];
  int m_net_out_put_w = output_tensor->get_shape()->dims[3];
  cout<<"m_net_out_put_h:"<<m_net_out_put_h<<"    "<<"m_net_out_put_w "<<m_net_out_put_w<<endl;
  
  upsample_scale = m_net_out_put_w/m_net_w;
  //4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  cout<<"aligned_net_w:"<<aligned_net_w<<endl;
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
  else if (tensor->get_dtype() == BM_UINT8){
    img_dtype = DATA_TYPE_EXT_1N_BYTE;
  }
  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
  assert(BM_SUCCESS == ret);

  // 5.converto
  float input_scale = tensor->get_scale();
  cout<<"input_scale:"<<input_scale<<endl;
  input_scale = input_scale * 1.0 / 255.f;
  cout<<"input_scale:"<<input_scale<<endl;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = 0;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = 0;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = 0;

  return 0;
}

void Real_ESRGAN::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}

int Real_ESRGAN::batch_size() {
  return max_batch;
};

int Real_ESRGAN::Detect(const std::vector<bm_image>& input_images,vector<cv::Mat>& output_images)
{
  int ret = 0;
  //3. preprocess
  m_ts->save("Real_ESRGAN preprocess", input_images.size());
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  m_ts->save("Real_ESRGAN preprocess", input_images.size());

  //4. forward
  m_ts->save("Real_ESRGAN inference", input_images.size());
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  m_ts->save("Real_ESRGAN inference", input_images.size());

  //5. post process
  m_ts->save("Real_ESRGAN postprocess", input_images.size());
  ret = post_process(input_images,output_images);
  CV_Assert(ret == 0);
  m_ts->save("Real_ESRGAN postprocess", input_images.size());

  return ret;
}

int Real_ESRGAN::pre_process(const std::vector<bm_image>& images){
  auto input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();
  //1. resize image
  int ret = 0;
  for(int i = 0; i < image_n; ++i) {
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
    }else{
      padding_attr.dst_crop_h = m_net_h;
      padding_attr.dst_crop_w = images[i].width*ratio;

      int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
    auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
        &padding_attr, &crop_rect, BMCV_INTER_NEAREST);
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
    if(need_copy) bm_image_destroy(image_aligned);
  }
  
  //2. converto
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);


  //3. attach to tensor
  if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
  return 0;
}

float Real_ESRGAN::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
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


int Real_ESRGAN::post_process(const std::vector<bm_image>& images, std::vector<cv::Mat>& output_images) 
{
 
  output_images.clear();
  
  std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
  for(int i=0; i<output_num; i++){
    outputTensors[i] = m_bmNetwork->outputTensor(i);
  }
  for(int batch_idx = 0; batch_idx < images.size(); ++ batch_idx)
  {
    auto& frame = images[batch_idx];
    int frame_width = frame.width;
    int frame_height = frame.height;


  float tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
  bool is_align_width = false;
  float r_w = (float)m_net_w / frame.width;
  float r_h = (float)m_net_h / frame.height;
  if (r_h > r_w) {
      int th = (int)(r_w * frame.height);
      ty1 = (m_net_h - th) / 2.0f;
  } else {
      int tw = (int)(r_h * frame.width);
      tx1 = (m_net_w - tw) / 2.0f;
  }
#endif

  int min_idx = 0;
  for(int i=0; i<output_num; i++)
  {
    auto output_shape = m_bmNetwork->outputTensor(i)->get_shape();
    auto output_dims = output_shape->num_dims;
    assert(output_dims == 4);
    if(min_dim>output_dims)
    {
      min_idx = i;
      min_dim = output_dims;
    }
  }
  auto output_tensor = outputTensors[min_idx];
  const bm_shape_t* shape = output_tensor->get_shape();
  if (!shape) {
    std::cout << "Shape is null" << std::endl;
    return -1;
  }
  int height = shape->dims[2];
  int width = shape->dims[3];
  int channel  = shape->dims[1];

  float* cpu_data = output_tensor->get_cpu_data()+batch_idx*height*width*channel;

  cv::Mat channelB(height, width, CV_8UC1);
  cv::Mat channelG(height, width, CV_8UC1);
  cv::Mat channelR(height, width, CV_8UC1);
  uchar* ptr_channelR = channelR.data;
  uchar* ptr_channelG = channelG.data;
  uchar* ptr_channelB = channelB.data;
  int area = height * width;

  cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
  for (int y = range.start; y < range.end; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      ptr_channelR[idx] = cv::saturate_cast<uchar>(cpu_data[idx] * 255.0f);
      ptr_channelG[idx] = cv::saturate_cast<uchar>(cpu_data[idx + area] * 255.0f);
      ptr_channelB[idx] = cv::saturate_cast<uchar>(cpu_data[idx + 2 * area] * 255.0f);
    }
  }});
  
  std::vector<cv::Mat> bgr_channels = {channelB, channelG, channelR};
  cv::Mat output_image;
  cv::merge(bgr_channels, output_image);
  
  // 根据tx1和ty1裁剪图像
  if (tx1 != 0) {
    int tx = tx1 * upsample_scale; // 调整比例
    output_image = output_image(cv::Rect(tx, 0, output_image.cols - 2*tx, output_image.rows));
  } else if (ty1 != 0) {
    int ty = ty1 * upsample_scale; // 调整比例
    output_image = output_image(cv::Rect(0, ty, output_image.cols, output_image.rows - 2*ty));
  }
  output_images.push_back(output_image);
  }
  return 0;
}

