//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dfine.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

const std::vector<std::vector<int>> colors = {{255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0}, \
                {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255}, {0, 170, 255}, {0, 85, 255}, \
                {0, 0, 255}, {85, 0, 255}, {170, 0, 255}, {255, 0, 255}, {255, 0, 170}, {255, 0, 85}, {255, 0, 0},\
                {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

DFine::DFine(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
  std::cout << "DFine ctor .." << std::endl;
  m_orig_size_mem_allocated = false;
}

DFine::~DFine() {
  std::cout << "DFine dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
  if (m_orig_size_mem_allocated) {
    bm_free_device(m_bmContext->handle(), m_orig_size_mem);
  }
}

int DFine::Init(float confThresh, const std::string& coco_names_file)
{
  m_confThreshold = confThresh;

  // load class names (optional, used for drawing text)
  if (!coco_names_file.empty()) {
    std::ifstream ifs(coco_names_file);
    if (ifs.is_open()) {
      std::string line;
      while(std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        m_class_names.push_back(line);
      }
    }
  }

  //1. get network
  m_bmNetwork = m_bmContext->network(0);

  //2. get input
  max_batch = m_bmNetwork->maxBatch();
  auto tensor = m_bmNetwork->inputTensor(0);
  m_net_h = tensor->get_shape()->dims[2];
  m_net_w = tensor->get_shape()->dims[3];

  //3. get output
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num == 3);

  // identify output indices by name
  m_labels_idx = m_boxes_idx = m_scores_idx = -1;
  for(int i = 0; i < output_num; i++) {
    std::string name = m_bmNetwork->m_netinfo->output_names[i];
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("label") != std::string::npos || lower.find("logit") != std::string::npos)
      m_labels_idx = i;
    else if (lower.find("box") != std::string::npos)
      m_boxes_idx = i;
    else if (lower.find("score") != std::string::npos)
      m_scores_idx = i;
  }
  // fallback: identify by shape (boxes has 3 dims; among 2D, first=labels, second=scores)
  if (m_boxes_idx == -1 || m_labels_idx == -1 || m_scores_idx == -1) {
    for (int i = 0; i < output_num; i++) {
      auto shape = m_bmNetwork->outputTensor(i)->get_shape();
      if (shape->num_dims == 3) m_boxes_idx = i;
    }
    std::vector<int> two_d;
    for (int i = 0; i < output_num; i++) {
      if (i != m_boxes_idx) two_d.push_back(i);
    }
    if (m_labels_idx == -1 && two_d.size() >= 1) m_labels_idx = two_d[0];
    if (m_scores_idx == -1 && two_d.size() >= 2) m_scores_idx = two_d[1];
  }
  assert(m_labels_idx >= 0 && m_boxes_idx >= 0 && m_scores_idx >= 0);

  //4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for(int i=0; i<max_batch; i++){
    auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
    if (ret != BM_SUCCESS) {
      throw std::runtime_error("bm_image_create failed");
    }
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());

  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8){
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
  if (ret != BM_SUCCESS) {
    throw std::runtime_error("bm_image_create_batch failed");
  }

  // 5.converto: alpha = input_scale / 255, beta = 0 (D-FINE normalizes by /255 only)
  float input_scale = tensor->get_scale();
  input_scale = input_scale * 1.0f / 255.f;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = 0;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = 0;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = 0;

  // 6. allocate orig_size device memory ([max_batch, 2] int32, value=[net_h, net_w] per image)
  int orig_size_count = max_batch * 2;
  std::vector<int32_t> orig_size_data(orig_size_count);
  for (int i = 0; i < max_batch; i++) {
    orig_size_data[i * 2] = m_net_h;
    orig_size_data[i * 2 + 1] = m_net_w;
  }
  ret = bm_malloc_device_byte(m_bmContext->handle(), &m_orig_size_mem, orig_size_count * sizeof(int32_t));
  if (ret != BM_SUCCESS) {
    throw std::runtime_error("bm_malloc_device_byte for orig_size failed");
  }
  ret = bm_memcpy_s2d(m_bmContext->handle(), m_orig_size_mem, orig_size_data.data());
  if (ret != BM_SUCCESS) {
    throw std::runtime_error("bm_memcpy_s2d for orig_size failed");
  }
  m_orig_size_mem_allocated = true;

  return 0;
}

void DFine::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}

int DFine::batch_size() {
  return max_batch;
};

int DFine::Detect(const std::vector<bm_image>& input_images, std::vector<DFineBoxVec>& boxes)
{
  int ret = 0;
  //3. preprocess
  m_ts->save("dfine preprocess", input_images.size());
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  m_ts->save("dfine preprocess", input_images.size());

  //4. forward
  m_ts->save("dfine inference", input_images.size());
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  m_ts->save("dfine inference", input_images.size());

  //5. post process
  m_ts->save("dfine postprocess", input_images.size());
  ret = post_process(input_images, boxes);
  CV_Assert(ret == 0);
  m_ts->save("dfine postprocess", input_images.size());
  return ret;
}

int DFine::pre_process(const std::vector<bm_image>& images){
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();
  int ret = 0;
  //1. resize image with letterbox (black padding)
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

    // letterbox: r = min(net_w/img_w, net_h/img_h), centered, black padding
    float r_w = (float)m_net_w / image1.width;
    float r_h = (float)m_net_h / image1.height;
    float ratio = std::min(r_w, r_h);
    int tw = (int)roundf(ratio * image1.width);
    int th = (int)roundf(ratio * image1.height);
    float pad_w = (float)(m_net_w - tw) / 2.0f;
    float pad_h = (float)(m_net_h - th) / 2.0f;

    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_stx = (int)roundf(pad_w - 0.1f);
    padding_attr.dst_crop_sty = (int)roundf(pad_h - 0.1f);
    padding_attr.dst_crop_w = tw;
    padding_attr.dst_crop_h = th;
    padding_attr.padding_r = 0;
    padding_attr.padding_g = 0;
    padding_attr.padding_b = 0;
    padding_attr.if_memset = 1;

    bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
    ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
        &padding_attr, &crop_rect, BMCV_INTER_LINEAR);
    if (ret != BM_SUCCESS) {
      throw std::runtime_error("bmcv_image_vpp_convert_padding failed");
    }

    if(need_copy) bm_image_destroy(image_aligned);
  }

  //2. converto (alpha = input_scale/255, beta = 0)
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);

  //3. attach image data to input tensor 0
  int batch_num = image_n;
  if(batch_num != max_batch) batch_num = m_bmNetwork->get_nearest_batch(batch_num);
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(batch_num, m_converto_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, batch_num);  // set real batch number

  //4. set orig_size input tensor 1 ([batch, 2] int32, value=[net_h, net_w])
  auto orig_size_tensor = m_bmNetwork->inputTensor(1);
  orig_size_tensor->set_device_mem(&m_orig_size_mem);
  orig_size_tensor->set_shape_by_dim(0, batch_num);

  return 0;
}

int DFine::post_process(const std::vector<bm_image> &images, std::vector<DFineBoxVec>& detected_boxes)
{
  auto labels_tensor = m_bmNetwork->outputTensor(m_labels_idx);
  auto boxes_tensor = m_bmNetwork->outputTensor(m_boxes_idx);
  auto scores_tensor = m_bmNetwork->outputTensor(m_scores_idx);

  float* labels_data = labels_tensor->get_cpu_data();
  float* boxes_data = boxes_tensor->get_cpu_data();
  float* scores_data = scores_tensor->get_cpu_data();

  // get num_queries from boxes shape [batch, N, 4]
  auto boxes_shape = boxes_tensor->get_shape();
  int num_queries = boxes_shape->dims[1];

  for(int batch_idx = 0; batch_idx < images.size(); ++batch_idx)
  {
    DFineBoxVec box_vec;
    auto& frame = images[batch_idx];

    // recompute ratio and padding (must match pre_process)
    float r_w = (float)m_net_w / frame.width;
    float r_h = (float)m_net_h / frame.height;
    float ratio = std::min(r_w, r_h);
    int tw = (int)roundf(ratio * frame.width);
    int th = (int)roundf(ratio * frame.height);
    float pad_w = (float)(m_net_w - tw) / 2.0f;
    float pad_h = (float)(m_net_h - th) / 2.0f;

    float* labels_ptr = labels_data + batch_idx * num_queries;
    float* boxes_ptr = boxes_data + batch_idx * num_queries * 4;
    float* scores_ptr = scores_data + batch_idx * num_queries;

    // filter by score threshold and map coordinates back (no NMS — model handles it internally)
    for (int q = 0; q < num_queries; q++) {
      float score = scores_ptr[q];
      if (score > m_confThreshold) {
        DFineBox box;
        box.class_id = (int)labels_ptr[q];
        box.score = score;
        float bx1 = boxes_ptr[q * 4 + 0];
        float by1 = boxes_ptr[q * 4 + 1];
        float bx2 = boxes_ptr[q * 4 + 2];
        float by2 = boxes_ptr[q * 4 + 3];
        box.x1 = (int)((bx1 - pad_w) / ratio);
        box.y1 = (int)((by1 - pad_h) / ratio);
        box.x2 = (int)((bx2 - pad_w) / ratio);
        box.y2 = (int)((by2 - pad_h) / ratio);
        box_vec.push_back(box);
      }
    }

    detected_boxes.push_back(box_vec);
  }

  return 0;
}

void DFine::draw_bmcv(bm_handle_t &handle, int classId, float conf, int x1, int y1, int x2, int y2, bm_image& frame)
{
  int colors_num = colors.size();
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(x1, 0), frame.width);
  rect.start_y = MIN(MAX(y1, 0), frame.height);
  rect.crop_w = MAX(MIN(x2 - x1, frame.width - rect.start_x), 0);
  rect.crop_h = MAX(MIN(y2 - y1, frame.height - rect.start_y), 0);
  int thickness = 2;
  if(rect.crop_w <= thickness * 2 || rect.crop_h <= thickness * 2){
    std::cout << "width or height too small, this rect will not be drawed: " <<
      "[" << rect.start_x << ", "<< rect.start_y << ", " << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
  } else{
    bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
  }
}
