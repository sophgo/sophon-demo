//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "directmhp.hpp"
#include <fstream>
#include <vector>
#include <string>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

DirectMHP::DirectMHP(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
  std::cout << "DirectMHP ctor .." << std::endl;
}

DirectMHP::~DirectMHP() {
  std::cout << "DirectMHP dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int DirectMHP::Init(float confThresh, float nmsThresh)
{
  m_confThreshold= confThresh;
  m_nmsThreshold = nmsThresh;


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
  min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

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
  input_scale = input_scale * 1.0 / 255.f;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = 0;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = 0;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = 0;

  return 0;
}

void DirectMHP::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}

int DirectMHP::batch_size() {
  return max_batch;
};

int DirectMHP::Detect(const std::vector<bm_image>& input_images, std::vector<DirectMHPBoxVec>& boxes)
{
  int ret = 0;
  //3. preprocess
  m_ts->save("directmhp preprocess", input_images.size());
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  m_ts->save("directmhp preprocess", input_images.size());

  //4. forward
  m_ts->save("directmhp inference", input_images.size());
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  m_ts->save("directmhp inference", input_images.size());

  //5. post process
  m_ts->save("directmhp postprocess", input_images.size());
  ret = post_process(input_images, boxes);
  CV_Assert(ret == 0);
  m_ts->save("directmhp postprocess", input_images.size());
  return ret;
}

int DirectMHP::pre_process(const std::vector<bm_image>& images){
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
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

float DirectMHP::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
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

int DirectMHP::post_process(const std::vector<bm_image> &images, std::vector<DirectMHPBoxVec>& detected_boxes)
{
  DirectMHPBoxVec directmhpbox_vec;
  std::vector<cv::Rect> bbox_vec;
  std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
  for(int i=0; i<output_num; i++){
      outputTensors[i] = m_bmNetwork->outputTensor(i);
  }
  
  for(int batch_idx = 0; batch_idx < images.size(); ++ batch_idx)
  {
    directmhpbox_vec.clear();
    auto& frame = images[batch_idx];
    int frame_width = frame.width;
    int frame_height = frame.height;

    int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(frame.width, frame.height, m_net_w, m_net_h, &isAlignWidth);
    if (isAlignWidth) {
      ty1 = (int)((m_net_h - (int)(frame_height*ratio)) / 2);
    }else{
      tx1 = (int)((m_net_w - (int)(frame_width*ratio)) / 2);
    }
#endif

    int min_idx = 0;
    int box_num = 0;

    auto out_tensor = outputTensors[min_idx];
    int nout = out_tensor->get_shape()->dims[min_dim-1];
    m_class_num = nout - 8;

    float* output_data = nullptr;
    std::vector<float> decoded_data;

    LOG_TS(m_ts, "post 1: get output");
    assert(box_num == 0 || box_num == out_tensor->get_shape()->dims[1]);
    box_num = out_tensor->get_shape()->dims[1];
    output_data = (float*)out_tensor->get_cpu_data() + batch_idx*box_num*nout;
    LOG_TS(m_ts, "post 1: get output");
    


    LOG_TS(m_ts, "post 2: filter boxes");
    int max_wh = 7680;
    bool agnostic = false;
    for (int i = 0; i < box_num; i++) {
      float* ptr = output_data+i*nout;
      float score = ptr[4];
      if (score > m_confThreshold) {
        int class_id = argmax(&ptr[5], m_class_num);
        float confidence = ptr[class_id + 5];
        if (confidence * score > m_confThreshold)
        {
            float centerX = ptr[0];
            float centerY = ptr[1];
            float width = ptr[2];
            float height = ptr[3];
            float pitch = (ptr[6]-0.5)*180;
            float yaw =  (ptr[7]-0.5)*360;
            float roll = (ptr[8]-0.5)*180;

            DirectMHPBox box;
            box.pitch = pitch;
            box.yaw = yaw;
            box.roll = roll;

            if (!agnostic)
              box.x = centerX - width / 2 + class_id * max_wh;
            else
              box.x = centerX - width / 2;
            if (box.x < 0) box.x = 0;
            if (!agnostic)
              box.y = centerY - height / 2 + class_id * max_wh;
            else
              box.y = centerY - height / 2;
            if (box.y < 0) box.y = 0;
            box.width = width;
            box.height = height;
            box.class_id = class_id;
            box.score = confidence * score;
            directmhpbox_vec.push_back(box);
        }

      }
    }
    LOG_TS(m_ts, "post 2: filter boxes");

    LOG_TS(m_ts, "post 3: nms");
    NMS(directmhpbox_vec, m_nmsThreshold);
    if (!agnostic)
      for (auto& box : directmhpbox_vec){
        box.x -= box.class_id * max_wh;
        box.y -= box.class_id * max_wh;
        box.x = (box.x - tx1) / ratio;
        box.y = (box.y - ty1) / ratio;
        box.width = (box.width) / ratio;
        box.height = (box.height) / ratio;
      }
    LOG_TS(m_ts, "post 3: nms");

        detected_boxes.push_back(directmhpbox_vec);
  }

  return 0;
}



int DirectMHP::argmax(float* data, int num){
  float max_value = 0.0;
  int max_index = 0;
  for(int i = 0; i < num; ++i) {
    float value = data[i];
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  return max_index;
}

void DirectMHP::NMS(DirectMHPBoxVec &dets, float nmsConfidence)
{
  int length = dets.size();
  int index = length - 1;

  std::sort(dets.begin(), dets.end(), [](const DirectMHPBox& a, const DirectMHPBox& b) {
      return a.score < b.score;
      });

  std::vector<float> areas(length);
  for (int i=0; i<length; i++)
  {
    areas[i] = dets[i].width * dets[i].height;
  }

  while (index  > 0)
  {
    int i = 0;
    while (i < index)
    {
      float left    = std::max(dets[index].x,   dets[i].x);
      float top     = std::max(dets[index].y,    dets[i].y);
      float right   = std::min(dets[index].x + dets[index].width,  dets[i].x + dets[i].width);
      float bottom  = std::min(dets[index].y + dets[index].height, dets[i].y + dets[i].height);
      float overlap = std::max(0.0f, right - left + 0.00001f) * std::max(0.0f, bottom - top + 0.00001f);
      if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence)
      {
        areas.erase(areas.begin() + i);
        dets.erase(dets.begin() + i);
        index --;
      }
      else
      {
        i++;
      }
    }
    index--;
  }
}

void DirectMHP::draw_bmcv(bm_handle_t &handle, int classId, float conf, float left, float top, float width, float height, float p, float r, float y, bm_image& frame, bool put_text_flag)   // Draw the predicted bounding box
{
  if (conf < 0.25) return;
  
  //Draw a rectangle displaying the bounding box
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(left, 0), frame.width);
  rect.start_y = MIN(MAX(top, 0), frame.height);
  rect.crop_w = MAX(MIN(width, frame.width - rect.start_x), 0);
  rect.crop_h = MAX(MIN(height, frame.height - rect.start_y), 0);
  
  float pitch = p * PI / 180;
  float yaw = -(y * PI / 180);
  float roll = r * PI / 180;


  float size = MAX(rect.crop_h, rect.crop_w) * 0.8;
  float centerX = rect.start_x + rect.crop_w / 2;
  float centerY = rect.start_y + rect.crop_h / 2;
  // X-Axis (pointing to right) drawn in red
  float x1 = size * (cos(yaw) * cos(roll)) + centerX;
  float y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + centerY;
  // Y-Axis (pointing to down) drawn in green
  float x2 = size * (-cos(yaw) * sin(roll)) + centerX;
  float y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + centerY;
  // Z-Axis (out of the screen) drawn in blue
  float x3 = size * (sin(yaw)) + centerX;
  float y3 = size * (-cos(yaw) * sin(pitch)) + centerY;
  // Plot head oritation line in black
  float scale_ratio = 2;
  float endx;
  float endy;
  if (centerX == x3) {
      endx = centerX;
      if (centerY < y3) {
          endy = centerY + (y3 - centerY) * scale_ratio;  
        } 
      else {
          endy = centerY - (centerY - y3) * scale_ratio;
        }
    } 
  else if (centerX > x3) {   
       endx = centerX - (centerX - x3) * scale_ratio;
       endy = centerY - (centerY - y3) * scale_ratio;
    } 
  else {
       endx = centerX + (x3 - centerX) * scale_ratio;
       endy = centerY + (y3 - centerY) * scale_ratio;
    }

  int thickness = 2;
  if(rect.crop_w <= thickness * 2 || rect.crop_h <= thickness * 2){
        std::cout << "width or height too small, this rect will not be drawed: " << 
              "[" << rect.start_x << ", "<< rect.start_y << ", " << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
  } else{
    bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, 255, 255, 255);
    
    endx = MAX(0, MIN(endx, frame.width));
    endy = MAX(0, MIN(endy, frame.height));
    bmcv_point_t start = {centerX, centerY};
    bmcv_point_t end = {endx, endy};
    bmcv_color_t color = {0, 255, 255};
    bmcv_image_draw_lines(handle, frame, &start, &end, 1, color, thickness);

    // X-Axis pointing to right. drawn in red
    x1 =  MAX(0, MIN(x1, frame.width));
    y1 =  MAX(0, MIN(y1, frame.height));
    bmcv_point_t start_x = {centerX, centerY};
    bmcv_point_t end_x = {x1, y1};
    bmcv_color_t color_x = {0, 0, 255};
    bmcv_image_draw_lines(handle, frame, &start_x, &end_x, 1, color_x, thickness);

    // Y-Axis pointing to down. drawn in green 
    x2 =  MAX(0, MIN(x2, frame.width));
    y2 =  MAX(0, MIN(y2, frame.height));
    bmcv_point_t start_y = {centerX, centerY};
    bmcv_point_t end_y = {x2, y2};
    bmcv_color_t color_y = {0, 255, 0};
    bmcv_image_draw_lines(handle, frame, &start_y, &end_y, 1, color_y, thickness);

    // Z-Axis (out of the screen) drawn in blue
    x3 =  MAX(0, MIN(x3, frame.width));
    y3 =  MAX(0, MIN(y3, frame.height));
    bmcv_point_t start_z = {centerX, centerY};
    bmcv_point_t end_z = {x3, y3};
    bmcv_color_t color_z = {255, 0, 0};
    bmcv_image_draw_lines(handle, frame, &start_z, &end_z, 1, color_z, thickness);


  } 
  if (put_text_flag){
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    bmcv_point_t org = {rect.start_x, rect.start_y};
    bmcv_color_t color = {255, 255, 255};
    float fontScale = 2; 
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;   
    }
  }
}
