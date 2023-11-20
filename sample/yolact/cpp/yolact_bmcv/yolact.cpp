//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolact.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

// #define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1
using namespace std;
using yolactBoxVec = std::vector<yolactBox>;

const std::vector<std::vector<int>> colors = {{255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0}, \
                {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255}, {0, 170, 255}, {0, 85, 255}, \
                {0, 0, 255}, {85, 0, 255}, {170, 0, 255}, {255, 0, 255}, {255, 0, 170}, {255, 0, 85}, {255, 0, 0},\
                {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

Yolact::Yolact(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
  std::cout << "Yolact ctor .." << std::endl;
}

Yolact::~Yolact() {
  std::cout << "Yolact dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int Yolact::Init(float confThresh, float nmsThresh, int keep_top_k, const std::string& coco_names_file)
{
  m_confThreshold= confThresh;
  m_nmsThreshold = nmsThresh;
  m_keep_top_k = keep_top_k;
  std::cout<<m_confThreshold<<std::endl;

  std::ifstream ifs(coco_names_file);
  if (ifs.is_open()) {
    std::string line;
    while(std::getline(ifs, line)) {
      line = line.substr(0, line.length() - 1);
      m_class_names.push_back(line);
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
  assert(output_num == 1 || output_num == 4);
  min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

  //4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  test.resize(max_batch);

  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for(int i=0; i<max_batch; i++){
    auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());

  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8){
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }

  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);

  assert(BM_SUCCESS == ret);

  // 5.convert
  vector<float> mean = {123.68, 116.17, 103.94};
  vector<float> std  = {58.40, 57.12, 57.38};
  vector<float> k_cofficient(3);
  vector<float> b_cofficient(3);

  for(int i=0;i<3;i++) {
    k_cofficient[i] = 1.0 / std[i];
    b_cofficient[i] = -mean[i]/std[i];
  }

  converto_attr.alpha_0 = k_cofficient[0];
  converto_attr.beta_0 = b_cofficient[0];
  converto_attr.alpha_1 = k_cofficient[1];
  converto_attr.beta_1 = b_cofficient[1];
  converto_attr.alpha_2 = k_cofficient[2];
  converto_attr.beta_2 = b_cofficient[2];

  converto_attr1.alpha_0 = 1;
  converto_attr1.beta_0 = 0;
  converto_attr1.alpha_1 = 1;
  converto_attr1.beta_1 = 0;
  converto_attr1.alpha_2 = 1;
  converto_attr1.beta_2 = 0;
  return 0;
}

bool Yolact::compareYolactBoxByScore(const yolactBox& box1, const yolactBox& box2) {
    return box1.score > box2.score;
}

void Yolact::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}

int Yolact::batch_size() {
  return max_batch;
};

int Yolact::Detect(const std::vector<bm_image>& input_images, std::vector<yolactBoxVec>& boxes)
{
  int ret = 0;
  //3. preprocess
  LOG_TS(m_ts, "yolact preprocess");
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "yolact preprocess");

  //4. forward
  LOG_TS(m_ts, "yolact inference");
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "yolact inference");

  //5. post process
  LOG_TS(m_ts, "yolact postprocess");
  ret = post_process(input_images, boxes);
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "yolact postprocess");

  return ret;
}

int Yolact::pre_process(const std::vector<bm_image>& images){
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
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number RGB_PLANAR FORMAT

  return 0;
}

int Yolact::post_process(const std::vector<bm_image> &images, std::vector<yolactBoxVec>& detected_boxes)
{
  yolactBoxVec yolobox_vec;
  std::vector<cv::Rect> bbox_vec;
  std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
  for(int i=0; i<output_num; i++){
      outputTensors[i] = m_bmNetwork->outputTensor(i);
  }

  for(int batch_idx = 0; batch_idx < images.size(); ++ batch_idx)
  {
    yolobox_vec.clear();
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

    // output_num:
    // output_num[0]: 1x19248x4    对应的Bbox坐标(x_left, y_left, w, h)
    // output_num[1]: 1x19248x81   对应的conf 
    // output_num[2]: 1x19248x32   对应的Mask
    // output_num[3]: 1x138x138x32 对应的prototype

    auto prototype_tensor = m_bmNetwork->outputTensor(3);
    auto bbox_tensor = m_bmNetwork->outputTensor(0);
    auto mask_tensor = m_bmNetwork->outputTensor(2);
    auto conf_tensor = m_bmNetwork->outputTensor(1);

    // #bbox:
    box_num = bbox_tensor->get_shape()->dims[1];
    // conf_tensor: [batch_size, #bbox, #class(with bg)]
    int nout = conf_tensor->get_shape()->dims[2];
    int m_class_num = nout - 1;
    
    // # prototype_elements
    int prototype_channel_number = prototype_tensor->get_shape()->dims[3];
    int prototype_eachchannel_step = prototype_tensor->get_shape()->dims[1] * prototype_tensor->get_shape()->dims[2];
    int prototype_step = prototype_eachchannel_step * prototype_channel_number;

    // # Mask_elements
    int mask_cof_step = mask_tensor->get_shape()->dims[2];
    assert(mask_cof_step == prototype_channel_number);

    // Step1: Decode conf_tensor and bbox_tensor
    std::vector<float> conf_data(box_num * m_class_num);
    std::vector<float> bbox_data(box_num * 4);

    float *conf_tensor_data = (float*)conf_tensor->get_cpu_data() + batch_idx*nout*box_num;
    float *bbox_tensor_data = (float*)bbox_tensor->get_cpu_data() + batch_idx*4*box_num;
    float *proto_tensor_data = (float*)prototype_tensor->get_cpu_data() + batch_idx*prototype_step;
    float *mask_tensor_data = (float*)mask_tensor->get_cpu_data() + batch_idx*mask_cof_step*box_num;
    
    // Decode:
    //(1) For each candidate box: get the argmax class id and conf
    //(2) If conf < min_conf: continue
    //(3) get bbox (predict_bbox + prios -> mid_point -> crop -> recover to original image size)
    //
    std::vector<std::vector<float>> priors;
    priors = make_prior(m_net_h, m_net_w);

    yolactBoxVec Sieved_Box;

    for(int anchor_id = 0; anchor_id<box_num; anchor_id++) {
          yolactBox yolact_box;
          std::vector<float> mask;

          float *conf_ptr = conf_tensor_data + anchor_id*nout;
          float *bbox_ptr = bbox_tensor_data + anchor_id*4;
          float *prototye_ptr = proto_tensor_data;
          float *mask_ptr = mask_tensor_data + anchor_id*mask_cof_step;

          int argmax_info_first;
          float argmax_info_second;
          std::tie(argmax_info_first, argmax_info_second) = argmax(conf_ptr, nout, m_confThreshold);
          if(argmax_info_first == 0) {
            continue;
          }
          mask.resize(prototype_eachchannel_step);

          // Use Yolact_box to get data
          yolact_box = decode_pos(bbox_ptr[0], bbox_ptr[1], bbox_ptr[2], bbox_ptr[3], priors[anchor_id], frame_width, frame_height);
          yolact_box.class_id = argmax_info_first;
          yolact_box.score = argmax_info_second;

          // Mask_Cof
          cv::Mat mask_cof(mask_cof_step, 1, CV_32FC1, mask_ptr);

          // Prototype
          cv::Mat prototype(prototype_eachchannel_step, prototype_channel_number, CV_32FC1, prototye_ptr);
          cv::Mat Mask_prototype = prototype * mask_cof;

          yolact_box.Mask_prototype = Mask_prototype;
          Sieved_Box.push_back(yolact_box);

    }


    // NMS
    LOG_TS(m_ts, "post 3: nms");
    #if USE_MULTICLASS_NMS
        std::vector<yolactBoxVec> class_vec(m_class_num);
        for (auto& box : Sieved_Box){
          class_vec[box.class_id -1].push_back(box);
        }
        for (auto& cls_box : class_vec){
          NMS(cls_box, m_nmsThreshold);
        }
        Sieved_Box.clear();
        for (auto& cls_box : class_vec){
          Sieved_Box.insert(Sieved_Box.end(), cls_box.begin(), cls_box.end());
        }
    #else
        NMS(Sieved_Box, m_nmsThreshold);
    #endif
    
    LOG_TS(m_ts, "post 3: nms");
    std::sort(Sieved_Box.begin(), Sieved_Box.end(), compareYolactBoxByScore);
    // if (Sieved_Box.size() > m_keep_top_k)
    //     Sieved_Box.resize(m_keep_top_k);
    std::cout << "Sieve Size is: " << Sieved_Box.size() << std::endl;
    detected_boxes.push_back(Sieved_Box);
    
  }
  return 0;
}

std::vector<std::vector<float>> Yolact::make_prior(int image_height, int image_width){
  std::vector<std::vector<float>> result_box;

  for(int k = 0; k < conv_ws.size(); k++) {
    int conv_w = conv_ws[k];
    int conv_h = conv_hs[k];
    double scale = scales[k];

    for(int i = 0;i < conv_h; i++) {
      for(int j =0;j < conv_w; j++) {
        double cx = (j + 0.5) / conv_w;
        double cy = (i + 0.5) / conv_h;

        for (const auto& ar : aspect_ratios) {
          std::vector<float> each_box;
          double aspect_ratio = std::sqrt(ar);

          double w = scale * aspect_ratio / image_width;
          double h = scale * aspect_ratio / image_height;
          h = w;

          each_box.push_back(cx);
          each_box.push_back(cy);
          each_box.push_back(w);
          each_box.push_back(h);
          result_box.push_back(each_box);
        }
      }
    }
  }

  return result_box;
}

yolactBox Yolact::decode_pos(float x, float y, float w, float h, std::vector<float> prior, int image_width, int image_height) {
  yolactBox yolact_box;

  // Decode
  float new_x = prior[0] + x * variances[0] * prior[3];
  float new_y = prior[1] + y * variances[0] * prior[2];
  float new_w = prior[3] * exp(w * variances[1]);
  float new_h = prior[2] * exp(h * variances[1]);
  
  // Center
  new_x = new_x - new_w / 2;
  new_y = new_y - new_h / 2;

  // Crop
  new_x = (new_x < 0) ? 0 : new_x;
  new_y = (new_y < 0) ? 0 : new_y;
  new_w = (new_w > 1) ? 1 : new_w;
  new_h = (new_h > 1) ? 1 : new_h;

  // Assign
  yolact_box.x = new_x * image_width;
  yolact_box.y = new_y * image_height;
  yolact_box.width = new_w * image_width + 1;
  yolact_box.height = new_h * image_height + 1;
  return yolact_box;
}

std::pair<int, float> Yolact::argmax(float* data, int num, float m_confThreshold) {
    float max_value = 0.0;
    int max_index = 0;
    for (int i = 1; i < num; i++) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }
    // return std::make_pair(max_index, max_value);
    if (max_value >= m_confThreshold) {
        return std::make_pair(max_index, max_value);
    } else {
        // return a NonMakeSense value, if max_value <= m_confThreshold
        return std::make_pair(0, 0.0);
    }
}


void Yolact::NMS(yolactBoxVec &dets, float nmsConfidence)
{
  int length = dets.size();
  int index = length - 1;

  std::sort(dets.begin(), dets.end(), [](const yolactBox& a, const yolactBox& b) {
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
      float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
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

void Yolact::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)   // Draw the predicted bounding box
{
// Draw a rectangle displaying the bounding box
cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

// Get the label for the class name and its confidence
std::string label = cv::format("%.2f", conf);
if ((int)m_class_names.size() >= m_class_num) {
    label = this->m_class_names[classId - 1] + ":" + label;
} else {
    label = std::to_string(classId - 1) + ":" + label;
}

// Display the label at the top of the bounding box
int fontFace = cv::FONT_HERSHEY_SIMPLEX;
double fontScale = 1;
int thickness = 2;
int baseline;
cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
top = std::max(top-10, labelSize.height-10);
cv::putText(frame, label, cv::Point(left, top), fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);

}

void Yolact::drawMask(int ClassId, int left, int top, int right, int bottom, int width, int height, cv::Mat Mask_prototype, cv::Mat& frame) {
  int colors_num = colors.size();

  // Normalize 1 / (1 + exp(-mask))
  cv::Mat expMasks;
  cv::exp(-Mask_prototype, expMasks);
  cv::Mat Normalized_Mask = 1.0 / (1.0 + expMasks);

  // Reshape to 138x138
  cv::Mat reshapedPrototype = Normalized_Mask.reshape(0, 138);

  // Resize to img_w, img_h
  cv::Mat resizedImage;
  cv::Size targetSize(height, width);
  cv::resize(reshapedPrototype, resizedImage, targetSize, 0, 0, cv::INTER_LINEAR);


  // Crop area out of bbox
  cv::Mat Crop_Mask;
  std::vector<int> boxes_mask = {left, top, right, bottom};
  Crop_Mask = crop_mask(resizedImage, boxes_mask);

  // Bit Process Crop_Mask = Crop_Mask > 0.5;
  cv::Mat BiCrop_Mask;
  double threshold_value = 0.5; // threshold = 0.5
  double max_value = 255.0;       // pix > 0.5 -> 1
  double threshold_type = cv::THRESH_BINARY;

  cv::threshold(Crop_Mask, BiCrop_Mask, threshold_value, max_value, threshold_type);

  // Add Weights
  BiCrop_Mask.convertTo(BiCrop_Mask, frame.type());
  
  cv::Scalar transparentColor(colors[ClassId % colors_num][0], colors[ClassId % colors_num][1], colors[ClassId % colors_num][2]); // BGR颜色通道

  // Traverse Image and Mask
  for (int i = 0; i < frame.rows; ++i) {
      for (int j = 0; j < frame.cols; ++j) {
          if (BiCrop_Mask.at<uchar>(i, j) > 0) {
              cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
              for (int c = 0; c < frame.channels(); ++c) {
                  pixel[c] = pixel[c] * 0.5 + transparentColor[c] * 0.5;
              }
              frame.at<cv::Vec3b>(i, j) = pixel;
          }
      }
  }

}


void Yolact::draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int width, int height, bm_image& frame, bool put_text_flag)   // Draw the predicted bounding box
{
  if (conf < 0.25) return;
  int colors_num = colors.size();
  //Draw a rectangle displaying the bounding box
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(left, 0), frame.width);
  rect.start_y = MIN(MAX(top, 0), frame.height);
  rect.crop_w = MAX(MIN(width, frame.width - left), 0);
  rect.crop_h = MAX(MIN(height, frame.height - top), 0);

  bmcv_image_draw_rectangle(handle, frame, 1, &rect, 3, colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
  if (put_text_flag){
    //Get the label for the class name and its confidence
    std::string label = m_class_names[classId - 1] + ":" + cv::format("%.2f", conf);
    bmcv_point_t org = {left, top-10};
    bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]};
    int thickness = 2;
    float fontScale = 1; 
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;   
    }
  }
}


std::pair<int, int> Yolact::sanitize_coordinates(int _x1, int _x2, int img_size, int padding) {
    // get x1 , x2
    int x1 = std::min(_x1, _x2);
    int x2 = std::max(_x1, _x2);

    // restrict x1/x2 within [padding, img_size-padding]
    x1 = std::min(std::max(x1 - padding, 0), 1000000);
    x2 = std::min(std::max(x2 + padding, 0), img_size);

    return std::make_pair(x1, x2);
}

cv::Mat Yolact::crop_mask(const cv::Mat& masks, std::vector<int> boxes, int padding) {
    int h = masks.rows;
    int w = masks.cols;

    int x1, x2, y1, y2;
    
    x1 = boxes[0] - padding > 0 ? boxes[0] - padding : 0.0;
    x2 = boxes[2] + padding < w ? boxes[2] + padding : w;
    y1 = boxes[1] - padding > 0 ? boxes[1] - padding : 0.0;
    y2 = boxes[3] + padding < h ? boxes[3] + padding : h;

    cv::Mat croped_mask(h, w, CV_8UC1, cv::Scalar(0));

    // Compute crop masks
    for (int i = y1; i < y2; i++) {
        for (int j = x1; j < x2; j++) {
            croped_mask.at<uchar>(i, j) = 255; 
        }
    }

    cv::Mat cropped_masks;
    cv::Mat mask_8UC1;
    masks.convertTo(mask_8UC1, CV_8UC1);

    cv::bitwise_and(mask_8UC1, croped_mask, cropped_masks); 
    return cropped_masks;
}