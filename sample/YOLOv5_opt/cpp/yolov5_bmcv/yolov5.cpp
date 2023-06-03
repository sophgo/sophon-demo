//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov5.hpp"
#include <fstream>
#include <vector>
#include <string>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0

const std::vector<std::vector<u_char>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloV5::YoloV5(std::shared_ptr<BMNNContext> context) : m_bmContext(context) {
  std::cout << "YoloV5 ctor .." << std::endl;
}

YoloV5::~YoloV5() {
  std::cout << "YoloV5 dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for (int i = 0; i < max_batch; i++) {
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
    bm_free_device(m_bmContext->handle(), out_dev_mem[i]);
    bm_free_device(m_bmContext->handle(), detect_num_mem[i]);
    delete[] output_tensor[i];
  }
}

int YoloV5::Init(float confThresh, float nmsThresh, const std::string& tpu_kernel_module_path, const std::string& coco_names_file) {
  m_confThreshold = confThresh;
  m_nmsThreshold = nmsThresh;
  std::ifstream ifs(coco_names_file);
  if (ifs.is_open()) {
    std::string line;
    while (std::getline(ifs, line)) {
      line = line.substr(0, line.length() - 1);
      m_class_names.push_back(line);
    }
  }

  // 1. get network
  m_bmNetwork = m_bmContext->network(0);

  // 2. get input
  max_batch = m_bmNetwork->maxBatch();
  auto tensor = m_bmNetwork->inputTensor(0);
  m_net_h = tensor->get_shape()->dims[2];
  m_net_w = tensor->get_shape()->dims[3];

  // 3. get output
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num == 1 || output_num == 3);
  min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

  // 4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for (int i = 0; i < max_batch; i++) {
    auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                               &m_resized_imgs[i], strides);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8) {
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype,
                                   m_converto_imgs.data(), max_batch);
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

  // 6.tpukernel postprocess
  if(m_net_h < 32 || m_net_w < 32 || m_net_h % 32 != 0 || m_net_w % 32 != 0 || m_net_h > 2048 || m_net_w > 2048 || (m_net_h + m_net_w) > 3072){
    std::cerr << "Unsupported shape for tpukernel postprocession!" << std::endl;
    exit(1);
  }
  tpu_kernel_module_t tpu_module;
  tpu_module = tpu_kernel_load_module_file(m_bmContext->handle(), tpu_kernel_module_path.c_str()); 
  if(output_num == 3){
    func_id = tpu_kernel_get_function(m_bmContext->handle(), tpu_module, "tpu_kernel_api_yolov5_detect_out");
    std::cout << "Using tpu_kernel_api_yolov5_detect_out, kernel funtion id: " << func_id << std::endl;
  }else if(output_num == 1){
    func_id = tpu_kernel_get_function(m_bmContext->handle(), tpu_module, "tpu_kernel_api_yolov5_out_without_decode");
    std::cout << "Using tpu_kernel_api_yolov5_out_without_decode, kernel funtion id: " << func_id << std::endl;
  }else{
    std::cerr << "Unsupport output format!" << std::endl;
    exit(1);
  }
  
  int out_len_max = 25200 * 7;
  int input_num = m_bmNetwork->outputTensorNum();
  int batch_num = 1; // tpu_kernel_api_yolov5_detect_out now only support batchsize=1.
  
  // allocate device memory
  bm_handle_t handle = m_bmContext->handle();
  bm_device_mem_t in_dev_mem[input_num];
  for(int i = 0; i < input_num; i++)
    in_dev_mem[i] = *m_bmNetwork->outputTensor(i)->get_device_mem();
  
  for(int i = 0; i < max_batch; i++){
    output_tensor[i] = new float[out_len_max];
    for (int j = 0; j < input_num; j++) {
      api[i].bottom_addr[j] = bm_mem_get_device_addr(in_dev_mem[j]) + i * in_dev_mem[j].size / max_batch;
      api_v2[i].bottom_addr = bm_mem_get_device_addr(in_dev_mem[j]) + i * in_dev_mem[j].size / max_batch;
    }
    ret = bm_malloc_device_byte(handle, &out_dev_mem[i], out_len_max * sizeof(float));
    assert(BM_SUCCESS == ret);
    ret = bm_malloc_device_byte(handle, &detect_num_mem[i], batch_num * sizeof(int32_t));
    assert(BM_SUCCESS == ret);

    /*initialize api for tpu_kernel_api_yolov5_out*/
    api[i].top_addr = bm_mem_get_device_addr(out_dev_mem[i]);
    api[i].detected_num_addr = bm_mem_get_device_addr(detect_num_mem[i]);
    // config
    api[i].input_num = input_num;
    api[i].batch_num = batch_num;
    for (int j = 0; j < input_num; ++j) {
      api[i].hw_shape[j][0] = m_bmNetwork->outputTensor(j)->get_shape()->dims[2];
      api[i].hw_shape[j][1] = m_bmNetwork->outputTensor(j)->get_shape()->dims[3];
    }
    api[i].num_classes = m_class_num;
    api[i].num_boxes = anchors[0].size();
    api[i].keep_top_k = 200;
    api[i].nms_threshold = MAX(0.1, m_nmsThreshold);
    api[i].confidence_threshold = MAX(0.1, m_confThreshold);
    auto it=api[i].bias;
    for (const auto& subvector2 : anchors) {
      for (const auto& subvector1 : subvector2) {
        it = copy(subvector1.begin(), subvector1.end(), it);
      }
    }
    for (int j = 0; j < input_num; j++) 
      api[i].anchor_scale[j] = m_net_h / m_bmNetwork->outputTensor(j)->get_shape()->dims[2];
    api[i].clip_box = 1;

    /*initialize api_v2 for tpu_kernel_api_yolov5_out_without_decode*/
    api_v2[i].top_addr = bm_mem_get_device_addr(out_dev_mem[i]);
    api_v2[i].detected_num_addr = bm_mem_get_device_addr(detect_num_mem[i]);
    api_v2[i].input_shape[0] = 1; //only support batchsize=1
    api_v2[i].input_shape[1] = m_bmNetwork->outputTensor(0)->get_shape()->dims[1];
    api_v2[i].input_shape[2] = m_bmNetwork->outputTensor(0)->get_shape()->dims[2];
    api_v2[i].keep_top_k = 200;
    api_v2[i].nms_threshold = MAX(0.1, m_nmsThreshold);
    api_v2[i].confidence_threshold = MAX(0.1, m_confThreshold);
    api_v2[i].agnostic_nms = 0;
    api_v2[i].max_hw = MAX(m_net_h, m_net_w);
  }

  return 0;
}

void YoloV5::enableProfile(TimeStamp* ts) {
  m_ts = ts;
}

int YoloV5::batch_size() {
  return max_batch;
};

int YoloV5::Detect(const std::vector<bm_image>& input_images, std::vector<YoloV5BoxVec>& boxes) {
  int ret = 0;
  // 3. preprocess
  LOG_TS(m_ts, "yolov5 preprocess");
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "yolov5 preprocess");

  // 4. forward
  LOG_TS(m_ts, "yolov5 inference");
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "yolov5 inference");

  // 5. post process
  LOG_TS(m_ts, "yolov5 postprocess");
  ret = post_process_tpu_kernel(input_images, boxes);
  CV_Assert(ret == 0);
  LOG_TS(m_ts, "yolov5 postprocess");
  return ret;
}

int YoloV5::pre_process(const std::vector<bm_image>& images) {
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();
  // 1. resize image
  int ret = 0;
  for (int i = 0; i < image_n; ++i) {
    bm_image image1 = images[i];
    bm_image image_aligned;
    bool need_copy = image1.width & (64 - 1);
    if (need_copy) {
      int stride1[3], stride2[3];
      bm_image_get_stride(image1, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_bmContext->handle(), image1.height, image1.width, image1.image_format, image1.data_type,
                      &image_aligned, stride2);

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
      padding_attr.dst_crop_h = images[i].height * ratio;
      padding_attr.dst_crop_w = m_net_w;

      int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
      padding_attr.dst_crop_sty = ty1;
      padding_attr.dst_crop_stx = 0;
    } else {
      padding_attr.dst_crop_h = m_net_h;
      padding_attr.dst_crop_w = images[i].width * ratio;

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
    if (need_copy)
      bm_image_destroy(image_aligned);
  }

  // 2. converto
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(),
                              m_converto_imgs.data());
  CV_Assert(ret == 0);

  // 3. attach to tensor
  if (image_n != max_batch)
    image_n = m_bmNetwork->get_nearest_batch(image_n);
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
  return 0;
}

float YoloV5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w) {
    *pIsAligWidth = true;
    ratio = r_w;
  } else {
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

void YoloV5::drawPred(int classId,
                      float conf,
                      int left,
                      int top,
                      int right,
                      int bottom,
                      cv::Mat& frame)  // Draw the predicted bounding box
{
  // Draw a rectangle displaying the bounding box
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

  // Get the label for the class name and its confidence
  std::string label = cv::format("%.2f", conf);
  if ((int)m_class_names.size() >= m_class_num) {
    label = this->m_class_names[classId] + ":" + label;
  } else {
    label = std::to_string(classId) + ":" + label;
  }

  // Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = std::max(top, labelSize.height);
  cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
}

void YoloV5::draw_bmcv(bm_handle_t& handle,
                       int classId,
                       float conf,
                       int left,
                       int top,
                       int width,
                       int height,
                       bm_image& frame,
                       bool put_text_flag)  // Draw the predicted bounding box
{
  if (conf < 0.25)
    return;
  int colors_num = colors.size();
  // Draw a rectangle displaying the bounding box
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(left, 0), frame.width);
  rect.start_y = MIN(MAX(top, 0), frame.height);
  rect.crop_w = MAX(MIN(width, frame.width - left), 0);
  rect.crop_h = MAX(MIN(height, frame.height - top), 0);
  bmcv_image_draw_rectangle(handle, frame, 1, &rect, 3, colors[classId % colors_num][0],
                            colors[classId % colors_num][1], colors[classId % colors_num][2]);
  if (put_text_flag) {
    // Get the label for the class name and its confidence
    std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
    // Display the label at the top of the bounding box
    bmcv_point_t org = {left, top};
    bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1],
                          colors[classId % colors_num][2]};
    int thickness = 2;
    float fontScale = 2;
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;
    }
  }
}

int YoloV5::post_process_tpu_kernel(const std::vector<bm_image>& images, std::vector<YoloV5BoxVec>& detected_boxes) {
  for(int i = 0; i < max_batch; i++){
    int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height, m_net_w, m_net_h, &isAlignWidth);
    if (isAlignWidth) {
      ty1 = (int)((m_net_h - (int)(images[i].height * ratio)) / 2);
    } else {
      tx1 = (int)((m_net_w - (int)(images[i].width * ratio)) / 2);
  }
#endif
    if(output_num == 3){
      tpu_kernel_launch(m_bmContext->handle(), func_id, &api[i], sizeof(api[i]));
    }else if(output_num == 1){
      tpu_kernel_launch(m_bmContext->handle(), func_id, &api_v2[i], sizeof(api_v2[i]));
    }
    bm_thread_sync(m_bmContext->handle());
    bm_memcpy_d2s_partial_offset(m_bmContext->handle(), (void*)(detect_num + i), detect_num_mem[i],1 * sizeof(int32_t), 0); // only support batchsize=1
    if(detect_num[i] > 0){
      bm_memcpy_d2s_partial_offset(m_bmContext->handle(), (void*)output_tensor[i], out_dev_mem[i], detect_num[i] * 7 * sizeof(float),
                                0);  // 25200*7
    }
    YoloV5BoxVec vec;
    detected_boxes.push_back(vec);
    for (int bid = 0; bid < detect_num[i]; bid++) {
      YoloV5Box temp_bbox;
      temp_bbox.class_id = *(output_tensor[i] + 7 * bid + 1);
      if (temp_bbox.class_id == -1) {
        continue;
      }
      temp_bbox.score = *(output_tensor[i] + 7 * bid + 2);
      float centerX = (*(output_tensor[i] + 7 * bid + 3) + 1 - tx1) / ratio - 1;
      float centerY = (*(output_tensor[i] + 7 * bid + 4) + 1 - ty1) / ratio - 1;
      temp_bbox.width = (*(output_tensor[i] + 7 * bid + 5) + 0.5) / ratio;
      temp_bbox.height = (*(output_tensor[i] + 7 * bid + 6) + 0.5) / ratio;

      temp_bbox.x = MAX(int(centerX - temp_bbox.width / 2), 0);
      temp_bbox.y = MAX(int(centerY - temp_bbox.height / 2), 0);
      detected_boxes[i].push_back(temp_bbox);  // 0
    }
  }


  return 0;
}