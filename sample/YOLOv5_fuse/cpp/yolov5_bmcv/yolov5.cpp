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
#define USE_MULTICLASS_NMS 1
#define RESIZE 0
#define DUMP_FILE 0

const std::vector<std::vector<int>> colors = {{255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0}, \
                {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255}, {0, 170, 255}, {0, 85, 255}, \
                {0, 0, 255}, {85, 0, 255}, {170, 0, 255}, {255, 0, 255}, {255, 0, 170}, {255, 0, 85}, {255, 0, 0},\
                {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloV5::YoloV5(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
  std::cout << "YoloV5 ctor .." << std::endl;
}

YoloV5::~YoloV5() {
  std::cout << "YoloV5 dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int YoloV5::Init(const std::string& coco_names_file)
{
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
  m_net_h = tensor->get_shape()->dims[1];
  m_net_w = tensor->get_shape()->dims[2];



  //3. get output
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num == 1 || output_num == 3);
  min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

  // printf("max_batch:%d;output_num:%d; min_dim:%d\n", max_batch, output_num, min_dim);
  
  //4. initialize bmimages
  m_resized_imgs.resize(max_batch);

  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for(int i=0; i<max_batch; i++){
    auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i]);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8){
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }

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

void YoloV5::enableProfile(TimeStamp *ts)
{
  m_ts = ts;
}

int YoloV5::batch_size() {
  return max_batch;
};

int YoloV5::Detect(const std::vector<bm_image>& input_images, std::vector<YoloV5BoxVec>& boxes)
{
  int ret = 0;
  //3. prepare data
  m_ts->save("yolov5 preprocess", input_images.size());
  ret = prepare_data(input_images);
  CV_Assert(ret == 0);
  m_ts->save("yolov5 preprocess", input_images.size());

  //4. forward
  m_ts->save("yolov5 inference", input_images.size());
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  m_ts->save("yolov5 inference", input_images.size());

  //5. get result
  m_ts->save("yolov5 postprocess", input_images.size());
  ret = get_result(input_images, boxes);
  CV_Assert(ret == 0);
  m_ts->save("yolov5 postprocess", input_images.size());
  return ret;
}

//  尺度变化+填充
int YoloV5::prepare_data(const std::vector<bm_image>& images){
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
    auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i]);
#endif
    assert(BM_SUCCESS == ret);
    
#if DUMP_FILE
    cv::Mat resized_img;
    cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
    std::string fname = cv::format("resized_img_%d.jpg", images[i].height);
    cv::imwrite(fname, resized_img);
    // printf("images:%d %d %d\n", images[i].height, images[i].width,  images[i].image_format);
    // printf("image_aligned:%d %d %d\n", image_aligned.height, image_aligned.width,  image_aligned.image_format);
    // printf("m_resized_imgs:%d %d %d\n", m_resized_imgs[i].height, m_resized_imgs[i].width,  m_resized_imgs[i].image_format);
#endif
    if(need_copy) bm_image_destroy(image_aligned);
  }

  //3. attach to tensor
  if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_resized_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
  return 0;
}


float YoloV5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
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

int YoloV5::get_result(const std::vector<bm_image>& images, std::vector<YoloV5BoxVec>& detected_boxes) {
  YoloV5BoxVec yolobox_vec;
  std::shared_ptr<BMNNTensor> output_tensor=m_bmNetwork->outputTensor(0);
  int box_num = output_tensor->get_shape()->dims[2];
  // printf("box_num %d \n",box_num);
  int box_id = 0;
  auto output_data = (float*)output_tensor->get_cpu_data();
  auto cout = 0;
  for(int batch_idx = 0; batch_idx < images.size(); ++ batch_idx){
    yolobox_vec.clear();
    auto& frame = images[batch_idx];
    int frame_width = frame.width;
    int frame_height = frame.height;

    int tx1 = 0, ty1 = 0;
    float rx = float(frame.width) / m_net_w;
    float ry = float(frame.height) / m_net_h;
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(frame.width, frame.height, m_net_w, m_net_h, &isAlignWidth);
    if (isAlignWidth) {
      ty1 = (int)((m_net_h - (int)(frame_height*ratio)) / 2);
    }else{
      tx1 = (int)((m_net_w - (int)(frame_width*ratio)) / 2);
    }
    rx = 1.0 / ratio;
    ry = 1.0 / ratio;
#endif

    while(int(*(output_data+cout)) == batch_idx && box_id < box_num){
      YoloV5Box box;
      // init box data
      float centerX = *(output_data+cout+3);
      float centerY = *(output_data+cout+4);
      float width = *(output_data+cout+5);
      float height = *(output_data+cout+6);
      // get bbox
      box.x = int((centerX - width / 2 - tx1) * rx);
      if (box.x < 0) box.x = 0;
      box.y = int((centerY - height / 2  - ty1) * ry);
      if (box.y < 0) box.y = 0;
      box.width = width * rx;
      box.height = height * ry;

      box.class_id = int(*(output_data+cout+1));
      box.score = *(output_data+cout+2);
      cout += 7;
      box_id++;
      yolobox_vec.emplace_back(box);
    }
    detected_boxes.emplace_back(yolobox_vec);
  }
  return 0;
}



void YoloV5::draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int width, int height, bm_image& frame, float draw_thresh, bool put_text_flag)   // Draw the predicted bounding box
{
  if (conf < draw_thresh) return;
  int colors_num = colors.size();
  //Draw a rectangle displaying the bounding box
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(left, 0), frame.width);
  rect.start_y = MIN(MAX(top, 0), frame.height);
  rect.crop_w = MAX(MIN(width, frame.width - rect.start_x), 0);
  rect.crop_h = MAX(MIN(height, frame.height - rect.start_y), 0);
  int thickness = 2;
  if(rect.crop_w <= thickness * 2 || rect.crop_h <= thickness * 2){
        std::cout << "width or height too small, this rect will not be drawed: " << 
              "[" << rect.start_x << ", "<< rect.start_y << ", " << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
  } else{
    bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
  } 
  if (put_text_flag){
    //Get the label for the class name and its confidence
    std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
    bmcv_point_t org = {rect.start_x, rect.start_y};
    bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]};
    float fontScale = 2; 
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;   
    }
  }
}
