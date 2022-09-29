/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include "yolov3.hpp"
#include "utils.hpp"
#include "cfg_parser.hpp"
using namespace std;

YOLO::YOLO(const std::string cfg_file, const std::string bmodel, int dev_id, float conf_thresh=0.5f, float nms_thresh=0.45f) {
  /* parser initialization*/
  biases_ = nullptr;
  masks_ = nullptr;
  num_ = 0;
  anchor_num_ = 0;
  classes_num_ = 0;
  char *tmp = (char*)cfg_file.c_str();
  cfg_parser(tmp);

  threshold_prob_ = conf_thresh;
  threshold_nms_ = nms_thresh;

  /* create device handler */
  bm_dev_request(&bm_handle_, dev_id);

  /* create inference runtime handler */
  p_bmrt_ = bmrt_create(bm_handle_);

  /* load bmodel by file */
  bool flag = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
  if (!flag) {
    std::cout << "ERROR: failed to load bmodel[" << bmodel << "] " << std::endl;
    exit(-1);
  }

  bmrt_get_network_names(p_bmrt_, &net_names_);
  std::cout << "> Load model " << net_names_[0] << " successfully" << std::endl;

  /* more info pelase refer to bm_net_info_t in bmdef.h */
  auto net_info = bmrt_get_network_info(p_bmrt_, net_names_[0]);
  std::cout << "input scale:" << net_info->input_scales[0] << std::endl;
  std::cout << "output scale:" << net_info->output_scales[0] << std::endl;
  std::cout << "input number:" << net_info->input_num << std::endl;
  std::cout << "output number:" << net_info->output_num << std::endl;
  bm_image_data_format_ext data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;

  /* TODO: get class number from net_info */

  /* get fp32/int8 type, the thresholds may be different */
  if (BM_FLOAT32 == net_info->input_dtypes[0]) {
    int8_flag_ = false;
    std::cout <<  "fp32 model" << std::endl;
    data_type = DATA_TYPE_EXT_FLOAT32;
  } else {
    int8_flag_ = true;
    std::cout <<  "int8 model" << std::endl;
  }
  bmrt_print_network_info(net_info);

  /*
   * only one input shape supported in the pre-built model
   * you can get stage_num from net_info
   */
  auto &input_shape = net_info->stages[0].input_shapes[0];
  /* malloc input and output system memory for preprocess data */
  int count = bmrt_shape_count(&input_shape);
  std::cout << "input count:" << count << std::endl;

  output_num_ = net_info->output_num;
  fm_size_ = new int[output_num_ * 2];
  for (int i = 0; i < output_num_; i++) {
    auto &output_shape = net_info->stages[0].output_shapes[i];
    count = bmrt_shape_count(&output_shape);
    std::cout << "output " << i << " count:" << count << std::endl;
    float* out = new float[count];
    outputs_.push_back(out);
    fm_size_[i * 2] = output_shape.dims[3];
    fm_size_[i * 2 + 1] = output_shape.dims[2];
    output_sizes_.push_back(output_shape.dims[1] *
            output_shape.dims[2] * output_shape.dims[3]);
  }

  batch_size_ = input_shape.dims[0];
  num_channels_ = input_shape.dims[1];
  net_h_ = input_shape.dims[2];
  net_w_ = input_shape.dims[3];
  input_shape_ = {4, {batch_size_, 3, net_h_, net_w_}};

  float input_scale = 1.0 / 255;
  if (int8_flag_) {
    input_scale *= net_info->input_scales[0];
  }
  convert_attr_.alpha_0 = input_scale;
  convert_attr_.beta_0 = 0;
  convert_attr_.alpha_1 = input_scale;
  convert_attr_.beta_1 = 0;
  convert_attr_.alpha_2 = input_scale;
  convert_attr_.beta_2 = 0;
  scaled_inputs_ = new bm_image[batch_size_];

  bm_status_t ret = bm_image_create_batch(bm_handle_, net_h_, net_w_,
                        FORMAT_RGB_PLANAR,
                        data_type,
                        scaled_inputs_, batch_size_);
  if (BM_SUCCESS != ret) {
    std::cerr << "ERROR: bm_image_create_batch failed" << std::endl;
    exit(1);
  }
  ts_ = nullptr;

}

YOLO::~YOLO() {
  if (masks_){
    delete []masks_;
    masks_ = nullptr;
  }
  if (biases_){
    delete []biases_;
    biases_ = nullptr;
  }

  delete []fm_size_;
  bm_image_destroy_batch(scaled_inputs_, batch_size_);
  if (scaled_inputs_) {
    delete []scaled_inputs_;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    delete [] reinterpret_cast<float*>(outputs_[i]);
  }
  free(net_names_);
  bmrt_destroy(p_bmrt_);
  bm_dev_free(bm_handle_);
}

void YOLO::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

void YOLO::preprocess(bm_image& in, bm_image& out) {
  bm_image_create(bm_handle_, net_h_, net_w_,
             FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &out, NULL);
  bmcv_rect_t crop_rect = {0, 0, in.width, in.height};
  bmcv_image_vpp_convert(bm_handle_, 1, in, &out, &crop_rect);
}

void YOLO::preForward(std::vector<cv::Mat>& images) {
  vector<bm_image> processed_imgs;
  images_.clear();
  for (size_t i = 0; i < images.size(); i++) {
    bm_image bmimg;
    bm_image processed_img;
    bm_image_from_mat(bm_handle_, images[i], bmimg);
    preprocess(bmimg, processed_img);
    bm_image_destroy(bmimg);
    processed_imgs.push_back(processed_img);
    images_.push_back(images[i]);
  }
  bmcv_image_convert_to(bm_handle_, batch_size_,
             convert_attr_, &processed_imgs[0], scaled_inputs_);

  for (size_t i = 0; i < images.size(); i++) {
    bm_image_destroy(processed_imgs[i]);
  }
}

void YOLO::forward() {
  bool res = bm_inference(p_bmrt_,
         scaled_inputs_, outputs_, input_shape_,
             reinterpret_cast<const char*>(net_names_[0]));
  if (!res) {
    std::cout << "ERROR : inference failed!!"<< std::endl;
    exit(1);
  }
}

static int nms_comparator(const void* pa, const void* pb) {
  detection a = *reinterpret_cast<const detection*>(pa);
  detection b = *reinterpret_cast<const detection*>(pb);
  float diff = 0;
  if (b.sort_class >= 0) {
    diff = a.prob[b.sort_class] - b.prob[b.sort_class];
  } else {
    diff = a.objectness - b.objectness;
  }
  return diff < 0 ? 1 : -1;
}

float YOLO::box_iou(yolov3_box a, yolov3_box b) {
  float area1 = a.w * a.h;
  float area2 = b.w * b.h;
  float wi = std::min((a.x + a.w / 2), (b.x + b.w / 2))
             - std::max((a.x - a.w / 2), (b.x - b.w / 2));
  float hi = std::min((a.y + a.h / 2), (b.y + b.h / 2))
             - std::max((a.y - a.h / 2), (b.y - b.h / 2));
  float area_i = std::max(wi, 0.0f) * std::max(hi, 0.0f);
  return area_i / (area1 + area2 - area_i);
}

void YOLO::do_nms_sort(
    detection* dets,
    int        total,
    int        classes,
    float      thresh) {
  int i, j, k;
  k = total - 1;
  for (i = 0; i <= k; ++i) {
    if (dets[i].objectness == 0) {
      detection swap = dets[i];
      dets[i] = dets[k];
      dets[k] = swap;
      --k;
      --i;
    }
  }
  total = k + 1;
  for (k = 0; k < classes; ++k) {
    for (i = 0; i < total; ++i) {
      dets[i].sort_class = k;
    }
    qsort(dets, total, sizeof(detection), nms_comparator);
    for (i = 0; i < total; ++i) {
      if (dets[i].prob[k] == 0) continue;
      yolov3_box a = dets[i].bbox;
      for (j = i + 1; j < total; ++j) {
        yolov3_box b = dets[j].bbox;
        if (box_iou(a, b) > thresh) {
          dets[j].prob[k] = 0;
        }
      }
    }
  }
}

layer YOLO::make_yolo_layer(
    int    blob_index,
    int    batch,
    int    w,
    int    h,
    int    n,
    int    total,
    int    classes) {
  layer l = {0};
  l.n = n;
  l.total = total;
  l.batch = batch;
  l.h = h;
  l.w = w;
  l.c = n * (classes + 4 + 1);
  l.out_w = l.w;
  l.out_h = l.h;
  l.out_c = l.c;
  l.classes = classes;
  l.inputs = l.w * l.h * l.c;
  l.biases = reinterpret_cast<float*>(calloc(total * 2, sizeof(float)));

  for (int i = 0; i < total * 2; ++i) {
    l.biases[i] = biases_[i];  /* init the anchor size with pre value */
  }

  l.mask = reinterpret_cast<int*>(calloc(n, sizeof(int)));
  for (int i = 0; i < l.n; ++i) {
      l.mask[i] = masks_[blob_index * 3 + i];
  }
  l.outputs = l.inputs;
  l.output = reinterpret_cast<float*>(calloc(batch* l.outputs, sizeof(float)));
  return l;
}

void YOLO::free_yolo_layer(layer l) {
  if (NULL != l.biases) {
    free(l.biases);
    l.biases = NULL;
  }
  if (NULL != l.mask) {
    free(l.mask);
    l.mask = NULL;
  }
  if (NULL != l.output) {
    free(l.output);
    l.output = NULL;
  }
}

int YOLO::entry_index(
    layer    l,
    int      batch,
    int      location,
    int      entry) {
  int n = location / (l.w * l.h);
  int loc = location % (l.w * l.h);
  return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1)
         + entry * l.w * l.h + loc;
}

void YOLO::forward_yolo_layer(
    const float*    input,
    layer           l) {
  memcpy(l.output, reinterpret_cast<const float *>(input),
         l.outputs * l.batch * sizeof(float));
}

int YOLO::yolo_num_detections(
    layer     l,
    float     thresh) {
  int i, n;
  int count = 0;
  for (i = 0; i < l.w * l.h; ++i) {
    for (n = 0; n < l.n; ++n) {
      int obj_index = entry_index(l, 0, n * l.w * l.h + i, 4);
      if (l.output[obj_index] > thresh) {
        ++count;
      }
    }
  }
  return count;
}

int YOLO::num_detections(
    std::vector<layer> layers_params,
    float              thresh) {
  unsigned int i;
  int s = 0;
  for (i = 0; i < layers_params.size(); ++i) {
    layer l = layers_params[i];
    s += yolo_num_detections(l, thresh);
  }
  return s;
}

detection* YOLO::make_network_boxes(
    std::vector<layer>    layers_params,
    float                 thresh,
    int*                  num) {
  layer l = layers_params[0];
  int nboxes = num_detections(layers_params, thresh);
  if (num) {
    *num = nboxes;
  }
  detection* dets =
      reinterpret_cast<detection*>(calloc(nboxes, sizeof(detection)));
  for (int i = 0; i < nboxes; ++i) {
    dets[i].prob = reinterpret_cast<float*>(calloc(l.classes, sizeof(float)));
  }
  return dets;
}

void YOLO::correct_yolo_boxes(
    detection*    dets,
    int           n,
    int           w,
    int           h,
    int           netw,
    int           neth,
    int           relative) {
  int new_w = 0;
  int new_h = 0;
  if (((float)netw / w) < ((float)neth / h)) {
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_h = neth;
    new_w = (w * neth) / h;
  }
  new_w = netw;
  new_h = neth;
  for (int i = 0; i < n; ++i) {
    yolov3_box b = dets[i].bbox;
    b.x = (b.x - (netw - new_w) / 2. / netw)
        / ((float)new_w / netw);
    b.y = (b.y - (neth - new_h) / 2. / neth)
        / ((float)new_h / neth);
    b.w *= (float)netw / new_w;
    b.h *= (float)neth / new_h;
    if (!relative) {
      b.x *= w;
      b.w *= w;
      b.y *= h;
      b.h *= h;
    }
    dets[i].bbox = b;
  }
}

yolov3_box YOLO::get_yolo_box(
     float*    x,
     float*    biases,
     int       n,
     int       index,
     int       i,
     int       j,
     int       lw,
     int       lh,
     int       w,
     int       h,
     int       stride) {
  yolov3_box b;
  if (x[index + 1 * stride] > 1 || x[index + 1 * stride] < 0) {
    std::cout << x[index + 1 * stride] << std::endl;
  }
  b.x = (i + x[index + 0 * stride]) / lw;
  b.y = (j + x[index + 1 * stride]) / lh;
  b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
  b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
  return b;
}

int YOLO::get_yolo_detections(
    layer         l,
    int           w,
    int           h,
    int           netw,
    int           neth,
    float         thresh,
    int*          map,
    int           relative,
    detection*    dets) {
  float* predictions = l.output;
  int count = 0;
  for (int i = 0; i < l.w * l.h; ++i) {
    int row = i / l.w;
    int col = i % l.w;
    for (int n = 0; n < l.n; ++n) {
      int obj_index = entry_index(l, 0, n * l.w * l.h + i, 4);
      float objectness = predictions[obj_index];
      if (objectness <= thresh) {
        continue;
      }
      int box_index = entry_index(l, 0, n * l.w * l.h + i, 0);
      dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n],
                                      box_index, col, row, l.w, l.h,
                                      netw, neth, l.w * l.h);
      dets[count].objectness = objectness;
      dets[count].classes = l.classes;
      for (int j = 0; j < l.classes; ++j) {
        int class_index = entry_index(l, 0, n * l.w * l.h + i, 4 + 1 + j);
        float prob = objectness * predictions[class_index];
        dets[count].prob[j] = (prob > thresh) ? prob : 0;
      }
      ++count;
    }
  }
  correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
  return count;
}

void YOLO::fill_network_boxes(
    std::vector<layer>    layers_params,
    int                   w,
    int                   h,
    float                 thresh,
    float                 hier,
    int*                  map,
    int                   relative,
    detection*            dets) {
  for (size_t j = 0; j < layers_params.size(); ++j) {
    layer l = layers_params[j];
    int count = get_yolo_detections(l, w, h, net_w_, net_h_,
                                    thresh, map, relative, dets);
    dets += count;
  }
}

detection* YOLO::get_network_boxes(
    std::vector<layer>    layers_params,
    int                   img_w,
    int                   img_h,
    float                 thresh,
    float                 hier,
    int*                  map,
    int                   relative,
    int*                  num) {
  detection* dets = make_network_boxes(layers_params, thresh, num);
  fill_network_boxes(layers_params, img_w, img_h,
                     thresh, hier, map, relative, dets);
  return dets;
}

detection* YOLO::get_detections(
    std::vector<float*>    blobs,
    int                    img_w,
    int                    img_h,
    int*                   nboxes) {
  int classes = classes_num_;
  float thresh = threshold_prob_;
  float hier_thresh = threshold_prob_;
  float nms = threshold_nms_;

  std::vector<layer> layers_params;
  layers_params.clear();
  for (unsigned int i = 0; i < blobs.size(); ++i) {
    layer l_params = make_yolo_layer(i, 1, fm_size_[2 * i],
                    fm_size_[2 * i + 1], anchor_num_,
                    sizeof(biases_) * num_ / (sizeof(float) * 2), classes);
    layers_params.push_back(l_params);
    forward_yolo_layer(blobs[i], l_params);  /* blobs[i] host_mem data */
  }
  detection *dets = get_network_boxes(layers_params, img_w, img_h,
                                      thresh, hier_thresh, 0, 1, nboxes);
  /* release layer memory */
  for (unsigned int index = 0; index < layers_params.size(); ++index) {
    free_yolo_layer(layers_params[index]);
  }
  if (nms) {
    do_nms_sort(dets, (*nboxes), classes, nms);
  }
  return dets;
}

void YOLO::free_detections(detection *dets, int nboxes) {
  for (int i = 0; i < nboxes; ++i) {
    free(dets[i].prob);
  }
  free(dets);
}

int YOLO::set_index(
    int    w,
    int    h,
    int    num,
    int    classes,
    int    batch,
    int    location,
    int    entry) {
  int n = location / (w * h);
  int loc = location % (w * h);
  int c = num * (classes + 4 + 1);
  int output = w * h * c;
  return batch * output + n * w * h * (4 + classes + 1)
         + entry * w * h + loc;
}

int YOLO::max_index(float* a, int n) {
  if (n <= 0) return -1;
  int i, max_i = 0;
  float max = a[0];
  for (i = 1; i < n; ++i) {
    if (a[i] > max) {
      max = a[i];
      max_i = i;
    }
  }
  return max_i;
}

std::vector<yolov3_DetectRect> YOLO::detection_yolov3_process(
    detection*    dets,
    int           nboxes,
    int           cols,
    int           rows) {
  int classes = classes_num_;
  float thresh = threshold_prob_;
  std::vector<yolov3_DetectRect> dets_all;
  for (int k = 0; k < nboxes; k++) {
    yolov3_box b = dets[k].bbox;
    int left = (b.x - b.w / 2.) * cols;
    int right = (b.x + b.w / 2.) * cols;
    int top = (b.y - b.h / 2.) * rows;
    int bot = (b.y + b.h / 2.) * rows;
    bot = std::max(bot, top);
    top = std::min(bot, top);
    if (left < 0) left = 0;
    if (right > cols - 1) right = cols - 1;
    if (top < 0) top = 0;
    if (bot > rows - 1) bot = rows - 1;
    int category = max_index(dets[k].prob, classes);
    if (dets[k].prob[category] > thresh) {
      yolov3_DetectRect det_k;
      det_k.left = left;
      det_k.right = right;
      det_k.top = top;
      det_k.bot = bot;
      det_k.category = category;
      det_k.score = dets[k].prob[category];
      dets_all.push_back(det_k);
    }
  }
  return dets_all;
}

std::vector<std::vector<yolov3_DetectRect> > YOLO::postForward() {
  std::vector<std::vector<yolov3_DetectRect> > det_results;
  for (int i = 0; i < batch_size_; i++) {
    int nboxes = 0;
    std::vector<float*> blobs;
    for (int j = 0; j < output_num_; j++) {
      blobs.push_back(reinterpret_cast<float*>(
                            outputs_[j]) + output_sizes_[j] * i);
    }

    detection* dets = get_detections(blobs,
                images_[i].cols, images_[i].rows, &nboxes);
    std::vector<yolov3_DetectRect> det_result =
                   detection_yolov3_process(dets, nboxes,
                         images_[i].cols, images_[i].rows);
    free_detections(dets, nboxes);
    det_results.push_back(det_result);
  }
  return det_results;
}

int YOLO::getBatchSize() {
  return batch_size_;
}

void YOLO::cfg_parser(char *filename){
  char *a = nullptr;
  int num = 0;

  list_t *sections = read_cfg(filename);
  node *n = sections->front;
  if(!n) {
    error("Config file has no sections");
  }

  section *s = (section *)n->val;
  list_t *options = s->options;
  if(!is_network(s)){
    error("First section must be [network] or [params]");
  } 

  anchor_num_ = option_find_int(options, "num_anchors", 3);
  num = option_find_int(options, "num_classes", 80);
  if (num <= 0){
    error("num_classes must be >= 1");
  }
  classes_num_ = num;

  a = option_find_str(options, "anchors", 0);
  biases_ = parse_float_list(a, &num);
  if (num % anchor_num_ != 0){
    error("please check your num_anchors and anchors in .cfg");
  }

  a = option_find_str(options, "masks", 0);
  masks_ = parse_int_list(a, &num_);
  if (num != (num_ * 2)){
    error("please check your num_anchors, anchors and masks in .cfg");
  }

  free_section(s);
  free_list(sections);

}