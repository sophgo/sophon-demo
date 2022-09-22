/* Copyright 2019-2024 by Bitmain Technologies Inc. All rights reserved.

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
#ifndef YOLOV3_HPP
#define YOLOV3_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "bmruntime_interface.h"
#include "utils.hpp"

#define USE_OPENCV
#include "bm_wrapper.hpp"

struct ObjRect {
  unsigned int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
};

struct yolov3_box {
  float x;
  float y;
  float w;
  float h;
};

struct yolov3_DetectRect {
  int left;
  int right;
  int top;
  int bot;
  float score;
  int category;
};

struct detection {
  yolov3_box bbox;
  int classes;
  float *prob;
  float *mask;
  float objectness;
  int sort_class;
};

struct layer {
  int batch;
  int total;
  int n, c, h, w;
  int out_n, out_c, out_h, out_w;
  int classes;
  int inputs, outputs;
  int *mask;
  float *biases;
  float *output;
  float *output_gpu;
};

/* TOOD: double check if the max number is reasonable */
#define MAX_YOLOV3_OUTPUT_NUM 4

class YOLO {
public:
  YOLO(const std::string bmodel, int dev_id);
  ~YOLO();
  void preForward(std::vector<cv::Mat>& images);
  void forward();
  std::vector<std::vector<yolov3_DetectRect> > postForward();
  void enableProfile(TimeStamp *ts);
  int getBatchSize();

private:
  void preprocess(bm_image& in, bm_image& out);

  /* bbox iou calculation */
  float box_iou(yolov3_box a, yolov3_box b);
  /* bbox nms */
  void do_nms_sort(detection *det, int total, int classes, float thresh);
  /* init the yolo layer */
  layer make_yolo_layer(int blob_index, int batch, int w, int h, int n, int total, int classes);
  /* free yolo layer */
  void free_yolo_layer(layer l);
  /* get the dst index */
  int entry_index(layer l, int batch, int location, int entry);
  /* copy inference output to yolo layer input */
  void forward_yolo_layer(const float *input, layer l);
  /* filter bbox with obj_prob */
  int yolo_num_detections(layer l, float thresh);
  /* calculation the bbox num */
  int num_detections(std::vector<layer> layers_params, float thresh);
  /* alloc for bbox */
  detection *make_network_boxes(
      std::vector<layer> layers_params,
      float              thresh,
      int                *num);
  /* link bbox to img size */
  void correct_yolo_boxes(
      detection* det,
      int        n,
      int        w,
      int        h,
      int        netw,
      int        neth,
      int        relative);
  /* link bbox to net input size */
  yolov3_box get_yolo_box(
      float* x,
      float* biases,
      int    n,
      int    index,
      int    i,
      int    j,
      int    lw,
      int    lh,
      int    w,
      int    h,
      int    stride);
  /* get bbox */
  int get_yolo_detections(
      layer      l,
      int        w,
      int        h,
      int        netw,
      int        neth,
      float      thresh,
      int*       map,
      int        relative,
      detection* dets);
  /* fill the bbox */
  void fill_network_boxes(
      std::vector<layer> layers_params,
      int                w,
      int                h,
      float              thresh,
      float              hier,
      int*               map,
      int                relative,
      detection*         dets);
  /* return the bbox result before nms */
  detection* get_network_boxes(
      std::vector<layer> layers_params,
      int                img_w,
      int                img_h,
      float              thresh,
      float              hier,
      int*               map,
      int                relative,
      int*               num);
  /* get detection result */
  detection* get_detections(
      std::vector<float*> blobs,
      int                 img_w,
      int                 img_h,
      int*                nboxes);
  /* free the bbox prob */
  void free_detections(detection* dets, int nboxes);
  /* post process set the dst index */
  int set_index(
      int w,
      int h,
      int num,
      int classes,
      int batch,
      int location,
      int entry);
  /* calculation the max index label */
  int max_index(float *a, int n);
  /* return final bbox */
  std::vector<yolov3_DetectRect> detection_yolov3_process(
      detection* dets,
      int        nboxes,
      int        rows,
      int        cols);

  /* handle of low level device */
  bm_handle_t bm_handle_;

  /* runtime helper */
  const char **net_names_;
  void *p_bmrt_;

  /* network input shape */
  int batch_size_;
  int num_channels_;

  /* anchor */
  float biases_[18] = { 12, 16, 19, 36, 40, 28, 36, 75,
                                76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
  int masks_[9] = { 0, 1, 2, 3, 4, 5, 6, 7, 8};
  const int anchor_num_ = 3;
  const size_t classes_num_ = 80;
  int* fm_size_;

  float threshold_prob_;
  float threshold_nms_;
  int net_h_;
  int net_w_;

  float input_scale;
  float output_scale;
  std::vector<void*> outputs_;
  bool int8_flag_;
  int output_num_;
  /* for profiling */
  TimeStamp *ts_;
  // linear transformation arguments of BMCV
  bmcv_convert_to_attr convert_attr_;
  bm_shape_t input_shape_;
  bm_image* scaled_inputs_;
  std::vector<cv::Mat> images_;
  std::vector<int> output_sizes_;
};

#endif /* YOLOV3_HPP */
