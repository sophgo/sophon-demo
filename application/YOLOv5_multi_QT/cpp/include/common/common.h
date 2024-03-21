// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#ifndef COMMON_H_
#define COMMON_H_


#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include <memory>
#include <vector>

#define MAX_BATCH 4
#define MAX_YOLO_INPUT_NUM 8
#define MAX_YOLO_ANCHOR_NUM 8
typedef struct {
  unsigned long long bottom_addr[MAX_YOLO_INPUT_NUM];
  unsigned long long top_addr;
  unsigned long long detected_num_addr;
  int input_num;
  int batch_num;
  int hw_shape[MAX_YOLO_INPUT_NUM][2];
  int num_classes;
  int num_boxes;
  int keep_top_k;
  float nms_threshold;
  float confidence_threshold;
  float bias[MAX_YOLO_INPUT_NUM * MAX_YOLO_ANCHOR_NUM * 2];
  float anchor_scale[MAX_YOLO_INPUT_NUM];
  int clip_box;
}__attribute__((packed)) tpu_kernel_api_yolov5NMS_t;


struct YoloV5Box {
  int x, y, width, height;
  float score;
  int class_id;
};

template<typename T>
struct FrameInfo{
    FrameInfo(int channel,std::shared_ptr<T> img):channel_id(channel),image_ptr(img){}
    int channel_id;
    std::shared_ptr<T> image_ptr;
};


struct FrameInfoDetect{
    FrameInfoDetect(int channel,std::shared_ptr<bm_image> img):channel_id(channel),image_ptr(img),boxs_vec(){}
    int channel_id;
    std::shared_ptr<bm_image> image_ptr;
    std::vector<YoloV5Box> boxs_vec;
};

inline float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth){
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w) {
        *alignWidth = true;
        ratio = r_w;
    } else {
        *alignWidth = false;
        ratio = r_h;
    }
    return ratio;
}

#endif