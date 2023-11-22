//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef PPYOLOE_H
#define PPYOLOE_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"

#define DEBUG 0

struct ppYoloeBox {
    int x, y, width, height;
    float score;
    int class_id;
};

using ppYoloeBoxVec = std::vector<ppYoloeBox>;

class ppYoloe {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_min_max_imgs;
    std::vector<bm_image> m_standard_imgs;
    bm_tensor_t m_input_ratio;

    // configuration
    float m_confThreshold = 0.5;
    float m_nmsThreshold = 0.5;

    std::vector<std::string> m_class_names;
    int m_class_num = 80;  // default is coco names
    int m_net_h, m_net_w;
    int max_batch;
    int m_output_num;
    bmcv_convert_to_attr standard_scaler;
    bmcv_convert_to_attr min_max_scaler;

    TimeStamp* m_ts;

   private:
    int pre_process(const std::vector<bm_image>& images);
    int post_process(const std::vector<bm_image>& images, std::vector<ppYoloeBoxVec>& detected_boxes);
    int argmax_interval(float *data, int class_num, int box_num);
    std::vector<float> get_img_ratio(int src_w, int src_h, int dst_w, int dst_h);
    void NMS(ppYoloeBoxVec& dets, float nmsConfidence);

   public:
    ppYoloe(std::shared_ptr<BMNNContext> context);
    virtual ~ppYoloe();
    int Init(float confThresh = 0.5, float nmsThresh = 0.5, const std::string& coco_names_file = "");
    void enableProfile(TimeStamp* ts);
    int batch_size();
    int Detect(const std::vector<bm_image>& images, std::vector<ppYoloeBoxVec>& boxes);
    void draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int right, int bottom, bm_image& frame, bool put_text_flag=false);
};

#endif
