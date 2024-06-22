//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV9_H
#define YOLOV9_H

#include <vector>
#include <cassert>
#include <set>
#include <unordered_map>
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"

struct YoloV9Box {
    float x1, y1, x2, y2;
    // int x, y, width, height;
    float score;
    int class_id;
};


using YoloV9BoxVec = std::vector<YoloV9Box>;

class YoloV9 {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;

    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;

    //configuration
    float m_confThreshold= 0.25;
    float m_nmsThreshold = 0.7;
    bool use_cpu_opt;
    std::vector<std::string> m_class_names;
    int m_class_num = 80; // default is coco names
    int mask_num = 0;
    int m_net_h, m_net_w;
    int max_batch;
    int output_num;
    int min_dim;
    int max_det=300;
    int max_wh=7680; // (pixels) maximum box width and height
    bmcv_convert_to_attr converto_attr;

    TimeStamp *m_ts;

    private:
    int pre_process(const std::vector<bm_image>& images);
    int post_process(const std::vector<bm_image>& images, std::vector<YoloV9BoxVec>& boxes);
    int post_process_opt(const std::vector<bm_image>& images, std::vector<YoloV9BoxVec>& detected_boxes);
    int get_max_value_neon(float* cls_conf,float &max_value ,int & max_index,int i,int nout);

    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
    void xywh2xyxy(YoloV9BoxVec& xyxyboxes, std::vector<std::vector<float>> box);
    static bool YoloV9Box_cmp(YoloV9Box a, YoloV9Box b);
    void NMS(YoloV9BoxVec &dets, float nmsConfidence);
    void clip_boxes(YoloV9BoxVec& yolobox_vec, int src_w, int src_h);

    public:
    YoloV9(std::shared_ptr<BMNNContext> context);
    virtual ~YoloV9();
    int Init(float confThresh=0.5, float nmsThresh=0.5, const std::string& coco_names_file="");
    void enableProfile(TimeStamp *ts);
    int batch_size();
    int Detect(const std::vector<bm_image>& images, std::vector<YoloV9BoxVec>& boxes);
    void draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int right, int bottom, bm_image& frame, bool put_text_flag=false);
};

#endif //!YOLOV9_H
