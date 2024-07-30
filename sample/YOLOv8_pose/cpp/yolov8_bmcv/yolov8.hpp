//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV8_H
#define YOLOV8_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0


struct YoloV8Box {
    float x1, y1, x2, y2;
    float score;
    int index;
    int class_id;
    std::vector<float> keyPoints;
};

using YoloV8BoxVec = std::vector<YoloV8Box>;

class YoloV8 {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;

    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;

    //configuration
    float m_confThreshold= 0.25;
    float m_nmsThreshold = 0.7;
    int m_points_num = 17; 
    int mask_num = 0;
    int m_net_h, m_net_w;
    int max_batch;
    int output_num;
    int min_dim;
    int max_det=300;
    int max_wh=7680; // (pixels) maximum box width and height
    bmcv_convert_to_attr converto_attr;

    TimeStamp *m_ts;
    unsigned int m_colorIndex;

    private:
    int pre_process(const std::vector<bm_image>& images);
    int post_process(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& boxes);

    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
    void NMS(YoloV8BoxVec &dets, float nmsConfidence);
    int ReTransPoseBox(YoloV8BoxVec& v, float tx, float ty, float r, int fw, int fh, float* d, int n);
    int ProcessPoseBox(YoloV8BoxVec& v, float* d, int n);
    bmcv_color_t GetBmColor();
    public:
    YoloV8(std::shared_ptr<BMNNContext> context);
    virtual ~YoloV8();
    int Init(float confThresh=0.5, float nmsThresh=0.5);
    void enableProfile(TimeStamp *ts);
    int batch_size();
    int Detect(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& boxes);
    void draw_bmcv(bm_handle_t& handle, YoloV8Box& b, bm_image& frame, bool putScore);
    static const std::vector<std::pair<int, int>> pointLinks;

};

#endif //!YOLOV8_H
