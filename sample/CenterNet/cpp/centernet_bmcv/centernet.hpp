//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef CENTERNET_HPP
#define CENTERNET_HPP

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#include "bm_wrapper.hpp"
#include "bmnn_utils.h"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

struct CenternetBox {
    int x, y, width, height;
    float score;
    int class_id;
};

using CenternetBoxVec = std::vector<CenternetBox>;

class Centernet {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;

    // configuration

    std::vector<std::string> m_class_names;
    int m_class_num = 80;  // default is coco names
    int m_hw_channels = 2; // height and width channel
    int m_offset_channels = 2;  // offset_channels
    int m_net_h, m_net_w;
    int max_batch;
    int output_num;
    int min_dim;

    std::unique_ptr<int[]>                    m_confidence_mask_ptr;              // centernet中每个grid置信度最高，且大于阈值的Mask
    std::unique_ptr<float[]>                  m_confidence_ptr;                   // centernet中每个grid置信度最高的confidence
    std::unique_ptr<float[]>                  m_detected_objs;    
    int m_detected_count;
    int m_feat_c;
    int m_feat_h;
    int m_feat_w;
    int m_area;
    bmcv_convert_to_attr converto_attr;

    TimeStamp* m_ts;

   private:
    int pre_process(const std::vector<bm_image>& images);
    int post_process(const std::vector<bm_image>& images,
                     std::vector<CenternetBoxVec>& boxes);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w,
                                         int src_h,
                                         int dst_w,
                                         int dst_h,
                                         bool* alignWidth);

   public:
    float m_confThreshold;
    Centernet(std::shared_ptr<BMNNContext> context);
    virtual ~Centernet();
    int Init(float confThresh = 0.5, const std::string& coco_names_file = "");
    void enableProfile(TimeStamp* ts);
    int batch_size();
    int Detect(const std::vector<bm_image>& images,
               std::vector<CenternetBoxVec>& boxes);
    void SimplyMaxpool2D(float* data);
    std::pair<int*, float*> MaxOfLocation(float** heatmap);
    int* topk(float* arr, int k=75);
    float FindMaxConfidenceObject(float score[], int count, int& idx);
    // void drawPred(int classId,
    //               float conf,
    //               int left,
    //               int top,
    //               int right,
    //               int bottom,
    //               cv::Mat& frame);
    void draw_bmcv(bm_handle_t& handle,
                   int classId,
                   float conf,
                   int left,
                   int top,
                   int right,
                   int bottom,
                   bm_image& frame,
                   bool put_text_flag = false,
                   float conf_threshold = 0.35);
    // friend bool cmp(const int & p, const int & q);
};

#endif