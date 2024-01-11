//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLO_H
#define YOLO_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "engine.h"
#include "utils.hpp"
#include "cvwrapper.h"

// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

struct YoloBox {
    int x, y, width, height;
    float score;
    int class_id;
};

using YoloBoxVec = std::vector<YoloBox>;

class Yolo {
    std::shared_ptr<sail::Engine>              engine;
    std::shared_ptr<sail::Bmcv>                bmcv;
    std::vector<std::string>                   graph_names;    
    std::vector<std::string>                   input_names;    
    std::vector<int>                           input_shape;   //1 input
    std::vector<std::string>                   output_names;   
    std::vector<std::vector<int>>              output_shape;  //1 or 3 output  
    bm_data_type_t                             input_dtype;    
    bm_data_type_t                             output_dtype;   
    std::shared_ptr<sail::Tensor>              input_tensor;
    std::vector<std::shared_ptr<sail::Tensor>> output_tensor;
    std::map<std::string, sail::Tensor*>       input_tensors; 
    std::map<std::string, sail::Tensor*>       output_tensors; 
    // configuration
    float m_confThreshold = 0.5;
    float m_nmsThreshold = 0.5;
    int m_anchors_type = 0;

    std::vector<std::string> m_class_names;
    int m_class_num = 80;  // default is coco names
    int m_net_h, m_net_w;
    int max_batch;
    int min_dim;
    float ab[6];

    TimeStamp* m_ts;

   private:
    int pre_process(sail::BMImage& input);
    template <std::size_t N>
    int pre_process(std::vector<sail::BMImage>& input);
    int post_process(std::vector<sail::BMImage>& images, std::vector<YoloBoxVec>& detected_boxes);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    static float sigmoid(float x);
    void NMS(YoloBoxVec& dets, float nmsConfidence);

   public:
    Yolo(int dev_id, std::string bmodel_file);
    virtual ~Yolo();
    int Init(float confThresh = 0.5, float nmsThresh = 0.5, const std::string& coco_names_file = "", int anchors_type=0);
    void enableProfile(TimeStamp* ts);
    int batch_size();
    int Detect(std::vector<sail::BMImage>& images, std::vector<YoloBoxVec>& boxes);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
    void draw_bmcv(int classId,
                   float conf,
                   int left,
                   int top,
                   int right,
                   int bottom,
                   sail::BMImage& frame,
                   bool put_text_flag = false);
};

#endif  //! YOLO_H
