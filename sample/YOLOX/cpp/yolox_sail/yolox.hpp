//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOX_H
#define YOLOX_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "engine.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
#include "cvwrapper.h"

// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

struct YoloXBox {
    int x, y, width, height;
    float score;
    int class_id;
};

using YoloXBoxVec = std::vector<YoloXBox>;

class YoloX {
    std::shared_ptr<sail::Engine>              engine;
    std::shared_ptr<sail::Bmcv>                bmcv;
    std::vector<std::string>                   graph_names;    
    std::vector<std::string>                   input_names;    
    std::vector<int>                           input_shape;   
    std::vector<std::string>                   output_names;   
    std::vector<int>                           output_shape;   
    bm_data_type_t                             input_dtype;    
    bm_data_type_t                             output_dtype;   
    std::shared_ptr<sail::Tensor>              input_tensor;
    std::shared_ptr<sail::Tensor>              output_tensor;
    std::map<std::string, sail::Tensor*>       input_tensors; 
    std::map<std::string, sail::Tensor*>       output_tensors; 

    // configuration
    float m_confThreshold = 0.5;
    float m_nmsThreshold = 0.5;

    std::vector<std::string> m_class_names;
    int m_class_num = 80;  // default is coco names
    int m_net_h, m_net_w;
    int max_batch;
    float ab[6];

    TimeStamp* m_ts;

   private:
    int pre_process(sail::BMImage& input);
    template <std::size_t N>
    int pre_process(std::vector<sail::BMImage>& input);
    int post_process(std::vector<sail::BMImage>& images, std::vector<YoloXBoxVec>& detected_boxes, bool p6);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    void NMS(YoloXBoxVec& dets, float nmsConfidence);

   public:
    YoloX(int dev_id, std::string bmodel_file);
    virtual ~YoloX();
    int Init(float confThresh = 0.5, float nmsThresh = 0.5, const std::string& coco_names_file = "");
    void enableProfile(TimeStamp* ts);
    int batch_size();
    int Detect(std::vector<sail::BMImage>& images, std::vector<YoloXBoxVec>& boxes);
    void draw_bmcv(int classId,float conf,int left,int top,int right,int bottom,sail::BMImage& frame,bool put_text_flag = false);
};

#endif  //! YOLOX_H
