//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOWORLD_HPP
#define YOLOWORLD_HPP

#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "json.hpp"

#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "bmnn_utils.h"

struct YoloWorldBox {
    float x1, y1, x2, y2;   // xyxy in original-image coords
    float score;
    int   class_id;
};
using YoloWorldBoxVec = std::vector<YoloWorldBox>;

class YoloWorld {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::shared_ptr<BMNNTensor>  m_images_tensor;    // input 0: [1,3,640,640]
    std::shared_ptr<BMNNTensor>  m_txt_feats_tensor; // input 1: [1,N,512]

    int m_net_h = 640;
    int m_net_w = 640;
    int m_num_classes = 80;
    int m_output_nout = 84;       // output channels: 4 + num_classes
    int m_output_anchors = 8400;  // output anchors (dim 2)
    int max_batch = 1;

    float m_confThreshold = 0.25f;
    float m_nmsThreshold  = 0.7f;
    bool  m_agnostic = false;

    // Precomputed class names
    std::vector<std::string> m_class_names;

    // Host input buffer for preprocessed image [3*640*640]
    float* m_input_buf = nullptr;

    // Device tensors for both inputs
    bm_tensor_t m_images_dev;
    bm_tensor_t m_txt_feats_dev;
    bool        m_tensors_allocated = false;

    // Output buffer (F16 outputs converted to F32)
    std::vector<float> m_output_buf;

    int        m_dev_id;
    TimeStamp* m_ts = nullptr;

public:
    YoloWorld(std::shared_ptr<BMNNContext> context, int dev_id);
    ~YoloWorld();
    int  Init(float confThresh, float nmsThresh, const std::string& class_names_file);
    int  loadTxtFeats(const std::string& npy_path);
    int  loadClassNames(const std::string& class_names_file);
    void enableProfile(TimeStamp* ts) { m_ts = ts; }
    int  batch_size() { return max_batch; }
    void drawPred(cv::Mat& frame, const YoloWorldBox& box);

    int  Detect(const std::vector<cv::Mat>& images,
                std::vector<YoloWorldBoxVec>& boxes);

private:
    int  pre_process(const cv::Mat& image, float ratio[2], int txy[2]);
    void letterbox(const cv::Mat& src, cv::Mat& dst, int new_h, int new_w,
                   float& ratio, int& tx, int& ty);
    int  post_process(float ratio[2], int txy[2], int img_w, int img_h,
                      YoloWorldBoxVec& boxes);
    void retrieveOutput(std::vector<float>& out_buf);
    void NMS(YoloWorldBoxVec& dets, float nmsConfidence);
};

#endif // YOLOWORLD_HPP
