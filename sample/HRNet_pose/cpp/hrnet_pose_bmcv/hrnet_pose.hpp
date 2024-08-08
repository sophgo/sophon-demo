//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef HRNET_POSE_H
#define HRNET_POSE_H

#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
#include "../dependencies/include/yolov5.hpp"

using namespace std;

class HRNetPose {
    
    shared_ptr<BMNNContext> m_bmContext;  
    shared_ptr<BMNNNetwork> m_bmNetwork;  
    
    vector<bm_image> m_resized_imgs; 
    vector<bm_image> m_converto_imgs; 

    bool m_flip = true;                      
    int max_batch;
    int m_net_h, m_net_w;                  
    vector<string> m_class_names;

    bmcv_convert_to_attr linear_trans_param_; 
    TimeStamp *m_ts; 

    private:

    int pre_process(const bm_image& image, YoloV5Box& box);

    int post_process(vector<cv::Mat>& heapMaps, YoloV5Box& box, vector<cv::Point2f>& keypoints, vector<float>& maxvals);

    void transform_preds(vector<cv::Point2f>& preds, YoloV5Box& box, vector<cv::Point2f>& keypoints);

    vector<float> mean_ = {0.485, 0.456, 0.406};
    vector<float> scale_ = {1 / 0.229, 1 / 0.224, 1 / 0.225};


    public:

    HRNetPose(shared_ptr<BMNNContext> context);

    virtual ~HRNetPose();

    int Init(bool flip, const string& coco_names_file);

    void enableProfile(TimeStamp *ts);

    int get_batch_size();

    void drawPose(vector<cv::Point2f> keypoints, cv::Mat& image);

    int poseEstimate(const bm_image& image, YoloV5Box& box, vector<cv::Point2f>& keypoints, vector<float>& maxvals, vector<cv::Mat>& heatMaps);

    vector<vector<YoloV5Box>> get_person_detection_boxes(vector<vector<YoloV5Box>>& yolov5_boxes, float person_thresh);
 
};

#endif