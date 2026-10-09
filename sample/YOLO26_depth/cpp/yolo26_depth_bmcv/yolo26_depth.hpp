//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLO26_DEPTH_H
#define YOLO26_DEPTH_H

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#define DEBUG 0

class Yolo26_depth {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    int m_net_h, m_net_w;
    int m_out_h, m_out_w;  // bmodel 输出深度图分辨率（== net_h/net_w，上采样已烘焙进图）
    bmcv_convert_to_attr converto_attr;
    TimeStamp tmp_ts;

private:
    int pre_process(const std::vector<bm_image>& images,
                    bm_tensor_t& input_tensor);
    int forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors);
    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    int post_process(const std::vector<bm_image>& input_images,
                     float* data_depth,
                     std::vector<bm_tensor_t>& output_tensors,
                     std::vector<cv::Mat>& depth_maps);

public:
    int batch_size = -1;
    TimeStamp* m_ts = NULL;
    Yolo26_depth(std::string bmodel_file, int dev_id = 0);
    ~Yolo26_depth();
    int Detect(const std::vector<bm_image>& images, std::vector<cv::Mat>& depth_maps);
    static cv::Mat depth_to_vis(const cv::Mat& depth);
    static void save_depth_npy(const cv::Mat& depth, const std::string& path);
};

#endif  //! YOLO26_DEPTH_H