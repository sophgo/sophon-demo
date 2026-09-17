//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLO26_SEM_H
#define YOLO26_SEM_H

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#define DEBUG 0

// Cityscapes 19 类（trainId 顺序），BGR 调色板
static const std::vector<std::vector<int>> kCityscapesPalette = {
    {128, 64, 128},   // road
    {244, 35, 232},   // sidewalk
    {70, 70, 70},     // building
    {102, 102, 156},  // wall
    {190, 153, 153},  // fence
    {153, 153, 153},  // pole
    {250, 170, 30},   // traffic light
    {220, 220, 0},    // traffic sign
    {107, 142, 35},   // vegetation
    {152, 251, 152},  // terrain
    {70, 130, 180},   // sky
    {220, 20, 60},    // person
    {255, 0, 0},      // rider
    {0, 0, 142},      // car
    {0, 0, 70},       // truck
    {0, 60, 100},     // bus
    {0, 80, 100},     // train
    {0, 0, 230},      // motorcycle
    {119, 11, 32},    // bicycle
};

class Yolo26_sem {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    int m_net_h, m_net_w;
    int m_out_h, m_out_w;  // bmodel 输出类别图空间分辨率（= net_h/net_w，bilinear+argmax 已烘焙进图）
    bmcv_convert_to_attr converto_attr;
    TimeStamp tmp_ts;

private:
    int pre_process(const std::vector<bm_image>& images,
                    bm_tensor_t& input_tensor,
                    std::vector<std::pair<float, float>>& ratios_batch);
    int forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors);
    int32_t* get_cpu_data(bm_tensor_t* tensor);
    int post_process(const std::vector<bm_image>& input_images,
                     int32_t* data_seg,
                     std::vector<bm_tensor_t>& output_tensors,
                     const std::vector<std::pair<float, float>>& ratios_batch,
                     std::vector<cv::Mat>& segmaps);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);

public:
    int batch_size = -1;
    TimeStamp* m_ts = NULL;
    Yolo26_sem(std::string bmodel_file, int dev_id = 0);
    ~Yolo26_sem();
    int Detect(const std::vector<bm_image>& images, std::vector<cv::Mat>& segmaps);
    void draw_result(cv::Mat& img, const cv::Mat& segmap, cv::Mat& blend_out);
};

#endif  //! YOLO26_SEM_H