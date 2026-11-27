//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef MOGE_HPP
#define MOGE_HPP

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#define USE_OPENCV 1
#include "bm_wrapper.hpp"

struct MoGeResult {
    float focal;
    float shift;
    float fx;
    float fy;
    float fov_x_deg;   // horizontal fov in degrees
    float fov_y_deg;   // vertical fov in degrees
    float depth_min;
    float depth_max;
    std::vector<float> depth;    // (H * W) metric depth
    std::vector<float> points;   // (H * W * 3) reprojected point map
    std::vector<float> normal;   // (H * W * 3) unit normal map
    std::vector<uint8_t> mask;   // (H * W) binary mask
};

class MoGe {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    int out_idx_points;
    int out_idx_normal;
    int out_idx_mask;
    int out_idx_metric_scale;

    TimeStamp tmp_ts;

private:
    int pre_process(const bm_image& src, bm_tensor_t& input_tensor);
    int forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors);
    int post_process(std::vector<bm_tensor_t>& output_tensors, MoGeResult& result);

public:
    int net_h;
    int net_w;
    int batch_size = 1;
    TimeStamp* m_ts = NULL;

    MoGe(std::string bmodel_file, int dev_id = 0);
    ~MoGe();

    // run full pipeline (preprocess -> inference -> postprocess) on a single BGR bm_image
    int Process(const bm_image& input, MoGeResult& result);
};

#endif  // MOGE_HPP