//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef FEAR_TRACKER_HPP
#define FEAR_TRACKER_HPP

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

struct Bbox {
    float x, y, w, h;
};

class FearTracker {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::shared_ptr<BMNNTensor>  m_template_tensor;  // input 0: [1,3,128,128]
    std::shared_ptr<BMNNTensor>  m_search_tensor;    // input 1: [1,3,256,256]

    // Tracking config (mirrors DEFAULT_CONFIG in fear_tracker.py)
    float penalty_k              = 0.062f;
    float window_influence        = 0.38f;
    float lr                      = 0.765f;
    int   total_stride             = 16;
    int   score_size               = 16;
    float template_bbox_offset     = 0.2f;
    int   search_context           = 2;
    int   instance_size            = 256;
    int   template_size            = 128;
    bool  smooth_mode              = false;

    // Grid and window (score_size x score_size, stored flat row-major)
    std::vector<float> m_grid_x;   // [ss*ss]
    std::vector<float> m_grid_y;   // [ss*ss]
    std::vector<float> m_window;   // [ss*ss]

    // ImageNet normalization
    float m_mean[3] = {0.485f, 0.456f, 0.406f};
    float m_std[3]  = {0.229f, 0.224f, 0.225f};

    // Tracking state
    Bbox       m_state_bbox;       // current bbox in original image coords
    Bbox       m_state_mapping;    // context box (extended bbox) in original image
    Bbox       m_state_prev_size;  // resized w,h in search-image coords
    cv::Scalar m_mean_color;       // mean color of initial frame (RGB)
    bool       m_initialized = false;

    // Host input buffers (preprocessed float32 CHW)
    float*  m_template_buf = nullptr;  // [3 * 128 * 128]
    float*  m_search_buf   = nullptr;  // [3 * 256 * 256]

    // Device tensors for both inputs
    bm_tensor_t m_template_dev;
    bm_tensor_t m_search_dev;
    bool        m_tensors_allocated = false;

    // Output identification (bbox_pred has 4 channels, cls_pred has 1)
    int m_bbox_idx = -1;
    int m_cls_idx  = -1;

    // Output buffers (managed as members to avoid leaks; F16 outputs are
    // converted to F32 here since BMNNTensor::get_cpu_data() does not handle F16)
    std::vector<float> m_bbox_output;  // [4 * 16 * 16]
    std::vector<float> m_cls_output;   // [1 * 16 * 16]

    int        m_dev_id;
    int        max_batch = 1;
    TimeStamp* m_ts = nullptr;

public:
    FearTracker(std::shared_ptr<BMNNContext> context, int dev_id);
    ~FearTracker();
    void Init();
    void enableProfile(TimeStamp* ts) { m_ts = ts; }
    void initialize(const cv::Mat& image_rgb, const Bbox& bbox);
    Bbox  update(const cv::Mat& image_rgb);

private:
    void makeGrid();
    void makeWindow();
    void preprocessToBuffer(const cv::Mat& image_rgb, float* buf, int H, int W);
    cv::Mat getExtendedCrop(const cv::Mat& image_rgb, const Bbox& bbox,
                            int crop_size, float offset,
                            const cv::Scalar& padding_value,
                            Bbox& resized_bbox, Bbox& context);
    Bbox extendBbox(const Bbox& bbox, float offset);
    Bbox ensureBboxBoundaries(const Bbox& bbox, int img_h, int img_w);
    Bbox clampBbox(const Bbox& bbox, int img_h, int img_w, int min_side = 3);
    Bbox postprocess(const float* bbox_pred, const float* cls_pred, float& max_score);
    Bbox rescaleBbox(const Bbox& bbox, const Bbox& padded_box);
    void retrieveOutput(int tensor_idx, std::vector<float>& out_buf);
};

#endif // FEAR_TRACKER_HPP
