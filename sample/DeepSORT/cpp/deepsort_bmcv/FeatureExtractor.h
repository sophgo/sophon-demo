//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <vector>
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "bmnn_utils.h"
#include "model.h"
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp

class FeatureExtractor {
   private:
    /* data */
   public:
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    int m_net_h, m_net_w;
    int max_batch;
    bmcv_convert_to_attr converto_attr;
    TimeStamp* m_ts;

   private:
    int pre_process(const bm_image& image, std::vector<bmcv_rect_t> crop_rects);
    int post_process(DETECTIONS& det, int start, int crop_size);

   public:
    FeatureExtractor(std::shared_ptr<BMNNContext> context);
    virtual ~FeatureExtractor();
    void Init();
    int batch_size();
    bool getRectsFeature(const bm_image& image, DETECTIONS& det);
    void enableProfile(TimeStamp* ts) { m_ts = ts; }
};
