//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef CLIP_HPP
#define CLIP_HPP

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <utility>
#include <filesystem>
#include "bmruntime_interface.h"

class CLIP {
public:
    void init(const std::string& image_model, const std::string& text_model, const int &dev_id);
    void deinit();
    std::vector<float> preprocess(const cv::Mat& image);
    std::vector<float> encode_image(const std::vector<float>& image);
    std::vector<float> encode_text(const std::vector<int>& text);
    std::vector<float> calculate_similarity(const std::vector<float>& image_features,
                                        const std::vector<std::vector<float>>& text_features);
    std::pair<std::vector<float>, std::vector<int>> topk(const std::vector<float>& x, int k);

    double encode_image_time;
    double encode_text_time;
    double preprocess_time;
    int top_k;

private:
    cv::Mat preprocess_cpu_letterbox(const cv::Mat& image);
    void letterbox(const cv::Mat& image, cv::Mat& outImage,
                const cv::Size& newShape = cv::Size(224, 224),
                const cv::Scalar& color = cv::Scalar(114, 114, 114),
                bool auto_ = true,
                bool scaleFill = false,
                bool scaleUp = true,
                int stride = 32);
    std::tuple<cv::Mat, std::pair<float, float>, std::pair<float, float>> letterbox(const cv::Mat& im, const cv::Size& new_shape, 
        const cv::Scalar& color = cv::Scalar(114, 114, 114), bool auto_pad = false, bool scaleFill = false, 
        bool scaleup = true, int stride = 32);
    void normalize(std::vector<float>& features);
    std::vector<float> softmax(const std::vector<float>& x);
    void net_launch(const bm_net_info_t *net, void *p_bmrt);

    size_t image_net_batch_size;
    size_t text_net_batch_size;
    size_t image_resolution;
    size_t embed_dim;
    bm_net_info_t* image_net;
    bm_net_info_t* text_net;
    std::vector<float> mean;
    std::vector<float> std;
    std::vector<std::vector<float>> text_projection;
    void *p_bmrt_image;
    void *p_bmrt_text;
    bm_handle_t bm_handle;
    bm_shape_t* image_net_input_shape;
    bm_shape_t* image_net_output_shape;
    bm_shape_t* text_net_output_shape;
    bm_shape_t* text_net_input_shape;
};

#endif // CLIP_HPP
