//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef LPRNET_HPP
#define LPRNET_HPP

#include <iomanip>
#include <string>
#define USE_OPENCV 1
#include <opencv2/opencv.hpp>
#include "bmnn_utils.h"
#include "bmruntime_interface.h"
#include "utils.hpp"

// #define MAX_BATCH 4
// #define INPUT_WIDTH 94
// #define INPUT_HEIGHT 24
#define BUFFER_SIZE (1024 * 500)

// char * get_res(int pred_num[], int len_char, int clas_char);

class LPRNET {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::shared_ptr<BMNNTensor> m_input_tensor;
    std::shared_ptr<BMNNTensor> m_output_tensor;

    int m_net_h, m_net_w;
    int m_num_channels;
    int max_batch;
    int m_dev_id;
    int output_num;
    int len_char;
    int clas_char;
    TimeStamp* ts_ = NULL;
    bm_tensor_t input_tensor;
   public:
    LPRNET(std::shared_ptr<BMNNContext> context, int m_dev_id);
    ~LPRNET();
    int Init();
    int Detect(const std::vector<cv::Mat>& input_images,
               std::vector<std::string>& results);
    void enableProfile(TimeStamp* ts);
    int batch_size();

   private:
    float* m_input_f32;
    int8_t* m_input_int8;
    cv::Mat m_mean;
    cv::Mat m_std;
    int m_input_count;
    int pre_process(const std::vector<cv::Mat>& images);
    int post_process(const std::vector<cv::Mat>& images,
                     std::vector<std::string>& results);
    void setStdMean(std::vector<float>& std, std::vector<float>& mean);
    void wrapInputLayer(std::vector<cv::Mat>* input_channels,
                        const int batch_id);
    void pre_process_image(const cv::Mat& img,
                           std::vector<cv::Mat>* input_channels);
    int argmax(float* data, int dsize);
    std::string get_res(int pred_num[], int len_char, int clas_char);
};

#endif /* LPRNET_HPP */
