//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <dirent.h>
#include <algorithm>
#include "json.hpp"

#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "bmnn_utils.h"
#define BUFFER_SIZE (1024 * 500)
#define DEBUG 0

class C3D {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::shared_ptr<BMNNTensor>  m_input_tensor;
    std::shared_ptr<BMNNTensor>  m_output_tensor;
    int m_step;
    int m_net_h, m_net_w;
    int m_clip_len;
    int m_num_channels;
    int max_batch;
    int m_dev_id;
    TimeStamp *m_ts;
    public:
        C3D(std::shared_ptr<BMNNContext> context, int step_len, int dev_id);
        ~C3D();
        void Init();
        int batch_size();
        int detect(const std::vector<std::string> &video_paths, std::vector<int> &preds);    
        void enableProfile(TimeStamp *ts);            
    private:
        float *m_input_f32;
        int8_t *m_input_int8;
        float *m_output_f32;
        int8_t *m_output_int8;
        cv::Mat m_mean;
        int m_input_count;
        int pre_process(const std::vector<cv::Mat> &decoded_frames);
        void setMean(std::vector<float> &values);
        void wrapInputLayer(std::vector<cv::Mat>* input_channels, int batch_id);
        void pre_process_video(const std::string video_path, std::vector<cv::Mat> &input_channels);
        void decode_video(const std::string video_path, std::vector<cv::Mat> &decoded_frames, int video_id);
};
