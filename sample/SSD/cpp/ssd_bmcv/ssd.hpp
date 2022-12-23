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
#include "json.hpp"

#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "bmnn_utils.h"
#define BUFFER_SIZE (1024 * 500)
#define DEBUG 0
#define USE_ASPECT_RATIO 0
struct SSDObjRect{
    unsigned int class_id;
    float score;
    float x1, y1, x2, y2;
    void printBox(){
        std::cout << "class: " << class_id 
                  << "; score: " << score 
                  << "; box:[" << x1 << ", "
                  << y1 << ", " << x2 << ", " << y2 << "]" <<std::endl;
    };
};
class SSD {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    float m_conf_thre;
    float m_nms_thre;
    int m_net_h, m_net_w;
    int max_batch;
    int m_dev_id;
    TimeStamp *m_ts;
    public:
        SSD(std::shared_ptr<BMNNContext> context, int dev_id, float conf_thre, float nms_thre);
        ~SSD();
        void Init();
        int batch_size();
        int detect(const std::vector<cv::Mat> &input_images, const std::vector<std::string> &input_names,
                        std::vector<std::vector<SSDObjRect> > &results, cv::VideoWriter *VideoWriter=NULL);    
        void enableProfile(TimeStamp *ts);            
    private:
        static float get_aspect_scaled_ratio(int src_w, int src_h, 
                                             int dst_w, int dst_h, bool *alignWidth);
        int pre_process(const std::vector<cv::Mat> &images);
        int post_process(const std::vector<cv::Mat> &images, const std::vector<std::string> &input_names,
                               std::vector<std::vector<SSDObjRect> > &results, cv::VideoWriter *VideoWriter=NULL);
};
