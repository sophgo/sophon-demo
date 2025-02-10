//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV8_DET_H
#define YOLOV8_DET_H

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bm_wrapper.hpp"
#define DEBUG 0

struct YoloV8Box {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

using YoloV8BoxVec = std::vector<YoloV8Box>;

class YoloV8_det {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    // configuration
    bool agnostic = false;
    float m_confThreshold = 0.25;
    float m_nmsThreshold = 0.7;
    std::vector<std::string> m_class_names;
    int m_class_num = -1;
    int m_net_h, m_net_w;
    int max_det = 300;
    int max_wh = 7680;  // (pixels) maximum box width and height
    bmcv_convert_to_attr converto_attr;
    TimeStamp tmp_ts;

private:
    int pre_process(const std::vector<bm_image>& images, 
                    bm_tensor_t& input_tensor,
                    std::vector<std::pair<int, int>>& txy_batch, 
                    std::vector<float>& ratios_batch);
    int forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors);
    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    int post_process(const std::vector<bm_image>& input_images, 
                     std::vector<bm_tensor_t>& output_tensors, 
                     const std::vector<std::pair<int, int>>& txy_batch, 
                     const std::vector<float>& ratios_batch, 
                     std::vector<YoloV8BoxVec>& boxes);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    int argmax(float* data, int num);
    void xywh2xyxy(YoloV8BoxVec& xyxyboxes, std::vector<std::vector<float>> box);
    void NMS(YoloV8BoxVec& dets, float nmsConfidence);
    void clip_boxes(YoloV8BoxVec& yolobox_vec, int src_w, int src_h);
public:
    int batch_size = -1;
    TimeStamp* m_ts = NULL;
    YoloV8_det(std::string bmodel_file, std::string coco_names_file, int dev_id = 0, float confThresh = 0.25, float nmsThresh = 0.7){
        std::ifstream ifs(coco_names_file);
        if (ifs.is_open()) {
            std::string line;
            while (std::getline(ifs, line)) {
                line = line.substr(0, line.length() - 1);
                m_class_names.push_back(line);
            }
        }

        // set thresh 
        m_confThreshold = confThresh;
        m_nmsThreshold = nmsThresh;

        // get handle
        assert(BM_SUCCESS == bm_dev_request(&handle, dev_id));

        // judge now is pcie or soc
        assert(BM_SUCCESS == bm_get_misc_info(handle, &misc_info));

        // create bmrt
        bmrt = bmrt_create(handle);
        if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
            std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
        }

        // get network names from bmodel
        const char **names;
        int num = bmrt_get_network_number(bmrt);
        if (num > 1){
            std::cout << "This bmodel have " << num << " networks, and this program will only take network 0." << std::endl;
        }
        bmrt_get_network_names(bmrt, &names);
        for(int i = 0; i < num; ++i) {
            network_names.push_back(names[i]);
        }
        free(names);

        // get netinfo by netname
        netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
        if (netinfo->stage_num > 1){
            std::cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << std::endl;
        }
        batch_size = netinfo->stages[0].input_shapes[0].dims[0];
        m_net_h = netinfo->stages[0].input_shapes[0].dims[2];
        m_net_w = netinfo->stages[0].input_shapes[0].dims[3];
        
        for (int i = 0; i < netinfo->output_num; i++) {
            auto& shape = netinfo->stages[0].output_shapes[i];
            if (shape.num_dims == 3) {
                m_class_num = shape.dims[2] - 4;
            }
        }
        if (m_class_num == -1) {
            throw std::runtime_error("Invalid model output shape.");
        }

        float input_scale = netinfo->input_scales[0] / 255.f;
        converto_attr.alpha_0 = input_scale;
        converto_attr.beta_0 = 0;
        converto_attr.alpha_1 = input_scale;
        converto_attr.beta_1 = 0;
        converto_attr.alpha_2 = input_scale;
        converto_attr.beta_2 = 0;

        // set temp timestamp
        m_ts = &tmp_ts;
    }
    ~YoloV8_det(){
        if (bmrt!=NULL) {
            bmrt_destroy(bmrt);
            bmrt = NULL;
        }  
        bm_dev_free(handle);
    };
    int Detect(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& boxes);
    void draw_result(cv::Mat& img, YoloV8BoxVec& result);
    void draw_result_bmcv(bm_image& img, YoloV8BoxVec& result, bool put_text_flag=false); 
};

#endif  //! YOLOV8_DET_H
