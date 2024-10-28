//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV8_H
#define YOLOV8_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#define DEBUG 0

struct obbBox{
    float x, y, w, h, angle, score;
    int class_id;
    friend std::ostream& operator<<(std::ostream& os, const obbBox& box) {
        os << "obbBox(x: " << box.x
           << ", y: " << box.y
           << ", w: " << box.w
           << ", h: " << box.h
           << ", angle: " << box.angle
           << ", score: " << box.score
           << ", class_id: " << box.class_id << ")";
        return os;
    }
};
using obbBoxVec = std::vector<obbBox>;

struct obbBox_xyxy{
    float x1, y1, x2, y2, x3, y3, x4, y4, score;
    int class_id;
};
using obbBoxVec_xyxy = std::vector<obbBox_xyxy>;


class YoloV8_obb {
    bm_handle_t handle;
    void *bmrt;
    const bm_net_info_t *netinfo;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    // configuration
    float m_confThreshold = 0.25;
    float m_nmsThreshold = 0.7;
    bool use_cpu_opt;
    std::vector<std::string> m_class_names = {"plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", 
    "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"};
    bool agnostic = false;
    int m_class_num = 15;  // default is coco names
    int mask_num = 0;
    int m_net_h, m_net_w;
    int output_num;
    int min_dim;
    int max_det = 300;
    int max_wh = 7680;  // (pixels) maximum box width and height
    bmcv_convert_to_attr converto_attr;


private:
    int pre_process(const std::vector<bm_image>& images, 
                    bm_tensor_t& input_tensor,
                    std::vector<std::pair<int, int>>& txy_batch, 
                    std::vector<float>& ratios_batch);
    int forward(bm_tensor_t& input_tensor, bm_tensor_t& output_tensor);
    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    int post_process(const std::vector<bm_image>& input_images, 
                     bm_tensor_t& output_tensor, 
                     const std::vector<std::pair<int, int>>& txy_batch, 
                     const std::vector<float>& ratios_batch, 
                     std::vector<obbBoxVec_xyxy>& boxes);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    int argmax(float* data, int num);
    void nms_rotated(obbBoxVec& dets, float nmsConfidence = 0.5);
    std::tuple<float, float, float> convariance_matrix(const obbBox& obb);
    float probiou(const obbBox& obb1, const obbBox& obb2, float eps = 1e-7);
    void regularize_rbox(obbBoxVec& obb);
    obbBoxVec_xyxy xywhr2xyxyxyxy(const obbBoxVec& obb);
public:
    int batch_size;
    TimeStamp* m_ts;
    YoloV8_obb(std::string bmodel_file, int dev_id = 0, float confThresh = 0.25, float nmsThresh = 0.7){
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
        if (batch_size > 1){
            std::cerr << "This bmodel have batchsize=" << batch_size << ", but this program only support batchsize=1." << std::endl;
            exit(1);
        }
        m_net_h = netinfo->stages[0].input_shapes[0].dims[2];
        m_net_w = netinfo->stages[0].input_shapes[0].dims[3];
        m_class_num = netinfo->stages[0].output_shapes[0].dims[2] - 5;

        float input_scale = netinfo->input_scales[0] / 255.f;
        converto_attr.alpha_0 = input_scale;
        converto_attr.beta_0 = 0;
        converto_attr.alpha_1 = input_scale;
        converto_attr.beta_1 = 0;
        converto_attr.alpha_2 = input_scale;
        converto_attr.beta_2 = 0;
    }
    ~YoloV8_obb(){
        if (bmrt!=NULL) {
            bmrt_destroy(bmrt);
            bmrt = NULL;
        }  
        bm_dev_free(handle);
    };
    int Init(float confThresh = 0.5, float nmsThresh = 0.5);
    int Detect(const std::vector<bm_image>& images, std::vector<obbBoxVec_xyxy>& boxes);
    void drawPred(obbBoxVec_xyxy& box_vec, cv::Mat& frame, float draw_thresh = 0.1);
};

#endif  //! YOLOV8_H
