//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef LIGHTSTEREO_H
#define LIGHTSTEREO_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#define DEBUG 0

class LightStereo {
    int m_dev_id = 0;
    bm_handle_t handle;
    void *bmrt;
    const bm_net_info_t *netinfo;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;
    bmcv_convert_to_attr converto_attr_left;
    bmcv_convert_to_attr converto_attr_right;
    int m_net_h, m_net_w;

private:
    int pre_process(const std::vector<bm_image>& images, 
                    bm_tensor_t& input_tensor, bool is_left);
    int forward(bm_tensor_t& input_tensor_left, bm_tensor_t& input_tensor_right, bm_tensor_t& output_tensor);
    float* get_cpu_data(bm_tensor_t* tensor, float scale);
public:
    int batch_size;
    TimeStamp* m_ts;
    LightStereo(std::string bmodel_file, int dev_id = 0){
        m_dev_id = dev_id;
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

        float input_scale_left = netinfo->input_scales[0];
        float input_scale_right = netinfo->input_scales[1];
        const std::vector<float> std = {0.229, 0.224, 0.225};
        const std::vector<float> mean = {0.485, 0.456, 0.406};
        converto_attr_left.alpha_0 = 1 / (255. * std[0]) * input_scale_left;
        converto_attr_left.alpha_1 = 1 / (255. * std[1]) * input_scale_left;
        converto_attr_left.alpha_2 = 1 / (255. * std[2]) * input_scale_left;
        converto_attr_left.beta_0 = (-mean[0] / std[0]) * input_scale_left;
        converto_attr_left.beta_1 = (-mean[1] / std[1]) * input_scale_left;
        converto_attr_left.beta_2 = (-mean[2] / std[2]) * input_scale_left;

        converto_attr_right.alpha_0 = 1 / (255. * std[0]) * input_scale_right;
        converto_attr_right.alpha_1 = 1 / (255. * std[1]) * input_scale_right;
        converto_attr_right.alpha_2 = 1 / (255. * std[2]) * input_scale_right;
        converto_attr_right.beta_0 = (-mean[0] / std[0]) * input_scale_right;
        converto_attr_right.beta_1 = (-mean[1] / std[1]) * input_scale_right;
        converto_attr_right.beta_2 = (-mean[2] / std[2]) * input_scale_right;

    }
    ~LightStereo(){
        if (bmrt!=NULL) {
            bmrt_destroy(bmrt);
            bmrt = NULL;
        }  
        bm_dev_free(handle);
    };
    std::vector<cv::Mat> process(const std::vector<bm_image>& left_imgs, const std::vector<bm_image>& right_imgs);
};

#endif  //! LIGHTSTEREO_H
