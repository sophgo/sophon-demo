//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef ARCFACE_H
#define ARCFACE_H

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#define DEBUG 0

class ArcFace {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    int m_net_h, m_net_w;
    int output_size_;
    bmcv_convert_to_attr converto_attr;
    TimeStamp tmp_ts;

private:
    int pre_process(const std::vector<bm_image>& images, bm_tensor_t& input_tensor);
    int forward(bm_tensor_t& input_tensor, bm_tensor_t& output_tensor);
    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    int post_process(const std::vector<bm_image>& input_images,
                     bm_tensor_t& output_tensor,
                     std::vector<std::vector<float>>& embeddings);

public:
    int batch_size = -1;
    TimeStamp* m_ts = NULL;
    ArcFace(std::string bmodel_file, int dev_id = 0){
        auto ret = bm_dev_request(&handle, dev_id);
        assert(BM_SUCCESS == ret);

        ret = bm_get_misc_info(handle, &misc_info);
        assert(BM_SUCCESS == ret);

        bmrt = bmrt_create(handle);
        if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
            std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
        }

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

        netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
        if (netinfo->stage_num > 1){
            std::cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << std::endl;
        }
        batch_size = netinfo->stages[0].input_shapes[0].dims[0];
        m_net_h = netinfo->stages[0].input_shapes[0].dims[2];
        m_net_w = netinfo->stages[0].input_shapes[0].dims[3];
        output_size_ = netinfo->stages[0].output_shapes[0].dims[1];

        // mean/scale is embedded in bmodel, so converto_attr just passes through
        float input_scale = netinfo->input_scales[0];
        converto_attr.alpha_0 = input_scale;
        converto_attr.beta_0 = 0;
        converto_attr.alpha_1 = input_scale;
        converto_attr.beta_1 = 0;
        converto_attr.alpha_2 = input_scale;
        converto_attr.beta_2 = 0;

        m_ts = &tmp_ts;
    }
    ~ArcFace(){
        if (bmrt!=NULL) {
            bmrt_destroy(bmrt);
            bmrt = NULL;
        }
        bm_dev_free(handle);
    };
    int Embed(const std::vector<bm_image>& images, std::vector<std::vector<float>>& embeddings);
};

#endif  //! ARCFACE_H
