//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef SUPERPOINT_H
#define SUPERPOINT_H
#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "json.hpp"

#define EPSILON 1e-19
class SuperPoint{
public:
    int batch_size;
    TimeStamp *ts;
    SuperPoint(std::string bmodel_file, int dev_id = 0){
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
    }

    ~SuperPoint(){
        if (bmrt!=NULL) {
            bmrt_destroy(bmrt);
            bmrt = NULL;
        }  
        bm_dev_free(handle);
    }
    
    void set_nms_radius(int r){
        nms_radius=r;
    }
    void set_keypoint_threshold(float t){
        keypoint_threshold=t;
    }
    void set_max_keypoint_size(int m){
        max_keypoint_size=m;
    }
    int get_network_input_h(){
        return m_net_h;
    }
    int get_network_input_w(){
        return m_net_w;
    }

    int detect(const bm_image& image, torch::Tensor& keypoints, torch::Tensor& scores,
                      torch::Tensor& descriptors);
private:
    bm_handle_t handle;
    void *bmrt;
    const bm_net_info_t *netinfo;
    std::vector<std::string> network_names;
    int m_net_h, m_net_w;
    bm_misc_info misc_info;
    int nms_radius=8;
    float keypoint_threshold=0.0002;
    int max_keypoint_size=1024;
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
    bm_image bm_image_align_width(bm_handle_t& handle, const bm_image& input, int align_width=64);
    int preprocess(const std::vector<bm_image>& images, bm_tensor_t& input_tensor);
    
    int forward(bm_tensor_t& input_tensors, std::vector<bm_tensor_t>& output_tensors);

    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    torch::Tensor simple_nms(torch::Tensor& scores, int nms_radius=4);
    torch::Tensor sample_descriptors(torch::Tensor keypoints, torch::Tensor descriptors, int s);
    int postprocess(std::vector<bm_tensor_t>& output_tensors, torch::Tensor& keypoints, torch::Tensor& scores,
                      torch::Tensor& descriptors);

};


#endif
