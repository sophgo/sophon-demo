//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef SUPERGLUE_H
#define SUPERGLUE_H
#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "json.hpp"
class SuperGlue{
public:
    int batch_size;
    TimeStamp *ts;
    SuperGlue(std::string bmodel_file, int dev_id = 0){
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
        batch_size = netinfo->stages[0].input_shapes[0].dims[0];
        for(int i = 0; i < netinfo->stage_num; i++){
            keypoints_sizes.push_back(netinfo->stages[i].input_shapes[0].dims[1]);
        }
    }

    ~SuperGlue(){
        if (bmrt!=NULL) {
            bmrt_destroy(bmrt);
            bmrt = NULL;
        }  
        bm_dev_free(handle);
    }
    
    std::vector<int> get_keypoints_size(){
        return keypoints_sizes;
    }
    int detect(torch::Tensor& keypoints0, torch::Tensor& scores0, torch::Tensor& descriptors0,
                torch::Tensor& keypoints1, torch::Tensor& scores1, torch::Tensor& descriptors1,
                torch::Tensor& matchings0, torch::Tensor& matchings0_score);

private:
    bm_handle_t handle;
    void *bmrt;
    const bm_net_info_t *netinfo;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;
    int current_stage=0;
    std::vector<int> keypoints_sizes; //different stage has different keypoint_size

    int preprocess(torch::Tensor& keypoints0, torch::Tensor& scores0, torch::Tensor& descriptors0,
                torch::Tensor& keypoints1, torch::Tensor& scores1, torch::Tensor& descriptors1,
                    std::vector<bm_tensor_t>& input_tensors);
    
    int forward(std::vector<bm_tensor_t>& input_tensors, std::vector<bm_tensor_t>& output_tensors);

    int get_cpu_data(bm_tensor_t* tensor, float scale, torch::Tensor& output_data);
    int postprocess(std::vector<bm_tensor_t>& output_tensors,
                    torch::Tensor& matchings0, torch::Tensor& matchings0_score);


};


#endif
