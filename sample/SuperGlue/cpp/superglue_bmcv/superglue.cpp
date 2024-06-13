//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "superglue.hpp"

int SuperGlue::detect(torch::Tensor& keypoints0, torch::Tensor& scores0, torch::Tensor& descriptors0,
                torch::Tensor& keypoints1, torch::Tensor& scores1, torch::Tensor& descriptors1,
                torch::Tensor& matchings0, torch::Tensor& matchings0_score){
    std::vector<bm_tensor_t> input_tensors;
    std::vector<bm_tensor_t> output_tensors;
    ts->save("superglue preprocess");
    int ret = preprocess(keypoints0, scores0, descriptors0, keypoints1, scores1, descriptors1, input_tensors);
    assert(ret == 0);
    ts->save("superglue preprocess");

    ts->save("superglue forward");
    ret = forward(input_tensors, output_tensors);
    assert(ret == 0);
    ts->save("superglue forward");

    ts->save("superglue postprocess");
    ret = postprocess(output_tensors, matchings0, matchings0_score);
    assert(ret == 0);
    ts->save("superglue postprocess");
    
    return 0;
}

int SuperGlue::preprocess(torch::Tensor& keypoints0, torch::Tensor& scores0, torch::Tensor& descriptors0,
                torch::Tensor& keypoints1, torch::Tensor& scores1, torch::Tensor& descriptors1,
                    std::vector<bm_tensor_t>& input_tensors){
    input_tensors.clear();
    input_tensors.resize(netinfo->input_num);
    char known_input_names[6][20] = {"keypoints0", "scores0", "descriptors0", "keypoints1", "scores1", "descriptors1"};
    for(int i = 0; i < netinfo->input_num; i++){
        if(std::strcmp(netinfo->input_names[i], known_input_names[i]) == 0){
            bmrt_tensor(&input_tensors[i], bmrt, netinfo->input_dtypes[i], netinfo->stages[current_stage].input_shapes[i]);
            bm_memset_device(handle, 0, input_tensors[i].device_mem);
        }else{
            std::cerr << "unknown bmodel input_name[" << i << "]: " << netinfo->input_names[i] <<std::endl;
            return -1;
        }
    }
    int keypoints0_size = MIN(keypoints0.size(0), keypoints_sizes[current_stage]);
    int keypoints1_size = MIN(keypoints1.size(0), keypoints_sizes[current_stage]);

    assert(BM_SUCCESS == bm_memcpy_s2d_partial_offset(handle, input_tensors[0].device_mem, (void*)keypoints0.data_ptr<float>(), keypoints0_size * 2 * sizeof(float), 0));
    assert(BM_SUCCESS == bm_memcpy_s2d_partial_offset(handle, input_tensors[1].device_mem, (void*)scores0.data_ptr<float>(), keypoints0_size * sizeof(float), 0));
    
    for(int i = 0; i < descriptors0.size(0); i++){
        float *desc = descriptors0[i].data_ptr<float>();
        assert(BM_SUCCESS == bm_memcpy_s2d_partial_offset(handle, 
                                                         input_tensors[2].device_mem, 
                                                         (void*)desc, 
                                                         keypoints0_size * sizeof(float), 
                                                         i * keypoints_sizes[current_stage] * sizeof(float)));
    }
    
    assert(BM_SUCCESS == bm_memcpy_s2d_partial_offset(handle, input_tensors[3].device_mem, (void*)keypoints1.data_ptr<float>(), keypoints1_size * 2 * sizeof(float), 0));
    assert(BM_SUCCESS == bm_memcpy_s2d_partial_offset(handle, input_tensors[4].device_mem, (void*)scores1.data_ptr<float>(), keypoints1_size * sizeof(float), 0));

    for(int i = 0; i < descriptors1.size(0); i++){
        float *desc = descriptors1[i].data_ptr<float>();
        assert(BM_SUCCESS == bm_memcpy_s2d_partial_offset(handle, 
                                                         input_tensors[5].device_mem, 
                                                         (void*)desc, 
                                                         keypoints1_size * sizeof(float), 
                                                         i * keypoints_sizes[current_stage] * sizeof(float)));
    }
    return 0;
}
    
int SuperGlue::forward(std::vector<bm_tensor_t>& input_tensors, std::vector<bm_tensor_t>& output_tensors){
    output_tensors.clear();
    output_tensors.resize(netinfo->output_num);
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, input_tensors.data(), netinfo->input_num,
                    output_tensors.data(), netinfo->output_num);
    assert(ok == true);
    assert(BM_SUCCESS == bm_thread_sync(handle));

    for(int i = 0; i < input_tensors.size(); i++){
        bm_free_device(handle, input_tensors[i].device_mem);
    }
    
    return 0;
}

/**
 * @name    get_cpu_data
 * @brief   get cpu data of tensor.
 *
 * @param   [in]           tensor   input tensor.
 * @param   [in]           scale    scale of tensor.
 * @retval  float*         tensor's cpu data.
 */
int SuperGlue::get_cpu_data(bm_tensor_t* tensor, float scale, torch::Tensor& output_data){
    int ret = 0;
    int count = bmrt_shape_count(&tensor->shape);
    output_data = torch::zeros({count}, torch::kFloat32);
    auto accesor = output_data.accessor<float, 1>();
    if (tensor->dtype == BM_FLOAT32) {
        ret = bm_memcpy_d2s_partial(handle, output_data.data_ptr<float>(), tensor->device_mem, count * sizeof(float));
        assert(BM_SUCCESS ==ret);
    } else if (BM_INT8 == tensor->dtype) {
        int8_t * pI8 = nullptr;
        int tensor_size = bmrt_tensor_bytesize(tensor);
        pI8 = new int8_t[tensor_size];
        assert(pI8 != nullptr);
        // dtype convert
        ret = bm_memcpy_d2s_partial(handle, pI8, tensor->device_mem, tensor_size);
        assert(BM_SUCCESS ==ret);
        for(int i = 0;i < count; ++ i) {
            accesor[i] = pI8[i] * scale;
        }
        delete [] pI8;
    }  else if (BM_UINT8 == tensor->dtype) {
        uint8_t * pUI8 = nullptr;
        int tensor_size = bmrt_tensor_bytesize(tensor);
        pUI8 = new uint8_t[tensor_size];
        assert(pUI8 != nullptr);
        // dtype convert
        ret = bm_memcpy_d2s_partial(handle, pUI8, tensor->device_mem, tensor_size);
        assert(BM_SUCCESS ==ret);
        for(int i = 0;i < count; ++ i) {
            accesor[i] = pUI8[i] * scale;
        }
        delete [] pUI8;
    }else{
        std::cerr << "unsupport dtype: " << tensor->dtype << std::endl;
        return -1;
    }
    return 0;
}

int SuperGlue::postprocess(std::vector<bm_tensor_t>& output_tensors,
                    torch::Tensor& matchings0, torch::Tensor& matchings0_score){
    matchings0.reset();
    matchings0_score.reset();
    std::map<std::string, bm_tensor_t> output_tensors_map;
    for(int i = 0; i < output_tensors.size(); i++){
        if(std::strcmp(netinfo->output_names[i], "matches0_Where") == 0
        || std::strcmp(netinfo->output_names[i], "matches0_Where_f32") == 0){
            assert(0 == get_cpu_data(&output_tensors[i], netinfo->output_scales[i], matchings0));
        } else if(std::strcmp(netinfo->output_names[i], "matching_scores0_Where") == 0
        || std::strcmp(netinfo->output_names[i], "matching_scores0_Where_f32") == 0){
            assert(0 == get_cpu_data(&output_tensors[i], netinfo->output_scales[i], matchings0_score));
        }
    }
    
    if(matchings0.numel() == 0){
        std::cerr << "Cannot find matches0_Where in bmodel's output_tensors" << std::endl;
        return -1;
    }
    if(matchings0_score.numel() == 0){
        std::cerr << "Cannot find matching_scores0_Where in bmodel's output_tensors" << std::endl;
        return -1;
    }

    for(int i = 0; i < output_tensors.size(); i++){
        bm_free_device(handle, output_tensors[i].device_mem);
    }
    return 0;
}