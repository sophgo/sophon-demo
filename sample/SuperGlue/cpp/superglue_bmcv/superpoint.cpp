//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "superpoint.hpp"
#include <fstream>

int SuperPoint::detect(const bm_image& image, torch::Tensor& keypoints, torch::Tensor& scores,
                      torch::Tensor& descriptors){
    bm_tensor_t input_tensor;
    std::vector<bm_tensor_t> output_tensors;
    std::vector<bm_image> images{image};
    ts->save("superpoint preprocess time", batch_size);
    // int ret = 0;
    // static int i = 0;
    // std::ifstream input_data("input_data_"+std::to_string(i++)+".dat",std::ios::binary);
    // float *input = new float[360*640];
    // input_data.read((char*)input, 360*640*sizeof(float));
    // assert(true == bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]));
    // bm_memcpy_s2d(handle, input_tensor.device_mem, input);
    int ret = preprocess(images, input_tensor);
    assert(ret == 0);
    ts->save("superpoint preprocess time", batch_size);

    ts->save("superpoint inference time", batch_size);
    ret = forward(input_tensor, output_tensors);
    assert(ret == 0);
    ts->save("superpoint inference time", batch_size);

    ts->save("superpoint postprocess time", batch_size);
    ret = postprocess(output_tensors, keypoints, scores, descriptors);
    assert(ret == 0);
    ts->save("superpoint postprocess time", batch_size);
    
    return 0;
}

float SuperPoint::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w){
        *pIsAligWidth = true;
        ratio = r_w;
    }
    else{
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}

/**
 * @name    bm_image_align_width
 * @brief   create another bm_image which is width aligned by input bm_image.
 *
 * @param   [in]           bm_handle   device handle, should be created.
 * @param   [in]           input       input bm_image, will not be destroyed in this function.
 * @param   [in]           align_width default:64
 * @return  bm_image       image_aligned  
 */
bm_image SuperPoint::bm_image_align_width(bm_handle_t& handle, const bm_image& input, int align_width){
    bm_image image_aligned;
    int stride1[3], stride2[3];
    bm_image_get_stride(input, stride1);
    stride2[0] = FFALIGN(stride1[0], 64);
    stride2[1] = FFALIGN(stride1[1], 64);
    stride2[2] = FFALIGN(stride1[2], 64);
    bm_image_create(handle, input.height, input.width, input.image_format, input.data_type, &image_aligned, stride2);
    bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
    bmcv_copy_to_atrr_t copyToAttr;
    memset(&copyToAttr, 0, sizeof(copyToAttr));
    copyToAttr.start_x = 0;
    copyToAttr.start_y = 0;
    copyToAttr.if_padding = 1;
    bmcv_image_copy_to(handle, copyToAttr, input, image_aligned);
    return image_aligned;
}

int SuperPoint::preprocess(const std::vector<bm_image>& images, bm_tensor_t& input_tensor){
    int ret = 0;
    
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    m_resized_imgs.resize(batch_size);
    m_converto_imgs.resize(batch_size);

    //create bm_images
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, m_resized_imgs.data(), batch_size, strides);
    assert(BM_SUCCESS == ret);
    
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (netinfo->input_dtypes[0] == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    } else if (netinfo->input_dtypes[0] == BM_UINT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE;
    }
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_GRAY, img_dtype, m_converto_imgs.data(), batch_size, NULL, -1, false);
    assert(BM_SUCCESS == ret);

    // resize
    for(int i = 0; i < images.size(); ++i) {
        bm_image image1 = images[i];
        bm_image image_aligned;
        bool need_copy = image1.width & (64-1);
        if(need_copy){
            image_aligned = bm_image_align_width(handle, image1, 64);
        } else {
            image_aligned = image1;
        }
        auto ret = bmcv_image_vpp_convert(handle, 1, images[i], &m_resized_imgs[i]);
        assert(BM_SUCCESS == ret);
        
    #if 0
        cv::Mat resized_img;
        cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
        std::string fname = cv::format("resized_img_%d.jpg", i);
        cv::imwrite(fname, resized_img);
    #endif
        if(need_copy) bm_image_destroy(image_aligned);
    }
    
    // create tensor for converto_img to attach
    assert(true == bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]));
    bm_image_attach_contiguous_mem(batch_size, m_converto_imgs.data(), input_tensor.device_mem);

    // normalize
    float input_scale = netinfo->input_scales[0];
    input_scale = input_scale * 1.0 / 255.f;
    bmcv_convert_to_attr converto_attr;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = 0;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = 0;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = 0;
    ret = bmcv_image_convert_to(handle, images.size(), converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
    assert(ret == 0);

    // destroy bm_images
    bm_image_destroy_batch(m_resized_imgs.data(), batch_size);
#if BMCV_VERSION_MAJOR > 1
    bm_image_detach_contiguous_mem(batch_size, m_converto_imgs.data());
#else
    bm_image_dettach_contiguous_mem(batch_size, m_converto_imgs.data());
#endif
    bm_image_destroy_batch(m_converto_imgs.data(), batch_size, false);

    return 0;
}

int SuperPoint::forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors){
    output_tensors.resize(netinfo->output_num);
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                    output_tensors.data(), netinfo->output_num);
    assert(BM_SUCCESS == bm_thread_sync(handle));
    bm_free_device(handle, input_tensor.device_mem);
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
float* SuperPoint::get_cpu_data(bm_tensor_t* tensor, float scale){
    int ret = 0;
    float *pFP32 = NULL;
    int count = bmrt_shape_count(&tensor->shape);
    if(misc_info.pcie_soc_mode == 1){ //soc
        if (tensor->dtype == BM_FLOAT32) {
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            assert(BM_SUCCESS == ret);
            pFP32 = (float*)addr;
        } else if (BM_INT8 == tensor->dtype) {
            int8_t * pI8 = nullptr;
            unsigned long long  addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            assert(BM_SUCCESS == ret);
            pI8 = (int8_t*)addr;
            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pI8[i] * scale;
            }
            ret = bm_mem_unmap_device_mem(handle, pI8, bm_mem_get_device_size(tensor->device_mem));
            assert(BM_SUCCESS == ret);
        }  else if (BM_UINT8 == tensor->dtype) {
            uint8_t * pUI8 = nullptr;
            unsigned long long  addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            assert(BM_SUCCESS == ret);
            pUI8 = (uint8_t*)addr;
            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pUI8[i] * scale;
            }
            ret = bm_mem_unmap_device_mem(handle, pUI8, bm_mem_get_device_size(tensor->device_mem));
            assert(BM_SUCCESS == ret);
        } else{
            std::cerr << "unsupport dtype: " << tensor->dtype << std::endl;
        }
    } else { //pcie
        if (tensor->dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pFP32, tensor->device_mem, count * sizeof(float));
            assert(BM_SUCCESS ==ret);
        } else if (BM_INT8 == tensor->dtype) {
            int8_t * pI8 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(tensor);
            pI8 = new int8_t[tensor_size];
            assert(pI8 != nullptr);
            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pI8, tensor->device_mem, tensor_size);
            assert(BM_SUCCESS ==ret);
            for(int i = 0;i < count; ++ i) {
                pFP32[i] = pI8[i] * scale;
            }
            delete [] pI8;
        }  else if (BM_UINT8 == tensor->dtype) {
            uint8_t * pUI8 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(tensor);
            pUI8 = new uint8_t[tensor_size];
            assert(pUI8 != nullptr);
            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pUI8, tensor->device_mem, tensor_size);
            assert(BM_SUCCESS ==ret);
            for(int i = 0;i < count; ++ i) {
                pFP32[i] = pUI8[i] * scale;
            }
            delete [] pUI8;
        }else{
            std::cerr << "unsupport dtype: " << tensor->dtype << std::endl;
        }
    }
    return pFP32;
}


//源码对齐版本后处理：
torch::Tensor SuperPoint::simple_nms(torch::Tensor& scores, int r){
    assert(r >= 0);

    auto max_pool = [&](const torch::Tensor& x) {
        return torch::nn::functional::max_pool2d(x, 
            torch::nn::functional::MaxPool2dFuncOptions({r * 2 + 1, r * 2 + 1}).stride(1).padding(r)
        );
    };
    auto zeros = torch::zeros_like(scores, torch::device(torch::kCPU));
    auto max_mask = scores == max_pool(scores);
    for (int i = 0; i < 2; ++i) {
        auto supp_mask = max_pool(max_mask.to(torch::kFloat)) > 0;
        auto supp_scores = torch::where(supp_mask, zeros, scores);
        auto new_max_mask = supp_scores == max_pool(supp_scores);
        max_mask = max_mask | (new_max_mask & (~supp_mask));
    }
    return torch::where(max_mask, scores, zeros);
}

torch::Tensor SuperPoint::sample_descriptors(torch::Tensor keypoints, torch::Tensor descriptors, int s) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto b = descriptors.size(0);
    auto c = descriptors.size(1);
    auto h = descriptors.size(2);
    auto w = descriptors.size(3);

    keypoints = keypoints - s / 2.0 + 0.5;
    keypoints = keypoints / torch::tensor({(w * s - s / 2.0 - 0.5), (h * s - s / 2.0 - 0.5)}, options).to(keypoints.device()).unsqueeze(0);
    keypoints = keypoints * 2 - 1; // Normalize to (-1, 1)

    // Check torch version if necessary
    // For version >= 1.3.0, set align_corners to true
    bool align_corners = true; // Assuming LibTorch version is >= 1.3.0

    // Perform grid sampling
    descriptors = torch::nn::functional::grid_sample(descriptors, keypoints.view({b, 1, -1, 2}), torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(align_corners));

    // Normalize descriptors
    descriptors = torch::nn::functional::normalize(descriptors.reshape({b, c, -1}), torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    return descriptors;
}

int SuperPoint::postprocess(std::vector<bm_tensor_t>& output_tensors, torch::Tensor& keypoints, torch::Tensor& scores,
                      torch::Tensor& descriptors){
    float* desc;
    float* prob;
    // static int ii = 0; //debug
    std::map<std::string, bm_tensor_t> output_tensors_map;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    ts->save("superpoint get output time");  
    for(int i = 0; i < output_tensors.size(); i++){
        if(std::strcmp(netinfo->output_names[i], "descriptors_Div") == 0
        || std::strcmp(netinfo->output_names[i], "descriptors_Div_f32") == 0){
            desc = get_cpu_data(&output_tensors[i], netinfo->output_scales[i]);
            descriptors = torch::from_blob((void*)desc, {output_tensors[i].shape.dims[0], output_tensors[i].shape.dims[1], output_tensors[i].shape.dims[2],output_tensors[i].shape.dims[3]}, options).to(torch::kCPU);
            output_tensors_map["descriptors"] = output_tensors[i];
        } else if(std::strcmp(netinfo->output_names[i], "scores_Where") == 0
        || std::strcmp(netinfo->output_names[i], "scores_Where_f32") == 0){
            prob = get_cpu_data(&output_tensors[i], netinfo->output_scales[i]);
            scores = torch::from_blob((void*)prob, {output_tensors[i].shape.dims[0], output_tensors[i].shape.dims[1], output_tensors[i].shape.dims[2]}, options).to(torch::kCPU);
            output_tensors_map["scores"] = output_tensors[i];
        }
    }
    ts->save("superpoint get output time");  
    scores = scores.squeeze(0);
    
    //Extract keypoints
    keypoints = at::nonzero(scores > keypoint_threshold);

    if(keypoints.numel() > 0){
        auto k_transposed = keypoints.t();
        scores = scores.index({k_transposed[0], k_transposed[1]});
    }else{
        scores.reset();
        std::cout<<"No keypoints detected."<<std::endl;
        return 0;
    }

    //Discard keypoints near the image borders
    int remove_border = 4;
    auto mask_h = (keypoints.select(1, 0) >= remove_border) & (keypoints.select(1, 0) < (m_net_h - remove_border));
    auto mask_w = (keypoints.select(1, 1) >= remove_border) & (keypoints.select(1, 1) < (m_net_w - remove_border));
    auto mask = mask_h & mask_w;
    keypoints = keypoints.index_select(0, mask.nonzero().squeeze());
    scores = scores.index_select(0, mask.nonzero().squeeze());
    // Keep the k keypoints with highest score
    if(scores.size(0) > max_keypoint_size){
        auto topk_result = scores.topk((max_keypoint_size), 0);
        scores = std::get<0>(topk_result);
        auto topk_indices = std::get<1>(topk_result);
        keypoints = keypoints.index_select(0, topk_indices);
    }
    // Convert (h, w) to (x, y)
    auto keypoints_flipped = torch::stack({keypoints.index_select(1, torch::tensor({1})), keypoints.index_select(1, torch::tensor({0}))}, 1).to(torch::kFloat);
    keypoints = keypoints_flipped.squeeze(2);
    
    // Extract descriptors
    descriptors = sample_descriptors(keypoints, descriptors, 8);
    descriptors = descriptors.squeeze(0);

    if(misc_info.pcie_soc_mode == 1){ // soc
        if(output_tensors_map["descriptors"].dtype != BM_FLOAT32){
            delete [] desc;
        } else {
            int tensor_size = bm_mem_get_device_size(output_tensors_map["descriptors"].device_mem);
            bm_status_t ret = bm_mem_unmap_device_mem(handle, desc, tensor_size);
            assert(BM_SUCCESS == ret);
        }
        if(output_tensors_map["scores"].dtype != BM_FLOAT32){
            delete [] desc;
        } else {
            int tensor_size = bm_mem_get_device_size(output_tensors_map["scores"].device_mem);
            bm_status_t ret = bm_mem_unmap_device_mem(handle, prob, tensor_size);
            assert(BM_SUCCESS == ret);
        }
    } else {
        delete [] desc;
        delete [] prob;
    }
    for(int i = 0; i < output_tensors.size(); i++){
        bm_free_device(handle, output_tensors[i].device_mem);
    }
    return 0;
}