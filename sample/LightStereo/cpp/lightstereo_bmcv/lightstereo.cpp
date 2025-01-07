//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include "lightstereo.hpp"

int LightStereo::pre_process(const std::vector<bm_image>& images, 
                            bm_tensor_t& input_tensor, bool is_left) {
    int ret = 0;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    m_resized_imgs.resize(batch_size);
    m_converto_imgs.resize(batch_size);

    //create bm_images
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, m_resized_imgs.data(), batch_size, strides);
    assert(BM_SUCCESS == ret);
    
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (netinfo->input_dtypes[0] == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    } else if (netinfo->input_dtypes[0] == BM_UINT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE;
    }
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), batch_size, NULL, -1, false);
    assert(BM_SUCCESS == ret);

    int image_n = images.size();
    // 1. resize image letterbox
    for (int i = 0; i < image_n; ++i) {
        bm_image image1 = images[i];
        if(image1.width <= m_net_w && image1.height <= m_net_h){
            bm_image image1_rgb_planar;
            ret = bm_image_create(handle, images[i].height, images[i].width, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &image1_rgb_planar);
            assert(BM_SUCCESS == ret);
            ret = bmcv_image_storage_convert(handle, 1, &image1, &image1_rgb_planar);
            assert(BM_SUCCESS == ret);

            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = m_net_h - image1.height;
            copyToAttr.if_padding = 1;
            copyToAttr.padding_r = 0;
            copyToAttr.padding_g = 0;
            copyToAttr.padding_b = 0;
            bmcv_image_copy_to(handle, copyToAttr, image1_rgb_planar, m_resized_imgs[i]);
            bm_image_destroy(image1_rgb_planar);
        }else{
            bm_image image_aligned;
            bool need_copy = image1.width & (64 - 1);
            if (need_copy) {
                int stride1[3], stride2[3];
                bm_image_get_stride(image1, stride1);
                stride2[0] = FFALIGN(stride1[0], 64);
                stride2[1] = FFALIGN(stride1[1], 64);
                stride2[2] = FFALIGN(stride1[2], 64);
                bm_image_create(handle, image1.height, image1.width, image1.image_format, image1.data_type,
                                &image_aligned, stride2);

                bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
                bmcv_copy_to_atrr_t copyToAttr;
                memset(&copyToAttr, 0, sizeof(copyToAttr));
                copyToAttr.start_x = 0;
                copyToAttr.start_y = 0;
                copyToAttr.if_padding = 1;
                bmcv_image_copy_to(handle, copyToAttr, image1, image_aligned);
            } else {
                image_aligned = image1;
            }
            auto ret = bmcv_image_vpp_convert(handle, 1, images[i], &m_resized_imgs[i]);
            assert(BM_SUCCESS == ret);
            if (need_copy)
                bm_image_destroy(image_aligned);
        }
    }

    // create tensor for converto_img to attach
    assert(true == bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]));
    bm_image_attach_contiguous_mem(batch_size, m_converto_imgs.data(), input_tensor.device_mem);

    auto converto_attr = is_left ? converto_attr_left : converto_attr_right;
    // 2. converto img
    ret = bmcv_image_convert_to(handle, image_n, converto_attr, m_resized_imgs.data(),
                                m_converto_imgs.data());
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

int LightStereo::forward(bm_tensor_t& input_tensor_left, bm_tensor_t& input_tensor_right, bm_tensor_t& output_tensor){
    assert(netinfo->input_num == 2 && netinfo->output_num == 1);
    bm_tensor_t input_tensors[2] = {input_tensor_left, input_tensor_right};
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, input_tensors, netinfo->input_num,
                    &output_tensor, netinfo->output_num);
    assert(ok == true);
    assert(BM_SUCCESS == bm_thread_sync(handle));
    bm_free_device(handle, input_tensor_left.device_mem);
    bm_free_device(handle, input_tensor_right.device_mem);
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
float* LightStereo::get_cpu_data(bm_tensor_t* tensor, float scale){
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

std::vector<cv::Mat> LightStereo::process(const std::vector<bm_image>& left_imgs, const std::vector<bm_image>& right_imgs){
    assert(left_imgs.size() <= batch_size);
    assert(left_imgs.size() <= right_imgs.size());
    for(int i = 0; i < left_imgs.size(); i++){
        assert(left_imgs[i].width == right_imgs[i].width);
        assert(left_imgs[i].height == right_imgs[i].height);
    }
    int ret = 0;
    bm_tensor_t input_tensor_left, input_tensor_right, output_tensor;
    m_ts->save("LightStereo preprocess", left_imgs.size());
    ret = pre_process(left_imgs, input_tensor_left, true);
    assert(ret == 0);
    ret = pre_process(right_imgs, input_tensor_right, false);
    assert(ret == 0);
    m_ts->save("LightStereo preprocess", left_imgs.size());

    m_ts->save("LightStereo inference", left_imgs.size());
    ret = forward(input_tensor_left, input_tensor_right, output_tensor);
    assert(ret == 0);
    m_ts->save("LightStereo inference", left_imgs.size());

    m_ts->save("LightStereo postprocess", left_imgs.size());
    float* tensor_data = get_cpu_data(&output_tensor, netinfo->output_scales[0]); //will be freed in the end of postprocess.
    std::vector<cv::Mat> output_mats;
    for(int i = 0; i < left_imgs.size(); i++){
        cv::Mat float_mat(m_net_h, m_net_w, CV_32FC1, tensor_data + i * m_net_h * m_net_w);
        if(left_imgs[i].width <= m_net_w && left_imgs[i].height <= m_net_h){
            cv::Rect bound = cv::Rect{0, m_net_h - left_imgs[i].height, left_imgs[i].width, left_imgs[i].height};
            output_mats.push_back(float_mat(bound).clone());
        }else{
            cv::Mat resized(left_imgs[i].height, left_imgs[i].width, CV_32FC1, cv::SophonDevice(m_dev_id));
            cv::resize(float_mat, resized, cv::Size(left_imgs[i].width, left_imgs[i].height));
            output_mats.push_back(resized);
        }
    }
    m_ts->save("LightStereo postprocess", left_imgs.size());
    if(misc_info.pcie_soc_mode == 1){ // soc
        if(output_tensor.dtype != BM_FLOAT32){
            delete [] tensor_data;
        } else {
            int tensor_size = bm_mem_get_device_size(output_tensor.device_mem);
            bm_status_t ret = bm_mem_unmap_device_mem(handle, tensor_data, tensor_size);
            assert(BM_SUCCESS == ret);
        }
        if(output_tensor.dtype != BM_FLOAT32){
            delete [] tensor_data;
        } else {
            int tensor_size = bm_mem_get_device_size(output_tensor.device_mem);
            bm_status_t ret = bm_mem_unmap_device_mem(handle, tensor_data, tensor_size);
            assert(BM_SUCCESS == ret);
        }
    } else {
        delete [] tensor_data;
    }
    bm_free_device(handle, output_tensor.device_mem);
    return output_mats;
}
