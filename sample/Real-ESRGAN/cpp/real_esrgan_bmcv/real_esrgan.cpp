//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "real_esrgan.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#define USE_ASPECT_RATIO 1

int Real_ESRGAN::process(const std::vector<bm_image>& input_images, std::vector<cv::Mat>& output_mats) {
    assert(input_images.size() <= batch_size);
    int ret = 0;
    bm_tensor_t input_tensor, output_tensor;
    m_ts->save("Real_ESRGAN preprocess", input_images.size());
    ret = pre_process(input_images, input_tensor);
    assert(ret == 0);
    m_ts->save("Real_ESRGAN preprocess", input_images.size());

    m_ts->save("Real_ESRGAN inference", input_images.size());
    ret = forward(input_tensor, output_tensor);
    assert(ret == 0);
    m_ts->save("Real_ESRGAN inference", input_images.size());

    m_ts->save("Real_ESRGAN postprocess", input_images.size());
    ret = post_process(input_images, output_tensor, output_mats);
    assert(ret == 0);
    m_ts->save("Real_ESRGAN postprocess", input_images.size());
    return ret;
}

float Real_ESRGAN::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w) {
        *pIsAligWidth = true;
        ratio = r_w;
    } else {
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}

int Real_ESRGAN::pre_process(const std::vector<bm_image>& images, bm_tensor_t& input_tensor) {
    int ret = 0;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    m_resized_imgs.resize(batch_size);
    m_converto_imgs.resize(batch_size);

    //create bm_images
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, m_input_format, DATA_TYPE_EXT_1N_BYTE, m_resized_imgs.data(), batch_size, strides);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (netinfo->input_dtypes[0] == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    } else if (netinfo->input_dtypes[0] == BM_UINT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE;
    }
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, m_input_format, img_dtype, m_converto_imgs.data(), batch_size, NULL, -1, false);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }

    int image_n = images.size();
    // 1. resize image letterbox
    for (int i = 0; i < image_n; ++i) {
        bm_image image1 = images[i];
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
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height, m_net_w, m_net_h, &isAlignWidth);
        int tx1 = 0, ty1 = 0;
        bmcv_padding_atrr_t padding_attr;
        memset(&padding_attr, 0, sizeof(padding_attr));
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        padding_attr.padding_b = 114;
        padding_attr.padding_g = 114;
        padding_attr.padding_r = 114;
        padding_attr.if_memset = 1;
        if (isAlignWidth) {
            padding_attr.dst_crop_h = images[i].height * ratio;
            padding_attr.dst_crop_w = m_net_w;

            ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);  // padding 大小
            padding_attr.dst_crop_sty = ty1;
            padding_attr.dst_crop_stx = 0;
        } else {
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = images[i].width * ratio;

            tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = tx1;
        }
        bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
        auto ret = bmcv_image_vpp_convert_padding(handle, 1, image_aligned, &m_resized_imgs[i],
                                                  &padding_attr, &crop_rect);
#else
        auto ret = bmcv_image_vpp_convert(handle, 1, images[i], &m_resized_imgs[i]);
        txy_batch.push_back(std::make_pair(0, 0));
        ratios_batch.push_back(std::make_pair((float)m_net_w/images[i].width,(float)m_net_h/images[i].height));
#endif
        if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
        if (need_copy)
            bm_image_destroy(image_aligned);
    }

    // create tensor for converto_img to attach
    ret = bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
    assert(true == ret);
    bm_image_attach_contiguous_mem(batch_size, m_converto_imgs.data(), input_tensor.device_mem);

    // 2. converto img /= 255
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

int Real_ESRGAN::forward(bm_tensor_t& input_tensor, bm_tensor_t& output_tensor){
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                    &output_tensor, netinfo->output_num);
    assert(ok == true);
    auto ret = bm_thread_sync(handle);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
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
float* Real_ESRGAN::get_cpu_data(bm_tensor_t* tensor, float scale){
    int ret = 0;
    float *pFP32 = NULL;
    int count = bmrt_shape_count(&tensor->shape);
    if(misc_info.pcie_soc_mode == 1){ //soc
        if (tensor->dtype == BM_FLOAT32) {
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
            pFP32 = (float*)addr;
        } else if (BM_INT8 == tensor->dtype) {
            int8_t * pI8 = nullptr;
            unsigned long long  addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
            pI8 = (int8_t*)addr;
            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pI8[i] * scale;
            }
            ret = bm_mem_unmap_device_mem(handle, pI8, bm_mem_get_device_size(tensor->device_mem));
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
        }  else if (BM_UINT8 == tensor->dtype) {
            uint8_t * pUI8 = nullptr;
            unsigned long long  addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
            pUI8 = (uint8_t*)addr;
            // dtype convert
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pUI8[i] * scale;
            }
            ret = bm_mem_unmap_device_mem(handle, pUI8, bm_mem_get_device_size(tensor->device_mem));
            if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
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

int Real_ESRGAN::post_process(const std::vector<bm_image>& input_images, 
                             bm_tensor_t& output_tensor, 
                             std::vector<cv::Mat>& output_mats){
    int ret = 0;
    if(netinfo->output_dtypes[0] == BM_UINT8){
        ret = post_process_bmcv(input_images, output_tensor, output_mats);
    }else{
        ret = post_process_opencv(input_images, output_tensor, output_mats);
    }
    return ret;
}

int Real_ESRGAN::post_process_bmcv(const std::vector<bm_image>& input_images, 
                             bm_tensor_t& output_tensor, 
                             std::vector<cv::Mat>& output_mats) {
    int ret = 0;
    int output_h = output_tensor.shape.dims[2];
    int output_w = output_tensor.shape.dims[3];
    int upsample_scale = output_w/m_net_w;
    
    bm_image raw_outputs[batch_size];
    ret = bm_image_create_batch(handle, output_h, output_w, m_input_format, 
                                DATA_TYPE_EXT_1N_BYTE, raw_outputs, batch_size, NULL, -1, false);
    assert(ret == BM_SUCCESS);
    ret = bm_image_attach_contiguous_mem(batch_size, raw_outputs, output_tensor.device_mem);
    assert(ret == BM_SUCCESS);
    for(int i = 0; i < batch_size; i++){
        float tx1 = 0, ty1 = 0;
    #if USE_ASPECT_RATIO
        float r_w = (float)m_net_w / input_images[i].width;
        float r_h = (float)m_net_h / input_images[i].height;
        if (r_h > r_w) {
            int th = (int)(r_w * input_images[i].height);
            ty1 = (m_net_h - th) / 2.0f;
        } else {
            int tw = (int)(r_h * input_images[i].width);
            tx1 = (m_net_w - tw) / 2.0f;
        }
    #endif
        cv::Mat output_image;
        output_image.create(cv::Size(output_w, output_h), m_input_format == FORMAT_GRAY ? CV_8UC1 : CV_8UC3, m_dev_id);
        cv::bmcv::toMAT(&raw_outputs[i], output_image);

        // 根据tx1和ty1裁剪图像
        if (tx1 != 0) {
            int tx = tx1 * upsample_scale; // 调整比例
            output_image = output_image(cv::Rect(tx, 0, output_image.cols - 2*tx, output_image.rows));
        } else if (ty1 != 0) {
            int ty = ty1 * upsample_scale; // 调整比例
            output_image = output_image(cv::Rect(0, ty, output_image.cols, output_image.rows - 2*ty));
        }
        output_mats.push_back(output_image);
    }
    ret = bm_image_dettach_contiguous_mem(batch_size, raw_outputs);
    assert(ret == BM_SUCCESS);
    ret = bm_image_destroy_batch(raw_outputs, batch_size, false);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    bm_free_device(handle, output_tensor.device_mem);
    return 0;
}

int Real_ESRGAN::post_process_opencv(const std::vector<bm_image>& input_images, 
                             bm_tensor_t& output_tensor, 
                             std::vector<cv::Mat>& output_mats) {
    int output_w = output_tensor.shape.dims[3];
    int upsample_scale = output_w/m_net_w;
    output_mats.clear();
    float* output_data = get_cpu_data(&output_tensor, netinfo->output_scales[0]);
    for(int batch_idx = 0; batch_idx < input_images.size(); ++ batch_idx)
    {
        auto& frame = input_images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;
        float tx1 = 0, ty1 = 0;
    #if USE_ASPECT_RATIO
        bool is_align_width = false;
        float r_w = (float)m_net_w / frame.width;
        float r_h = (float)m_net_h / frame.height;
        if (r_h > r_w) {
            int th = (int)(r_w * frame.height);
            ty1 = (m_net_h - th) / 2.0f;
        } else {
            int tw = (int)(r_h * frame.width);
            tx1 = (m_net_w - tw) / 2.0f;
        }
    #endif

        int height = output_tensor.shape.dims[2];
        int width = output_tensor.shape.dims[3];
        int channel  = output_tensor.shape.dims[1];

        float* cpu_data = output_data + batch_idx*height*width*channel;
        cv::Mat output_image;
        if(m_input_format == FORMAT_RGB_PLANAR){
            cv::Mat channelB(height, width, CV_8UC1);
            cv::Mat channelG(height, width, CV_8UC1);
            cv::Mat channelR(height, width, CV_8UC1);
            uchar* ptr_channelR = channelR.data;
            uchar* ptr_channelG = channelG.data;
            uchar* ptr_channelB = channelB.data;
            int area = height * width;

            cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = y * width + x;
                    ptr_channelR[idx] = cv::saturate_cast<uchar>(cpu_data[idx] * 255.0f);
                    ptr_channelG[idx] = cv::saturate_cast<uchar>(cpu_data[idx + area] * 255.0f);
                    ptr_channelB[idx] = cv::saturate_cast<uchar>(cpu_data[idx + 2 * area] * 255.0f);
                }
            }});
            
            std::vector<cv::Mat> bgr_channels = {channelB, channelG, channelR};
            cv::merge(bgr_channels, output_image);
        }else{ // gray
            cv::Mat output_image_f32(height, width, CV_32FC1, cpu_data);
            output_image_f32.convertTo(output_image, CV_8UC1, 255.0);
        }

        
        // 根据tx1和ty1裁剪图像
        if (tx1 != 0) {
            int tx = tx1 * upsample_scale; // 调整比例
            output_image = output_image(cv::Rect(tx, 0, output_image.cols - 2*tx, output_image.rows));
        } else if (ty1 != 0) {
            int ty = ty1 * upsample_scale; // 调整比例
            output_image = output_image(cv::Rect(0, ty, output_image.cols, output_image.rows - 2*ty));
        }
        output_mats.push_back(output_image);
    }
    if(misc_info.pcie_soc_mode == 1){ // soc
        if(output_tensor.dtype != BM_FLOAT32){
            delete [] output_data;
        } else {
            int tensor_size = bm_mem_get_device_size(output_tensor.device_mem);
            bm_status_t ret = bm_mem_unmap_device_mem(handle, output_data, tensor_size);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime 操作失败");
            }
        }
    } else {
        delete [] output_data;
    }
    bm_free_device(handle, output_tensor.device_mem);
    return 0;
}