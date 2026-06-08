//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "arcface.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

int ArcFace::Embed(const std::vector<bm_image>& input_images, std::vector<std::vector<float>>& embeddings) {
    assert(input_images.size() <= (size_t)batch_size);
    int ret = 0;
    bm_tensor_t input_tensor, output_tensor;
    m_ts->save("arcface preprocess", input_images.size());
    ret = pre_process(input_images, input_tensor);
    assert(ret == 0);
    m_ts->save("arcface preprocess", input_images.size());

    m_ts->save("arcface inference", input_images.size());
    ret = forward(input_tensor, output_tensor);
    assert(ret == 0);
    m_ts->save("arcface inference", input_images.size());

    m_ts->save("arcface postprocess", input_images.size());
    ret = post_process(input_images, output_tensor, embeddings);
    assert(ret == 0);
    m_ts->save("arcface postprocess", input_images.size());
    return ret;
}

int ArcFace::pre_process(const std::vector<bm_image>& images,
                         bm_tensor_t& input_tensor) {
    int ret = 0;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    m_resized_imgs.resize(batch_size);
    m_converto_imgs.resize(batch_size);

    // create bm_images for resize
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                                DATA_TYPE_EXT_1N_BYTE, m_resized_imgs.data(), batch_size, strides);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime error creating resized images");
    }

    // create bm_images for convert_to (float32)
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (netinfo->input_dtypes[0] == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    } else if (netinfo->input_dtypes[0] == BM_UINT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE;
    }
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                                img_dtype, m_converto_imgs.data(), batch_size, NULL, -1, false);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime error creating converto images");
    }

    int image_n = images.size();
    // 1. resize directly to 112x112 (no letterbox for ArcFace)
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
            bm_image_create(handle, image1.height, image1.width,
                            image1.image_format, image1.data_type, &image_aligned, stride2);
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

        auto ret = bmcv_image_vpp_convert(handle, 1, image_aligned, &m_resized_imgs[i]);
        if (ret != BM_SUCCESS) {
            throw std::runtime_error("BMRuntime error in resize");
        }
        if (need_copy) bm_image_destroy(image_aligned);
    }

    // create tensor for converto_img to attach
    ret = bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
    assert(true == ret);
    bm_image_attach_contiguous_mem(batch_size, m_converto_imgs.data(), input_tensor.device_mem);

    // 2. convert to float32 (mean/scale handled by bmodel)
    for (int i = 0; i < image_n; ++i) {
        ret = bmcv_image_convert_to(handle, 1, converto_attr,
                                    &m_resized_imgs[i], &m_converto_imgs[i]);
        assert(ret == 0);
    }

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

int ArcFace::forward(bm_tensor_t& input_tensor, bm_tensor_t& output_tensor){
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                    &output_tensor, netinfo->output_num);
    assert(ok == true);
    auto ret = bm_thread_sync(handle);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime error in forward");
    }
    bm_free_device(handle, input_tensor.device_mem);
    return 0;
}

float* ArcFace::get_cpu_data(bm_tensor_t* tensor, float scale){
    int ret = 0;
    float *pFP32 = NULL;
    int count = bmrt_shape_count(&tensor->shape);
    if(misc_info.pcie_soc_mode == 1){ // soc
        if (tensor->dtype == BM_FLOAT32) {
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error mmap");
            }
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error invalidate");
            }
            pFP32 = (float*)addr;
        } else if (BM_INT8 == tensor->dtype) {
            int8_t * pI8 = nullptr;
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error mmap");
            }
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error invalidate");
            }
            pI8 = (int8_t*)addr;
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pI8[i] * scale;
            }
            ret = bm_mem_unmap_device_mem(handle, pI8, bm_mem_get_device_size(tensor->device_mem));
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error unmap");
            }
        } else if (BM_UINT8 == tensor->dtype) {
            uint8_t * pUI8 = nullptr;
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error mmap");
            }
            ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error invalidate");
            }
            pUI8 = (uint8_t*)addr;
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pUI8[i] * scale;
            }
            ret = bm_mem_unmap_device_mem(handle, pUI8, bm_mem_get_device_size(tensor->device_mem));
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error unmap");
            }
        } else{
            std::cerr << "unsupport dtype: " << tensor->dtype << std::endl;
        }
    } else { // pcie
        if (tensor->dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pFP32, tensor->device_mem, count * sizeof(float));
            assert(BM_SUCCESS == ret);
        } else if (BM_INT8 == tensor->dtype) {
            int8_t * pI8 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(tensor);
            pI8 = new int8_t[tensor_size];
            assert(pI8 != nullptr);
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pI8, tensor->device_mem, tensor_size);
            assert(BM_SUCCESS == ret);
            for(int i = 0; i < count; ++ i) {
                pFP32[i] = pI8[i] * scale;
            }
            delete [] pI8;
        } else if (BM_UINT8 == tensor->dtype) {
            uint8_t * pUI8 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(tensor);
            pUI8 = new uint8_t[tensor_size];
            assert(pUI8 != nullptr);
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(handle, pUI8, tensor->device_mem, tensor_size);
            assert(BM_SUCCESS == ret);
            for(int i = 0; i < count; ++ i) {
                pFP32[i] = pUI8[i] * scale;
            }
            delete [] pUI8;
        } else{
            std::cerr << "unsupport dtype: " << tensor->dtype << std::endl;
        }
    }
    return pFP32;
}

int ArcFace::post_process(const std::vector<bm_image>& input_images,
                          bm_tensor_t& output_tensor,
                          std::vector<std::vector<float>>& embeddings) {
    embeddings.clear();
    float* output_data = get_cpu_data(&output_tensor, netinfo->output_scales[0]);

    for(unsigned int batch_idx = 0; batch_idx < input_images.size(); ++ batch_idx) {
        std::vector<float> emb(output_size_);
        memcpy(emb.data(), output_data + batch_idx * output_size_, output_size_ * sizeof(float));

        // L2 normalize
        float sum_sq = 0.0f;
        for (int j = 0; j < output_size_; j++) {
            sum_sq += emb[j] * emb[j];
        }
        float norm = std::sqrt(sum_sq) + 1e-10f;
        for (int j = 0; j < output_size_; j++) {
            emb[j] /= norm;
        }
        embeddings.push_back(emb);
    }

    if(misc_info.pcie_soc_mode == 1){ // soc
        if(output_tensor.dtype != BM_FLOAT32){
            delete [] output_data;
        } else {
            int tensor_size = bm_mem_get_device_size(output_tensor.device_mem);
            bm_status_t ret = bm_mem_unmap_device_mem(handle, output_data, tensor_size);
            if (ret != BM_SUCCESS) {
                throw std::runtime_error("BMRuntime error unmap");
            }
        }
    } else {
        delete [] output_data;
    }
    bm_free_device(handle, output_tensor.device_mem);
    return 0;
}
