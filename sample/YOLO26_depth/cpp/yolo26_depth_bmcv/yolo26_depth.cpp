//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolo26_depth.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

int Yolo26_depth::Detect(const std::vector<bm_image>& input_images, std::vector<cv::Mat>& depth_maps) {
    assert(input_images.size() <= batch_size);
    int ret = 0;
    bm_tensor_t input_tensor;
    std::vector<bm_tensor_t> output_tensors;
    output_tensors.resize(netinfo->output_num);
    m_ts->save("yolo26_depth preprocess", input_images.size());
    ret = pre_process(input_images, input_tensor);
    assert(ret == 0);
    m_ts->save("yolo26_depth preprocess", input_images.size());

    m_ts->save("yolo26_depth inference", input_images.size());
    ret = forward(input_tensor, output_tensors);
    assert(ret == 0);
    // 与 Python 例程（sail SYSIO/SYSO 的 asnumpy 在 predict 内）同口径：
    // 输出深度图「设备 → 主机」的读回计入 inference。
    float* data_depth = get_cpu_data(&output_tensors[0], netinfo->output_scales[0]);
    m_ts->save("yolo26_depth inference", input_images.size());

    m_ts->save("yolo26_depth postprocess", input_images.size());
    ret = post_process(input_images, data_depth, output_tensors, depth_maps);
    assert(ret == 0);
    m_ts->save("yolo26_depth postprocess", input_images.size());
    return ret;
}

int Yolo26_depth::pre_process(const std::vector<bm_image>& images,
                              bm_tensor_t& input_tensor) {
    int ret = 0;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    m_resized_imgs.resize(batch_size);
    m_converto_imgs.resize(batch_size);

    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                m_resized_imgs.data(), batch_size, strides);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("bm_image_create_batch failed (resized)");
    }

    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (netinfo->input_dtypes[0] == BM_INT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    } else if (netinfo->input_dtypes[0] == BM_UINT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE;
    }
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype,
                                m_converto_imgs.data(), batch_size, NULL, -1, false);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("bm_image_create_batch failed (converto)");
    }

    int image_n = images.size();
    // stretch resize（无 letterbox padding，与官方 val 的 scale_fill=True 一致）
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
        auto ret = bmcv_image_vpp_convert(handle, 1, image_aligned, &m_resized_imgs[i]);
        if (ret != BM_SUCCESS) {
            throw std::runtime_error("stretch resize failed");
        }
        if (need_copy)
            bm_image_destroy(image_aligned);
    }

    // /255 -> float32, attach to input tensor
    ret = bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
    assert(true == ret);
    bm_image_attach_contiguous_mem(batch_size, m_converto_imgs.data(), input_tensor.device_mem);

    ret = bmcv_image_convert_to(handle, image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
    assert(ret == 0);

    bm_image_destroy_batch(m_resized_imgs.data(), batch_size);
#if BMCV_VERSION_MAJOR > 1
    bm_image_detach_contiguous_mem(batch_size, m_converto_imgs.data());
#else
    bm_image_dettach_contiguous_mem(batch_size, m_converto_imgs.data());
#endif
    bm_image_destroy_batch(m_converto_imgs.data(), batch_size, false);

    return ret;
}

int Yolo26_depth::forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors) {
    assert(netinfo->input_num == 1 && netinfo->output_num == 1);
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                                 output_tensors.data(), netinfo->output_num);
    assert(ok == true);
    auto ret = bm_thread_sync(handle);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("bm_thread_sync failed");
    }
    bm_free_device(handle, input_tensor.device_mem);
    return ret;
}

float* Yolo26_depth::get_cpu_data(bm_tensor_t* tensor, float scale) {
    int ret = 0;
    float* pFP32 = NULL;
    int count = bmrt_shape_count(&tensor->shape);
    if (misc_info.pcie_soc_mode == 1) {  // soc
        unsigned long long addr;
        ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
        if (ret != BM_SUCCESS) {
            throw std::runtime_error("bm_mem_mmap_device_mem failed");
        }
        ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
        if (ret != BM_SUCCESS) {
            throw std::runtime_error("bm_mem_invalidate_device_mem failed");
        }
        if (tensor->dtype == BM_FLOAT32) {
            pFP32 = (float*)addr;
        } else if (tensor->dtype == BM_INT8) {
            pFP32 = new float[count];
            int8_t* pI8 = (int8_t*)addr;
            for (int i = 0; i < count; ++i) pFP32[i] = pI8[i] * scale;
        } else if (tensor->dtype == BM_UINT8) {
            pFP32 = new float[count];
            uint8_t* pU8 = (uint8_t*)addr;
            for (int i = 0; i < count; ++i) pFP32[i] = pU8[i] * scale;
        } else {
            throw std::runtime_error("unsupported output dtype");
        }
    } else {  // pcie
        if (tensor->dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            ret = bm_memcpy_d2s_partial(handle, pFP32, tensor->device_mem, count * sizeof(float));
            assert(BM_SUCCESS == ret);
        } else if (tensor->dtype == BM_INT8) {
            int8_t* pI8 = new int8_t[count];
            ret = bm_memcpy_d2s_partial(handle, pI8, tensor->device_mem, count * sizeof(int8_t));
            assert(BM_SUCCESS == ret);
            pFP32 = new float[count];
            for (int i = 0; i < count; ++i) pFP32[i] = pI8[i] * scale;
            delete[] pI8;
        } else if (tensor->dtype == BM_UINT8) {
            uint8_t* pU8 = new uint8_t[count];
            ret = bm_memcpy_d2s_partial(handle, pU8, tensor->device_mem, count * sizeof(uint8_t));
            assert(BM_SUCCESS == ret);
            pFP32 = new float[count];
            for (int i = 0; i < count; ++i) pFP32[i] = pU8[i] * scale;
            delete[] pU8;
        } else {
            throw std::runtime_error("unsupported output dtype");
        }
    }
    return pFP32;
}

int Yolo26_depth::post_process(const std::vector<bm_image>& input_images,
                               float* data_depth,
                               std::vector<bm_tensor_t>& output_tensors,
                               std::vector<cv::Mat>& depth_maps) {
    // bmodel 直接输出 [batch, 1, net_h, net_w] float32 度量深度图（米），上采样与
    // cal_a/cal_b 标定已烘焙进图；输出分辨率 == 输入分辨率，stretch 预处理无 letterbox，
    // 因而深度图与官方 val 口径一致：直接在 net 分辨率逐像素比对，无需裁边/缩放。
    int oh = m_out_h;
    int ow = m_out_w;
    size_t plane = (size_t)oh * ow;

    for (int batch_idx = 0; batch_idx < (int)input_images.size(); ++batch_idx) {
        const float* src = data_depth + batch_idx * plane;
        cv::Mat depth(oh, ow, CV_32FC1);
        std::memcpy(depth.data, src, plane * sizeof(float));
        depth_maps.push_back(depth);
    }

    // free output tensor device memory
    bool mmaped = (misc_info.pcie_soc_mode == 1) && (output_tensors[0].dtype == BM_FLOAT32);
    if (misc_info.pcie_soc_mode == 1) {
        if (!mmaped) {
            delete[] data_depth;
        } else {
            int tensor_size = bm_mem_get_device_size(output_tensors[0].device_mem);
            bm_mem_unmap_device_mem(handle, data_depth, tensor_size);
        }
    } else {
        delete[] data_depth;
    }
    for (int i = 0; i < (int)output_tensors.size(); i++) {
        bm_free_device(handle, output_tensors[i].device_mem);
    }
    return 0;
}

cv::Mat Yolo26_depth::depth_to_vis(const cv::Mat& depth) {
    // 归一化到 [0, 95 分位]，inferno 伪彩色，与 python utils.depth_colorize 同口径
    int n = depth.rows * depth.cols;
    std::vector<float> vals;
    vals.reserve(n);
    for (int i = 0; i < n; ++i) {
        float v = depth.at<float>(i);
        if (v > 0.0f && std::isfinite(v)) vals.push_back(v);
    }
    float maxv = 1.0f;
    if (vals.size() > 100) {
        size_t k = (size_t)(vals.size() * 0.95);
        std::nth_element(vals.begin(), vals.begin() + k, vals.end());
        maxv = vals[k];
        if (maxv <= 0.0f) maxv = 1.0f;
    } else if (!vals.empty()) {
        maxv = *std::max_element(vals.begin(), vals.end());
    }
    cv::Mat norm;
    depth.convertTo(norm, CV_8UC1, 255.0 / maxv, 0.0);
    cv::Mat color;
    cv::applyColorMap(norm, color, cv::COLORMAP_INFERNO);
    return color;
}

void Yolo26_depth::save_depth_npy(const cv::Mat& depth, const std::string& path) {
    // 写最小 .npy（float32 小端，C 序），供 tools/eval_depth.py 读取
    // numpy v1.0 格式：magic(6) + version(2) + header_len(uint16 LE, 2) + header(空格补齐 + '\n')
    int h = depth.rows;
    int w = depth.cols;
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         std::to_string(h) + ", " + std::to_string(w) + "), }";
    // 前缀(magic+version+len=10) + header 补齐到 16 字节
    size_t pad = (16 - ((10 + header.size() + 1) % 16)) % 16;
    std::string full_header = header + std::string(pad, ' ') + "\n";
    uint16_t header_len = (uint16_t)full_header.size();

    std::ofstream ofs(path, std::ios::binary);
    ofs.write("\x93NUMPY", 6);
    ofs.write("\x01\x00", 2);
    ofs.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    ofs.write(full_header.c_str(), full_header.size());
    ofs.write(reinterpret_cast<const char*>(depth.data), (size_t)h * w * sizeof(float));
    ofs.close();
}

Yolo26_depth::Yolo26_depth(std::string bmodel_file, int dev_id) {
    auto ret = bm_dev_request(&handle, dev_id);
    assert(BM_SUCCESS == ret);

    ret = bm_get_misc_info(handle, &misc_info);
    assert(BM_SUCCESS == ret);

    bmrt = bmrt_create(handle);
    if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
        std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
    }

    const char** names;
    int num = bmrt_get_network_number(bmrt);
    if (num > 1) {
        std::cout << "This bmodel have " << num << " networks, and this program will only take network 0." << std::endl;
    }
    bmrt_get_network_names(bmrt, &names);
    for (int i = 0; i < num; ++i) {
        network_names.push_back(names[i]);
    }
    free(names);

    netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
    if (netinfo->stage_num > 1) {
        std::cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << std::endl;
    }
    batch_size = netinfo->stages[0].input_shapes[0].dims[0];
    m_net_h = netinfo->stages[0].input_shapes[0].dims[2];
    m_net_w = netinfo->stages[0].input_shapes[0].dims[3];

    // bmodel 输出 [batch, 1, net_h, net_w] 度量深度图（上采样已烘焙进图）
    assert(netinfo->output_num == 1);
    auto& shape = netinfo->stages[0].output_shapes[0];
    if (shape.num_dims != 4 || shape.dims[1] != 1) {
        throw std::runtime_error("Invalid model output shape, expect [N, 1, H, W] depth map.");
    }
    m_out_h = shape.dims[2];
    m_out_w = shape.dims[3];

    float input_scale = netinfo->input_scales[0] / 255.f;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = 0;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = 0;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = 0;

    m_ts = &tmp_ts;
}

Yolo26_depth::~Yolo26_depth() {
    if (bmrt != NULL) {
        bmrt_destroy(bmrt);
        bmrt = NULL;
    }
    bm_dev_free(handle);
}