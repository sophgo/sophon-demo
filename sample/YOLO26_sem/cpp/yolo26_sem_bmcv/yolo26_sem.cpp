//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolo26_sem.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <climits>
#include <stdexcept>
#define USE_ASPECT_RATIO 1

int Yolo26_sem::Detect(const std::vector<bm_image>& input_images, std::vector<cv::Mat>& segmaps) {
    assert(input_images.size() <= batch_size);
    int ret = 0;
    bm_tensor_t input_tensor;
    std::vector<bm_tensor_t> output_tensors;
    output_tensors.resize(netinfo->output_num);
    std::vector<std::pair<float, float>> ratios_batch;
    m_ts->save("yolo26_sem preprocess", input_images.size());
    ret = pre_process(input_images, input_tensor, ratios_batch);
    assert(ret == 0);
    m_ts->save("yolo26_sem preprocess", input_images.size());

    m_ts->save("yolo26_sem inference", input_images.size());
    ret = forward(input_tensor, output_tensors);
    assert(ret == 0);
    // 与 Python 例程（sail SYSIO/SYSO 的 asnumpy 在 predict 内）同口径：
    // 输出类别图「设备 → 主机」的读回计入 inference，postprocess 只做主机侧的
    // int32→uint8 转换 + 去 padding + 缩放。
    int32_t* data_seg = get_cpu_data(&output_tensors[0]);
    m_ts->save("yolo26_sem inference", input_images.size());

    m_ts->save("yolo26_sem postprocess", input_images.size());
    ret = post_process(input_images, data_seg, output_tensors, ratios_batch, segmaps);
    assert(ret == 0);
    m_ts->save("yolo26_sem postprocess", input_images.size());
    return ret;
}

float Yolo26_sem::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w) {
        *pIsAligWidth = true;
        return r_w;
    } else {
        *pIsAligWidth = false;
        return r_h;
    }
}

int Yolo26_sem::pre_process(const std::vector<bm_image>& images,
                            bm_tensor_t& input_tensor,
                            std::vector<std::pair<float, float>>& ratios_batch) {
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
    // 1. letterbox resize (pad 114)
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
            padding_attr.dst_crop_sty = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
            padding_attr.dst_crop_stx = 0;
        } else {
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = images[i].width * ratio;
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
        }
        ratios_batch.push_back(std::make_pair(ratio, ratio));
        bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
        auto ret = bmcv_image_vpp_convert_padding(handle, 1, image_aligned, &m_resized_imgs[i],
                                                  &padding_attr, &crop_rect);
#else
        auto ret = bmcv_image_vpp_convert(handle, 1, images[i], &m_resized_imgs[i]);
        ratios_batch.push_back(std::make_pair((float)m_net_w / images[i].width,
                                              (float)m_net_h / images[i].height));
#endif
        if (ret != BM_SUCCESS) {
            throw std::runtime_error("letterbox resize failed");
        }
        if (need_copy)
            bm_image_destroy(image_aligned);
    }

    // 2. /255 -> float32, attach to input tensor
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

int Yolo26_sem::forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors) {
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

int32_t* Yolo26_sem::get_cpu_data(bm_tensor_t* tensor) {
    // bmodel 输出为 int32 类别图 [batch, 1, net_h, net_w]，直接按 int32 读回
    int ret = 0;
    int32_t* p = NULL;
    int bytes = bmrt_tensor_bytesize(tensor);
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
        p = (int32_t*)addr;
    } else {  // pcie
        p = new int32_t[bytes / sizeof(int32_t)];
        ret = bm_memcpy_d2s_partial(handle, p, tensor->device_mem, bytes);
        assert(BM_SUCCESS == ret);
    }
    return p;
}

int Yolo26_sem::post_process(const std::vector<bm_image>& input_images,
                             int32_t* data_seg,
                             std::vector<bm_tensor_t>& output_tensors,
                             const std::vector<std::pair<float, float>>& ratios_batch,
                             std::vector<cv::Mat>& segmaps) {
    // bmodel 直接输出 [batch, 1, net_h, net_w] int32 类别图（bilinear 上采样 + argmax
    // 已烘焙进图，与 ultralytics 后处理顺序一致：先上采样 logits 再 argmax）。
    // data_seg 已在 inference 阶段读回主机（get_cpu_data），此处只做主机侧后处理。
    int oh = m_out_h;  // == m_net_h（1024）
    int ow = m_out_w;  // == m_net_w（2048）
    size_t plane = (size_t)oh * ow;

    for (int batch_idx = 0; batch_idx < (int)input_images.size(); ++batch_idx) {
        int frame_w = input_images[batch_idx].width;
        int frame_h = input_images[batch_idx].height;

        // int32 类别图 -> uint8 cv::Mat
        const int32_t* src = data_seg + batch_idx * plane;
        cv::Mat classmap(oh, ow, CV_8UC1);
        uint8_t* dst = classmap.data;
        for (size_t i = 0; i < plane; ++i) {
            dst[i] = (uint8_t)src[i];
        }

        // 去掉 letterbox padding（与 pre_process 的缩放比一致）
        float r = ratios_batch[batch_idx].first;
        int scaled_w = (int)std::round(r * frame_w);
        int scaled_h = (int)std::round(r * frame_h);
        int x1 = (m_net_w - scaled_w) / 2;
        int y1 = (m_net_h - scaled_h) / 2;
        cv::Mat crop = classmap(cv::Rect(x1, y1, scaled_w, scaled_h));

        // 缩回原图尺寸
        cv::Mat out;
        cv::resize(crop, out, cv::Size(frame_w, frame_h), 0, 0, cv::INTER_NEAREST);
        segmaps.push_back(out);
    }

    // free output tensor device memory
    if (misc_info.pcie_soc_mode == 1) {  // soc
        int tensor_size = bm_mem_get_device_size(output_tensors[0].device_mem);
        bm_mem_unmap_device_mem(handle, data_seg, tensor_size);
    } else {
        delete[] data_seg;
    }
    for (int i = 0; i < (int)output_tensors.size(); i++) {
        bm_free_device(handle, output_tensors[i].device_mem);
    }
    return 0;
}

void Yolo26_sem::draw_result(cv::Mat& img, const cv::Mat& segmap, cv::Mat& blend_out) {
    cv::Mat color_seg(segmap.rows, segmap.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    int nc = (int)kCityscapesPalette.size();
    for (int y = 0; y < segmap.rows; ++y) {
        for (int x = 0; x < segmap.cols; ++x) {
            int c = segmap.at<uint8_t>(y, x);
            if (c >= 0 && c < nc) {
                auto& col = kCityscapesPalette[c];
                color_seg.at<cv::Vec3b>(y, x) = cv::Vec3b(col[0], col[1], col[2]);
            }
        }
    }
    cv::addWeighted(img, 0.5, color_seg, 0.5, 0, blend_out);
}

Yolo26_sem::Yolo26_sem(std::string bmodel_file, int dev_id) {
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

    // bmodel 输出 int32 类别图 [1, 1, net_h, net_w]（bilinear 上采样 + argmax 已烘焙进图）
    assert(netinfo->output_num == 1);
    auto& shape = netinfo->stages[0].output_shapes[0];
    if (shape.num_dims != 4 || shape.dims[1] != 1) {
        throw std::runtime_error("Invalid model output shape, expect [N, 1, H, W] class map.");
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

Yolo26_sem::~Yolo26_sem() {
    if (bmrt != NULL) {
        bmrt_destroy(bmrt);
        bmrt = NULL;
    }
    bm_dev_free(handle);
}