//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov8_obb.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1
#define USE_NEON 0

double pythonRound(double number) {
    double integer_part = 0.0;
    double fractional_part = std::modf(number, &integer_part);

    if (fractional_part > 0.5 || (fractional_part == 0.5 && fmod(integer_part, 2.0) == 1.0)) {
        integer_part += 1.0;
    }

    return integer_part;
}

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

int YoloV8_obb::Detect(const std::vector<bm_image>& input_images, std::vector<obbBoxVec_xyxy>& boxes) {
    assert(input_images.size() <= batch_size);
    int ret = 0;
    bm_tensor_t input_tensor, output_tensor;
    std::vector<std::pair<int, int>> txy_batch;
    std::vector<float> ratios_batch;
    m_ts->save("yolov8 preprocess", input_images.size());
    ret = pre_process(input_images, input_tensor, txy_batch, ratios_batch);
    assert(ret == 0);
    m_ts->save("yolov8 preprocess", input_images.size());

    m_ts->save("yolov8 inference", input_images.size());
    ret = forward(input_tensor, output_tensor);
    assert(ret == 0);
    m_ts->save("yolov8 inference", input_images.size());

    m_ts->save("yolov8 postprocess", input_images.size());
    ret = post_process(input_images, output_tensor, txy_batch, ratios_batch, boxes);
    assert(ret == 0);
    m_ts->save("yolov8 postprocess", input_images.size());
    return ret;
}

float YoloV8_obb::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

int YoloV8_obb::pre_process(const std::vector<bm_image>& images, 
                            bm_tensor_t& input_tensor, 
                            std::vector<std::pair<int, int>>& txy_batch, 
                            std::vector<float>& ratios_batch) {
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
        txy_batch.push_back(std::make_pair(tx1, ty1));
        ratios_batch.push_back(ratio);
        bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
        auto ret = bmcv_image_vpp_convert_padding(handle, 1, image_aligned, &m_resized_imgs[i],
                                                  &padding_attr, &crop_rect);
#else
        auto ret = bmcv_image_vpp_convert(handle, 1, images[i], &m_resized_imgs[i]);
        txy_batch.push_back(std::make_pair(0, 0));
        ratios_batch.push_back(1.0);
#endif
        assert(BM_SUCCESS == ret);
        if (need_copy)
            bm_image_destroy(image_aligned);
    }

    // create tensor for converto_img to attach
    assert(true == bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]));
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

int YoloV8_obb::forward(bm_tensor_t& input_tensor, bm_tensor_t& output_tensor){
    // static int count = 0;
    // std::ifstream input_data("../../python/dummy_inputs/"+std::to_string(count++)+".bin", std::ios::binary);
    // static float *input = new float[3*1024*1024];
    // input_data.read((char*)input, 3*1024*1024*sizeof(float));
    // bm_memcpy_s2d(handle, input_tensor.device_mem, input);

    assert(netinfo->input_num == 1 && netinfo->output_num == 1);
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                    &output_tensor, netinfo->output_num);
    assert(ok == true);
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
float* YoloV8_obb::get_cpu_data(bm_tensor_t* tensor, float scale){
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


int YoloV8_obb::post_process(const std::vector<bm_image>& input_images, 
                             bm_tensor_t& output_tensor, 
                             const std::vector<std::pair<int, int>>& txy_batch, 
                             const std::vector<float>& ratios_batch, 
                             std::vector<obbBoxVec_xyxy>& detected_boxes) {
    float* tensor_data = get_cpu_data(&output_tensor, netinfo->output_scales[0]); //will be freed in the end of postprocess.
    for (int batch_idx = 0; batch_idx < input_images.size(); ++batch_idx) {
        obbBoxVec yolobox_vec;
        auto& frame = input_images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int box_num = output_tensor.shape.dims[1];
        int nout = output_tensor.shape.dims[2];
        float* output_data = NULL;

        output_data = tensor_data + batch_idx * box_num * nout; //output_tensor: [bs, box_num, class_num + 5]

        // Candidates
        LOG_TS(m_ts, "yolov8_obb_post: get box");
        float* cls_conf = output_data + 4; //output_tensor's last dim: [x, y, w, h, cls_conf0, ..., cls_conf14, rotate_angle]
        for (int i = 0; i < box_num; i++) {
#if USE_MULTICLASS_NMS
            // multilabel
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[i * nout + j];
                if (cur_value > m_confThreshold) {
                    obbBox box;
                    box.score = cur_value;
                    box.class_id = j;
                    int c = agnostic ? 0 : box.class_id * max_wh;
                    box.x = output_data[i * nout + 0] + c;
                    box.y = output_data[i * nout + 1] + c;
                    box.w = output_data[i * nout + 2];
                    box.h = output_data[i * nout + 3];
                    box.angle = output_data[(i + 1) * nout - 1];
                    yolobox_vec.push_back(box);
                }
            }
#else
            // best class
            obbBox box;
            box.class_id = argmax(output_data + i * nout + 4, m_class_num);
            box.score = output_data[i * nout + 4 + box.class_id];
            if(box.score <= m_confThreshold){
                continue;
            }
            int c = agnostic ? 0 : box.class_id * max_wh;
            box.x = output_data[i * nout + 0] + c;
            box.y = output_data[i * nout + 1] + c;
            box.w = output_data[i * nout + 2];
            box.h = output_data[i * nout + 3];
            box.angle = output_data[(i + 1) * nout - 1];
            yolobox_vec.push_back(box);
#endif
        }
        LOG_TS(m_ts, "yolov8_obb_post: get box");

        //debug
        // yolobox_vec.clear();
        // static int count = 0;
        // std::ifstream in("../../python/nmsed_boxes/"+std::to_string(count)+".bin", std::ios::binary);
        // in.seekg(0, std::ios::end);
        // std::streampos size = in.tellg();
        // // std::cout<<"boxes size:"<<size<<std::endl;
        // in.seekg(0, std::ios::beg);
        // float* box_data = new float[size / 4];
        // in.read((char*)box_data, size);
        // int box_count = size / 4 / 7;
        // for(int i = 0; i < box_count; i++){
        //     obbBox b;
        //     int c = (int)box_data[i*7+5] * max_wh;
        //     b.x = box_data[i*7] + c;
        //     b.y = box_data[i*7+1] + c;
        //     b.w = box_data[i*7+2];
        //     b.h = box_data[i*7+3];
        //     b.score = box_data[i*7+4];
        //     b.class_id = (int)box_data[i*7+5];
        //     b.angle = box_data[i*7+6];
        //     yolobox_vec.push_back(b);
        // }
        // delete box_data;
        // in.close();

        LOG_TS(m_ts, "yolov8_obb_post: nms_rotated");
        nms_rotated(yolobox_vec, m_nmsThreshold);
        LOG_TS(m_ts, "yolov8_obb_post: nms_rotated");

        if (yolobox_vec.size() > max_det) {
            yolobox_vec.erase(yolobox_vec.begin(), yolobox_vec.begin() + (yolobox_vec.size() - max_det));
        }

        if(!agnostic){
            for (int i = 0; i < yolobox_vec.size(); i++) {
                int c = yolobox_vec[i].class_id * max_wh;
                yolobox_vec[i].x = yolobox_vec[i].x - c;
                yolobox_vec[i].y = yolobox_vec[i].y - c;
            }
        }

        regularize_rbox(yolobox_vec);

        int tx1 = txy_batch[batch_idx].first;
        int ty1 = txy_batch[batch_idx].second;
        float ratio = ratios_batch[batch_idx];
        float inv_ratio = 1.0 / ratio;
        for (int i = 0; i < yolobox_vec.size(); i++) {
            yolobox_vec[i].x = std::round((yolobox_vec[i].x - tx1) * inv_ratio);
            yolobox_vec[i].y = std::round((yolobox_vec[i].y - ty1) * inv_ratio);
            yolobox_vec[i].w = std::round(yolobox_vec[i].w * inv_ratio);
            yolobox_vec[i].h = std::round(yolobox_vec[i].h * inv_ratio);
        }
        detected_boxes.push_back(xywhr2xyxyxyxy(yolobox_vec));
    }
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
    return 0;
}

int YoloV8_obb::argmax(float* data, int num) {
    float max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < num; ++i) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }

    return max_index;
}

void YoloV8_obb::regularize_rbox(obbBoxVec& obbVec){
    for(auto& obb : obbVec){
        if(obb.h > obb.w){
            std::swap(obb.w, obb.h);
            obb.angle = obb.angle + M_PI / 2;
        }
        obb.angle = std::fmod(obb.angle, M_PI);
        if(obb.angle < 0){
            obb.angle += M_PI;
        }
    }
}

obbBoxVec_xyxy YoloV8_obb::xywhr2xyxyxyxy(const obbBoxVec& obbVec){
    obbBoxVec_xyxy obbVec_;
    for(auto& obb : obbVec){
        obbBox_xyxy obb_;
        float cos_value = std::cos(obb.angle);
        float sin_value = std::sin(obb.angle);

        // Calculate half-dimensions rotated
        float dx1 = (obb.w / 2) * cos_value;
        float dy1 = (obb.w / 2) * sin_value;
        float dx2 = (obb.h / 2) * sin_value;
        float dy2 = (obb.h / 2) * cos_value;

        // Calculate corners
        obb_.class_id = obb.class_id;
        obb_.score = obb.score;
        obb_.x1 = std::round(obb.x + dx1 + dx2);
        obb_.y1 = std::round(obb.y + dy1 - dy2);
        obb_.x2 = std::round(obb.x + dx1 - dx2);
        obb_.y2 = std::round(obb.y + dy1 + dy2);
        obb_.x3 = std::round(obb.x - dx1 - dx2);
        obb_.y3 = std::round(obb.y - dy1 + dy2);
        obb_.x4 = std::round(obb.x - dx1 + dx2);
        obb_.y4 = std::round(obb.y - dy1 - dy2);
        obbVec_.push_back(obb_);
    }
    return obbVec_;
}

std::tuple<float, float, float> YoloV8_obb::convariance_matrix(const obbBox& obb){
    float w = obb.w;
    float h = obb.h;
    float r = obb.angle;
    float a = w * w / 12.0;
    float b = h * h / 12.0;
    float cos_r = std::cos(r);
    float sin_r = std::sin(r);
    float a_val = a * cos_r * cos_r + b * sin_r * sin_r;
    float b_val = a * sin_r * sin_r + b * cos_r * cos_r;
    float c_val = (a - b) * cos_r * sin_r;
    return std::make_tuple(a_val, b_val, c_val);
}

float YoloV8_obb::probiou(const obbBox& obb1, const obbBox& obb2, float eps){
    // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    float a1, b1, c1, a2, b2, c2;
    std::tie(a1, b1, c1) = convariance_matrix(obb1);
    std::tie(a2, b2, c2) = convariance_matrix(obb2);
    float x1 = obb1.x, y1 = obb1.y;
    float x2 = obb2.x, y2 = obb2.y;
    float t1 = ((a1 + a2) * std::pow(y1 - y2, 2) + (b1 + b2) * std::pow(x1 - x2, 2)) / ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps);
    float t3 = std::log(((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2)) / (4 * std::sqrt(std::max(a1 * b1 - c1 * c1, 0.0f)) * std::sqrt(std::max(a2 * b2 - c2 * c2, 0.0f)) + eps) + eps);
    float bd = 0.25 * t1 + 0.5 * t2 + 0.5 * t3;
    bd = std::max(std::min(bd, 100.0f), eps);
    float hd = std::sqrt(1.0 - std::exp(-bd) + eps);
    return 1 - hd;
}


void YoloV8_obb::nms_rotated(obbBoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const obbBox& a, const obbBox& b) { return a.score < b.score; });

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float iou = probiou(dets[index], dets[i]);
            if (iou >= nmsConfidence) {
                dets.erase(dets.begin() + i);
                index--;
            } else {
                i++;
            }
        }
        index--;
    }
}

void YoloV8_obb::drawPred(obbBoxVec_xyxy& box_vec, cv::Mat& frame, float draw_thresh)   // Draw the predicted bounding box
{
    for (int n = 0; n < box_vec.size(); n++) {
        cv::Point rook_points[4];
        rook_points[0] = cv::Point(int(box_vec[n].x1), int(box_vec[n].y1));
        rook_points[1] = cv::Point(int(box_vec[n].x2), int(box_vec[n].y2));
        rook_points[2] = cv::Point(int(box_vec[n].x3), int(box_vec[n].y3));
        rook_points[3] = cv::Point(int(box_vec[n].x4), int(box_vec[n].y4));

        const cv::Point* ppt[1] = {rook_points};
        int npt[] = {4};
        std::string label = m_class_names[box_vec[n].class_id] + cv::format(":%.2f", box_vec[n].score);
        // std::string label = m_class_names[box_vec[n].class_id] + ":" + std::to_string(box_vec[n].score);
        cv::Scalar color(colors[box_vec[n].class_id][0], colors[box_vec[n].class_id][1], colors[box_vec[n].class_id][2]);
        if (box_vec[n].score > draw_thresh) {
            cv::polylines(frame, ppt, npt, 1, 1, color, 2, 8, 0);
            cv::putText(frame, label, cv::Point(int(box_vec[n].x1), int(box_vec[n].y1 - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }
    }
}