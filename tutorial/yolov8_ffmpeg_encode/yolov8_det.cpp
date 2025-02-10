//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov8_det.hpp"
#include <string>
#include <vector>
#include <cmath>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

int YoloV8_det::Detect(const std::vector<bm_image>& input_images, std::vector<YoloV8BoxVec>& boxes) {
    assert(input_images.size() <= batch_size);
    int ret = 0;
    bm_tensor_t input_tensor;
    std::vector<bm_tensor_t> output_tensors;
    output_tensors.resize(netinfo->output_num);
    std::vector<std::pair<int, int>> txy_batch;
    std::vector<float> ratios_batch;
    m_ts->save("yolov8 preprocess", input_images.size());
    ret = pre_process(input_images, input_tensor, txy_batch, ratios_batch);
    assert(ret == 0);
    m_ts->save("yolov8 preprocess", input_images.size());

    m_ts->save("yolov8 inference", input_images.size());
    ret = forward(input_tensor, output_tensors);
    assert(ret == 0);
    m_ts->save("yolov8 inference", input_images.size());

    m_ts->save("yolov8 postprocess", input_images.size());
    ret = post_process(input_images, output_tensors, txy_batch, ratios_batch, boxes);
    assert(ret == 0);
    m_ts->save("yolov8 postprocess", input_images.size());
    return ret;
}

float YoloV8_det::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

int YoloV8_det::pre_process(const std::vector<bm_image>& images, 
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

int YoloV8_det::forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors){
    // static int count = 0;
    // std::ifstream input_data("../../python/dummy_inputs/"+std::to_string(count++)+".bin", std::ios::binary);
    // static float *input = new float[3*1024*1024];
    // input_data.read((char*)input, 3*1024*1024*sizeof(float));
    // bm_memcpy_s2d(handle, input_tensor.device_mem, input);

    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                    output_tensors.data(), netinfo->output_num);
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
float* YoloV8_det::get_cpu_data(bm_tensor_t* tensor, float scale){
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


int YoloV8_det::post_process(const std::vector<bm_image>& input_images, 
                             std::vector<bm_tensor_t>& output_tensors, 
                             const std::vector<std::pair<int, int>>& txy_batch, 
                             const std::vector<float>& ratios_batch, 
                             std::vector<YoloV8BoxVec>& detected_boxes) {
    float* data_box = NULL;
    bm_tensor_t tensor_box;
    for(int i = 0; i < output_tensors.size(); i++) {
        if(output_tensors[i].shape.num_dims == 3){
            tensor_box = output_tensors[i];
            data_box = get_cpu_data(&output_tensors[i], netinfo->output_scales[i]);
        }
    }

    for (int batch_idx = 0; batch_idx < input_images.size(); ++batch_idx) {
        YoloV8BoxVec yolobox_vec;
        auto& frame = input_images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int box_num = tensor_box.shape.dims[1];
        int nout = tensor_box.shape.dims[2];
        float* batch_data_box =  data_box + batch_idx * box_num * nout; //output_tensor: [bs, box_num, class_num + 5]

        // Candidates
        for (int i = 0; i < box_num; i++) {
            int box_index = i * nout;
            float* cls_conf = batch_data_box + box_index + 4; //output_tensor's last dim: [x, y, w, h, cls_conf0, ..., cls_conf14, rotate_angle]
#if USE_MULTICLASS_NMS
            // multilabel
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[j];
                if (cur_value > m_confThreshold) {
                    YoloV8Box box;
                    box.score = cur_value;
                    box.class_id = j;
                    float centerX = batch_data_box[box_index];
                    float centerY = batch_data_box[box_index + 1];
                    float width = batch_data_box[box_index + 2];
                    float height = batch_data_box[box_index + 3];

                    int c = agnostic ? 0 : box.class_id * max_wh;
                    box.x1 = centerX - width / 2 + c;
                    box.y1 = centerY - height / 2 + c;
                    box.x2 = box.x1 + width;
                    box.y2 = box.y1 + height;
                    yolobox_vec.push_back(box);
                }
            }
#else
            // best class
            YoloV8Box box;
            box.class_id = argmax(batch_data_box + box_index + 4, m_class_num);
            box.score = batch_data_box[box_index + 4 + box.class_id];
            if(box.score <= m_confThreshold){
                continue;
            }
            int c = agnostic ? 0 : box.class_id * max_wh;
            float centerX = batch_data_box[box_index];
            float centerY = batch_data_box[box_index + 1];
            float width = batch_data_box[box_index + 2];
            float height = batch_data_box[box_index + 3];
            box.x1 = centerX - width / 2 + c;
            box.y1 = centerY - height / 2 + c;
            box.x2 = box.x1 + width;
            box.y2 = box.y1 + height;
            yolobox_vec.push_back(box);
#endif
        }
        NMS(yolobox_vec, m_nmsThreshold);

        if (yolobox_vec.size() > max_det) {
            yolobox_vec.erase(yolobox_vec.begin(), yolobox_vec.begin() + (yolobox_vec.size() - max_det));
        }

        if(!agnostic){
            for (int i = 0; i < yolobox_vec.size(); i++) {
                int c = yolobox_vec[i].class_id * max_wh;
                yolobox_vec[i].x1 = yolobox_vec[i].x1 - c;
                yolobox_vec[i].y1 = yolobox_vec[i].y1 - c;
                yolobox_vec[i].x2 = yolobox_vec[i].x2 - c;
                yolobox_vec[i].y2 = yolobox_vec[i].y2 - c;
            }
        }

        int tx1 = txy_batch[batch_idx].first;
        int ty1 = txy_batch[batch_idx].second;
        float ratio = ratios_batch[batch_idx];
        float inv_ratio = 1.0 / ratio;
        for (int i = 0; i < yolobox_vec.size(); i++) {
            yolobox_vec[i].x1 = std::round((yolobox_vec[i].x1 - tx1) * inv_ratio);
            yolobox_vec[i].y1 = std::round((yolobox_vec[i].y1 - ty1) * inv_ratio);
            yolobox_vec[i].x2 = std::round((yolobox_vec[i].x2 - tx1) * inv_ratio);
            yolobox_vec[i].y2 = std::round((yolobox_vec[i].y2 - ty1) * inv_ratio);
        }
        clip_boxes(yolobox_vec, frame_width, frame_height);
        detected_boxes.push_back(yolobox_vec);
    }

    for(int i = 0; i < output_tensors.size(); i++) {
        float* tensor_data = NULL;
        if(output_tensors[i].shape.num_dims == 3){
            tensor_data = data_box;
        }

        if(misc_info.pcie_soc_mode == 1){ // soc
            if(output_tensors[i].dtype != BM_FLOAT32){
                delete [] tensor_data;
            } else {
                int tensor_size = bm_mem_get_device_size(output_tensors[i].device_mem);
                bm_status_t ret = bm_mem_unmap_device_mem(handle, tensor_data, tensor_size);
                assert(BM_SUCCESS == ret);
            }
            if(output_tensors[i].dtype != BM_FLOAT32){
                delete [] tensor_data;
            } else {
                int tensor_size = bm_mem_get_device_size(output_tensors[i].device_mem);
                bm_status_t ret = bm_mem_unmap_device_mem(handle, tensor_data, tensor_size);
                assert(BM_SUCCESS == ret);
            }
        } else {
            delete [] tensor_data;
        }
        bm_free_device(handle, output_tensors[i].device_mem);
    }
    return 0;
}

int YoloV8_det::argmax(float* data, int num) {
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

void YoloV8_det::clip_boxes(YoloV8BoxVec& yolobox_vec, int src_w, int src_h) {
    for (int i = 0; i < yolobox_vec.size(); i++) {
        yolobox_vec[i].x1 = std::max((float)0.0, std::min(yolobox_vec[i].x1, (float)src_w));
        yolobox_vec[i].y1 = std::max((float)0.0, std::min(yolobox_vec[i].y1, (float)src_h));
        yolobox_vec[i].x2 = std::max((float)0.0, std::min(yolobox_vec[i].x2, (float)src_w));
        yolobox_vec[i].y2 = std::max((float)0.0, std::min(yolobox_vec[i].y2, (float)src_h));
    }
}

void YoloV8_det::xywh2xyxy(YoloV8BoxVec& xyxyboxes, std::vector<std::vector<float>> box) {
    for (int i = 0; i < box.size(); i++) {
        YoloV8Box tmpbox;
        tmpbox.x1 = box[i][0] - box[i][2] / 2;
        tmpbox.y1 = box[i][1] - box[i][3] / 2;
        tmpbox.x2 = box[i][0] + box[i][2] / 2;
        tmpbox.y2 = box[i][1] + box[i][3] / 2;
        xyxyboxes.push_back(tmpbox);
    }
}

void YoloV8_det::NMS(YoloV8BoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YoloV8Box& a, const YoloV8Box& b) { return a.score < b.score; });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        float width = dets[i].x2 - dets[i].x1;
        float height = dets[i].y2 - dets[i].y1;
        areas[i] = width * height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].x1, dets[i].x1);
            float top = std::max(dets[index].y1, dets[i].y1);
            float right = std::min(dets[index].x2, dets[i].x2);
            float bottom = std::min(dets[index].y2, dets[i].y2);
            float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index--;
            } else {
                i++;
            }
        }
        index--;
    }
}

void YoloV8_det::draw_result(cv::Mat& img, YoloV8BoxVec& result) {
    for (int i = 0; i < result.size(); i++) {
        if(result[i].score < 0.25) continue;
        int left, top;
        left = result[i].x1;
        top = result[i].y1;
        int color_num = i;
        cv::Scalar color(colors[result[i].class_id % 25][0], colors[result[i].class_id % 25][1],
                         colors[result[i].class_id % 25][2]);
        cv::Rect bound = {result[i].x1, result[i].y1, result[i].x2 - result[i].x1, result[i].y2 - result[i].y1};

        rectangle(img, bound, color, 2);
        std::string label = std::string(m_class_names[result[i].class_id]) + std::to_string(result[i].score);
        putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
}

void YoloV8_det::draw_result_bmcv(bm_image& img, YoloV8BoxVec& result, bool put_text_flag) {
    if (img.image_format != FORMAT_YUV420P){
        throw std::runtime_error("Only support yuv420p format.");
    }
    for (int i = 0; i < result.size(); i++) {
        if(result[i].score < 0.01) continue;
        int left, top, width, height, classId;
        left = result[i].x1;
        top = result[i].y1;
        width = result[i].x2 - left;
        height = result[i].y2 - top;
        classId = result[i].class_id;
        int colors_num = colors.size();
        //Draw a rectangle displaying the bounding box
        bmcv_rect_t rect;
        rect.start_x = MIN(MAX(left, 0), img.width);
        rect.start_y = MIN(MAX(top, 0), img.height);
        rect.crop_w = MAX(MIN(width, img.width - rect.start_x), 0);
        rect.crop_h = MAX(MIN(height, img.height - rect.start_y), 0);
        if (BM_SUCCESS != bmcv_image_draw_rectangle(handle, 
                                                    img, 
                                                    1, 
                                                    &rect, 
                                                    3, 
                                                    colors[classId % colors_num][0], 
                                                    colors[classId % colors_num][1], 
                                                    colors[classId % colors_num][2])){
            std::cout << "bmcv draw rect error !!!" << std::endl;  
        }
        
        if (put_text_flag){
            //Get the label for the class name and its confidence
            std::string label = m_class_names[classId] + ":" + cv::format("%.2f", result[i].score);
            bmcv_point_t org = {left, top};
            bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]};
            int thickness = 2;
            float fontScale = 2; 
            if (BM_SUCCESS != bmcv_image_put_text(handle, 
                                                  img, 
                                                  label.c_str(), 
                                                  org, 
                                                  color, 
                                                  fontScale, 
                                                  thickness)) {
                std::cout << "bmcv put text error !!!" << std::endl;   
            }
        }
    }
}