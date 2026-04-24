//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov8_seg_fuse.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

int YoloV8SegFuse::Detect(const std::vector<bm_image>& input_images, std::vector<YoloV8BoxVec>& boxes) {
    assert(input_images.size() <= batch_size);
    int ret = 0;
    bm_tensor_t input_tensor;
    std::vector<bm_tensor_t> output_tensors;
    output_tensors.resize(2);
    std::vector<std::pair<int, int>> txy_batch;
    std::vector<std::pair<float, float>> ratios_batch;
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

float YoloV8SegFuse::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

int YoloV8SegFuse::pre_process(const std::vector<bm_image>& images, 
                            bm_tensor_t& input_tensor, 
                            std::vector<std::pair<int, int>>& txy_batch, 
                            std::vector<std::pair<float, float>>& ratios_batch) {
    int ret = 0;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    m_resized_imgs.resize(batch_size);
    m_converto_imgs.resize(batch_size);

    //create bm_images
    ret = bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE, m_resized_imgs.data(), batch_size, NULL, -1, false);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }

    // create tensor for model input
    ret = bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
    assert(true == ret);
    ret = bm_image_attach_contiguous_mem(batch_size, m_resized_imgs.data(), input_tensor.device_mem);
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
        txy_batch.push_back(std::make_pair(tx1, ty1));
        ratios_batch.push_back(std::make_pair(ratio,ratio));
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

    // destroy bm_images
    ret = bm_image_dettach_contiguous_mem(batch_size, m_resized_imgs.data());
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    ret = bm_image_destroy_batch(m_resized_imgs.data(), batch_size, false);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    return ret;
}

int YoloV8SegFuse::forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors){
    // static int count = 0;
    // std::ifstream input_data("../../python/dummy_inputs/"+std::to_string(count++)+".bin", std::ios::binary);
    // static float *input = new float[3*1024*1024];
    // input_data.read((char*)input, 3*1024*1024*sizeof(float));
    // bm_memcpy_s2d(handle, input_tensor.device_mem, input);

    assert(netinfo->input_num == 1 && netinfo->output_num == 2);
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                    output_tensors.data(), netinfo->output_num);
    assert(ok == true);
    auto ret = bm_thread_sync(handle);
    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
    bm_free_device(handle, input_tensor.device_mem);
    return ret;
}

/**
 * @name    get_cpu_data
 * @brief   get cpu data of tensor.
 *
 * @param   [in]           tensor   input tensor.
 * @param   [in]           scale    scale of tensor.
 * @retval  float*         tensor's cpu data.
 */
float* YoloV8SegFuse::get_cpu_data(bm_tensor_t* tensor, float scale){
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

int YoloV8SegFuse::post_process(const std::vector<bm_image>& input_images, 
                             std::vector<bm_tensor_t>& output_tensors, 
                             const std::vector<std::pair<int, int>>& txy_batch, 
                             const std::vector<std::pair<float,float>>& ratios_batch,
                             std::vector<YoloV8BoxVec>& detected_boxes) {
    float* data_box = NULL;
    bm_tensor_t tensor_box;
    bm_tensor_t tensor_mask;
    for(int i = 0; i < output_tensors.size(); i++) {
        if(output_tensors[i].shape.num_dims == 3) {
            tensor_mask = output_tensors[i];
        }else if(output_tensors[i].shape.num_dims == 2){
            tensor_box = output_tensors[i];
            data_box = get_cpu_data(&output_tensors[i], netinfo->output_scales[i]);
        }
    }
    if(tensor_mask.shape.dims[0] <= 1){
        detected_boxes.resize(batch_size);
        return 0;
    }

    for (int batch_idx = 0; batch_idx < input_images.size(); ++batch_idx) {
        YoloV8BoxVec yolobox_vec;
        auto& frame = input_images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int box_num = tensor_box.shape.dims[0];
        int nout = tensor_box.shape.dims[1];
        float* batch_data_box =  data_box + batch_idx * box_num * nout; //output_tensor: [bs, box_num, class_num + 5]

        // Candidates
        for (int i = 0; i < box_num - 1; i++) {
            int box_index = i * nout;
            YoloV8Box box;
            box.score = batch_data_box[box_index + 4];
            box.class_id = batch_data_box[box_index + 5];
            box.x1 = batch_data_box[box_index];
            box.y1 = batch_data_box[box_index + 1];
            box.x2 = batch_data_box[box_index + 2];
            box.y2 = batch_data_box[box_index + 3];
            yolobox_vec.push_back(box);
        }

        int tx1 = txy_batch[batch_idx].first;
        int ty1 = txy_batch[batch_idx].second;
        float ratio_x = ratios_batch[batch_idx].first;
        float ratio_y = ratios_batch[batch_idx].second;
        float inv_ratio_x = 1.0 / ratio_x;
        float inv_ratio_y = 1.0 / ratio_y;
        for (int i = 0; i < yolobox_vec.size(); i++) {
            yolobox_vec[i].x1 = std::round((yolobox_vec[i].x1 - tx1) * inv_ratio_x);
            yolobox_vec[i].y1 = std::round((yolobox_vec[i].y1 - ty1) * inv_ratio_y);
            yolobox_vec[i].x2 = std::round((yolobox_vec[i].x2 - tx1) * inv_ratio_x);
            yolobox_vec[i].y2 = std::round((yolobox_vec[i].y2 - ty1) * inv_ratio_y);
        }
        clip_boxes(yolobox_vec, frame_width, frame_height);
        
        ImageInfo para = {cv::Size(frame_width, frame_height), {ratio_x, ratio_y, tx1, ty1}};
        YoloV8BoxVec yolobox_vec_tmp;
        for (int i = 0; i < yolobox_vec.size(); i++) {
            if (yolobox_vec[i].x2 > yolobox_vec[i].x1 + 1 && yolobox_vec[i].y2 > yolobox_vec[i].y1 + 1) {
                int mask_size = tensor_mask.shape.dims[1] * tensor_mask.shape.dims[2];
                bm_image batch_mask;
                bm_image_create(handle, tensor_mask.shape.dims[1], tensor_mask.shape.dims[2], FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &batch_mask);
                bm_device_mem_t batch_mask_devmem = bm_mem_from_device(
                                                        bm_mem_get_device_addr(tensor_mask.device_mem) + i * mask_size,
                                                        mask_size);
                bm_image_attach(batch_mask, &batch_mask_devmem);
                get_mask(batch_mask, 
                         para,
                         cv::Rect{yolobox_vec[i].x1, 
                                  yolobox_vec[i].y1, 
                                  yolobox_vec[i].x2 - yolobox_vec[i].x1,
                                  yolobox_vec[i].y2 - yolobox_vec[i].y1},
                        yolobox_vec[i].mask_img);
                yolobox_vec_tmp.push_back(yolobox_vec[i]);
                bm_image_detach(batch_mask);
                bm_image_destroy(batch_mask);
            }
        }
        detected_boxes.push_back(yolobox_vec_tmp);
    }

    for(int i = 0; i < output_tensors.size(); i++) {
        float* tensor_data = NULL;
        if(output_tensors[i].shape.num_dims == 2){
            tensor_data = data_box;        
            if(misc_info.pcie_soc_mode == 1){ // soc
                if(output_tensors[i].dtype != BM_FLOAT32){
                    delete [] tensor_data;
                } else {
                    int tensor_size = bm_mem_get_device_size(output_tensors[i].device_mem);
                    bm_status_t ret = bm_mem_unmap_device_mem(handle, tensor_data, tensor_size);
                    if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
                }
            } else {
                delete [] tensor_data;
            }
        }
        bm_free_device(handle, output_tensors[i].device_mem);
    }
    return 0;
}

void YoloV8SegFuse::get_mask(bm_image& mask_data,
                             const ImageInfo& para,
                             cv::Rect bound,
                             cv::Mat& mask_out) {
    cv::Vec4f trans = para.trans;
    int r_x = floor(trans[2] / m_net_w * (m_net_w / 4));
    int r_y = floor(trans[3] / m_net_h * (m_net_h / 4)); 
    int r_w = (m_net_w / 4) - 2 * r_x;
    int r_h = (m_net_h / 4) - 2 * r_y;
    r_w = MAX(r_w, 1);
    r_h = MAX(r_h, 1);

    int ret = 0;
    bm_image mask_resized;
    ret = bm_image_create(handle, para.raw_size.height, para.raw_size.width, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, &mask_resized);
    assert(ret == BM_SUCCESS);
    bmcv_resize_t attr1;
    attr1.start_x = r_x;
    attr1.start_y = r_y;
    attr1.in_width = r_w;
    attr1.in_height = r_h;
    attr1.out_width = para.raw_size.width;
    attr1.out_height = para.raw_size.height;

    bmcv_resize_image attr0;
    attr0.resize_img_attr = &attr1;
    attr0.roi_num = 1;
    attr0.stretch_fit = 1;
    attr0.interpolation = BMCV_INTER_LINEAR;

    ret = bmcv_image_resize(handle, 1, &attr0, &mask_data, &mask_resized);
    assert(ret == BM_SUCCESS);

    bm_image mask_bmimg;
    ret = bm_image_create(handle, mask_resized.height, mask_resized.width, mask_resized.image_format, mask_resized.data_type, &mask_bmimg);
    assert(ret == BM_SUCCESS);

    ret = bmcv_image_threshold(handle, mask_resized, mask_bmimg, 127, 1, BM_THRESH_BINARY);
    assert(ret == BM_SUCCESS);

    int mask_out_size = mask_resized.width * mask_resized.height;
    uint8_t* mask_out_data = new uint8_t[mask_out_size];
    memset(mask_out_data, 0, mask_out_size);
    ret = bm_image_copy_device_to_host(mask_bmimg, (void**)&mask_out_data);
    assert(ret == BM_SUCCESS);
    
    cv::Mat mask_out_(mask_resized.height, mask_resized.width, CV_8UC1, mask_out_data);
    mask_out = mask_out_(bound).clone();
    
    delete mask_out_data;
    ret = bm_image_destroy(mask_bmimg);
    assert(ret == BM_SUCCESS);
    ret = bm_image_destroy(mask_resized);
    assert(ret == BM_SUCCESS);
}

void YoloV8SegFuse::clip_boxes(YoloV8BoxVec& yolobox_vec, int src_w, int src_h) {
    for (int i = 0; i < yolobox_vec.size(); i++) {
        yolobox_vec[i].x1 = std::max((float)0.0, std::min(yolobox_vec[i].x1, (float)src_w));
        yolobox_vec[i].y1 = std::max((float)0.0, std::min(yolobox_vec[i].y1, (float)src_h));
        yolobox_vec[i].x2 = std::max((float)0.0, std::min(yolobox_vec[i].x2, (float)src_w));
        yolobox_vec[i].y2 = std::max((float)0.0, std::min(yolobox_vec[i].y2, (float)src_h));
    }
}

void YoloV8SegFuse::xywh2xyxy(YoloV8BoxVec& xyxyboxes, std::vector<std::vector<float>> box) {
    for (int i = 0; i < box.size(); i++) {
        YoloV8Box tmpbox;
        tmpbox.x1 = box[i][0] - box[i][2] / 2;
        tmpbox.y1 = box[i][1] - box[i][3] / 2;
        tmpbox.x2 = box[i][0] + box[i][2] / 2;
        tmpbox.y2 = box[i][1] + box[i][3] / 2;
        xyxyboxes.push_back(tmpbox);
    }
}

void YoloV8SegFuse::draw_result(cv::Mat& img, YoloV8BoxVec& result, float draw_thresh) {
    cv::Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++) {
        if(result[i].score < draw_thresh) continue;
        int left, top;
        left = result[i].x1;
        top = result[i].y1;
        int color_num = i;
        cv::Scalar color(colors[result[i].class_id % 25][0], colors[result[i].class_id % 25][1],
                         colors[result[i].class_id % 25][2]);
        cv::Rect bound = {result[i].x1, result[i].y1, result[i].x2 - result[i].x1, result[i].y2 - result[i].y1};

        rectangle(img, bound, color, 2);
        if (result[i].mask_img.rows && result[i].mask_img.cols > 0) {
            mask(bound).setTo(color, result[i].mask_img);
        }
        std::string label = std::string(m_class_names[result[i].class_id]) + std::to_string(result[i].score);
        putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
    addWeighted(img, 0.6, mask, 0.4, 0, img);  // add mask to src
}