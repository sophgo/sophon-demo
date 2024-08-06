//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov8.hpp"
#include <fstream>
#include <string>
#include <vector>
#if IS_SOC
#include <arm_neon.h>
#endif
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 0
#define USE_NEON 0

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

const std::vector<std::pair<int, int>> YoloV8::pointLinks = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11},
    {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2},
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}
};

bmcv_color_t YoloV8::GetBmColor() {
    bmcv_color_t c;
    int id = m_colorIndex % colors.size();

    c.r = colors[id][0];
    c.g = colors[id][1];
    c.b = colors[id][2];
    if (++m_colorIndex > 1 << 30) {
        m_colorIndex = 0;
    }
    return c;
}

YoloV8::YoloV8(std::shared_ptr<BMNNContext> context) : 
    m_bmContext(context),
    m_colorIndex(0) {
    std::cout << "YoloV8 ctor .." << std::endl;
}

YoloV8::~YoloV8() {
    std::cout << "YoloV8 dtor ..." << std::endl;
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for (int i = 0; i < max_batch; i++) {
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
}

int YoloV8::Init(float confThresh, float nmsThresh) {
    m_confThreshold = confThresh;
    m_nmsThreshold = nmsThresh;

    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    // 2. get input
    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    // 3. get output
    output_num = m_bmNetwork->outputTensorNum();
    assert(output_num > 0);
    min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;
    // 4. initialize bmimages
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);

    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int i = 0; i < max_batch; i++) {
        auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                   &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype,
                                     m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    // 5.converto
    float input_scale = tensor->get_scale();
    input_scale = input_scale * 1.0 / 255.f;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = 0;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = 0;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = 0;

    return 0;
}

void YoloV8::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int YoloV8::batch_size() {
    return max_batch;
};

int YoloV8::Detect(const std::vector<bm_image>& input_images, std::vector<YoloV8BoxVec>& boxes) {
    int ret = 0;
    // 3. preprocess
    m_ts->save("yolov8 preprocess", input_images.size());
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    m_ts->save("yolov8 preprocess", input_images.size());

    // 4. forward
    m_ts->save("yolov8 inference", input_images.size());
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("yolov8 inference", input_images.size());

    // 5. post process
    m_ts->save("yolov8 postprocess", input_images.size());
    ret = post_process(input_images, boxes);
    
    CV_Assert(ret == 0);
    m_ts->save("yolov8 postprocess", input_images.size());
    return ret;
}

int YoloV8::pre_process(const std::vector<bm_image>& images) {
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    int image_n = images.size();

    // 1. resize image letterbox
    int ret = 0;
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
            bm_image_create(m_bmContext->handle(), image1.height, image1.width, image1.image_format, image1.data_type,
                            &image_aligned, stride2);

            bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;
            bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
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

            int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);  // padding 大小
            padding_attr.dst_crop_sty = ty1;
            padding_attr.dst_crop_stx = 0;
        } else {
            padding_attr.dst_crop_h = m_net_h;
            padding_attr.dst_crop_w = images[i].width * ratio;

            int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = tx1;
        }

        bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
        auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
                                                  &padding_attr, &crop_rect);
#else
        auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i], &m_resized_imgs[i]);
#endif
        assert(BM_SUCCESS == ret);
        if (need_copy)
            bm_image_destroy(image_aligned);
    }

    // 2. converto img /= 255
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(),
                                m_converto_imgs.data());
    CV_Assert(ret == 0);

    // 3. attach to tensor
    if (image_n != max_batch)
        image_n = m_bmNetwork->get_nearest_batch(image_n);
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number

    return 0;
}

float YoloV8::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
    assert(dst_w > 0 && dst_h > 0 && src_w > 0 && src_h > 0);
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

int YoloV8::post_process(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& detected_boxes) {
    YoloV8BoxVec yolobox_vec;
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {

        auto& frame = images[batch_idx];
        int box_num = outputTensor->get_shape()->dims[2];
        int box_dataLen = outputTensor->get_shape()->dims[1];
        int pd_len = box_dataLen - 5;
        bool isAlignWidth = false;
        float* output_data = nullptr;
        float ratio = get_aspect_scaled_ratio(frame.width, frame.height, m_net_w, m_net_h, &isAlignWidth);
        float tx = isAlignWidth ? 0 : (m_net_w - (float)(frame.width * ratio)) / 2;
        float ty = !isAlignWidth ? 0 : (m_net_h - (float)(frame.height * ratio)) / 2;

        yolobox_vec.clear();
        assert(box_dataLen != 0 && box_num != 0 && pd_len % 3 == 0);
        m_points_num = pd_len / 3;

        LOG_TS(m_ts, "post 1: get output");
        output_data = (float*)outputTensor->get_cpu_data() + batch_idx * box_num * box_dataLen;
        LOG_TS(m_ts, "post 1: get output");

        LOG_TS(m_ts, "post 2: get detections matrix");
        ProcessPoseBox(yolobox_vec, output_data, box_num);
        LOG_TS(m_ts, "post 2: get detections matrix");

        LOG_TS(m_ts, "post 3: nms");
        NMS(yolobox_vec, m_nmsThreshold);
        ReTransPoseBox(yolobox_vec, tx, ty, ratio, frame.width, frame.height, output_data, box_num);
        LOG_TS(m_ts, "post 3: nms");

        detected_boxes.push_back(yolobox_vec);
    }

    return 0;
}


int YoloV8::ProcessPoseBox(YoloV8BoxVec& v, float* d, int n) {
    for (int i = 0; i < n; i++) {
        if (d[4 * n + i] < m_confThreshold) {
            continue;
        }

        YoloV8Box b;
        float cx = d[i + 0 * n];
        float cy = d[i + 1 * n];
        float w = d[i + 2 * n];
        float h = d[i + 3 * n];

        b.x1 = cx - w / 2;
        b.y1 = cy - h / 2;
        b.x2 = b.x1 + w;
        b.y2 = b.y1 + h;
        b.index = i;
        b.score = d[4 * n + i];
        v.push_back(b);
    }
}

int YoloV8::ReTransPoseBox(YoloV8BoxVec& v, float tx, float ty, float r, int fw, int fh, float* d, int n) {
    int i = 0;
    for (YoloV8Box &b : v) {
        b.x1 = (b.x1 - tx) / r;
        b.y1 = (b.y1 - ty) / r;
        b.x2 = (b.x2 - tx) / r;
        b.y2 = (b.y2 - ty) / r;

        b.x1 = b.x1 < 0 ? 0 : (b.x1 > fw ? fw : b.x1);
        b.x2 = b.x2 < 0 ? 0 : (b.x2 > fw ? fw : b.x2);
        b.y1 = b.y1 < 0 ? 0 : (b.y1 > fh ? fh : b.y1);
        b.y2 = b.y2 < 0 ? 0 : (b.y2 > fh ? fh : b.y2);

        b.keyPoints.reserve(m_points_num * 3);
        for (int i = 0; i < m_points_num; i++) {

            float px = d[(5 + i * 3 + 0)* n + b.index];
            float py = d[(5 + i * 3 + 1)* n + b.index];

            px = (px - tx) / r;
            py = (py - ty) / r;

            px = px < 0 ? 0 : (px > fw ? fw : px);
            py = py < 0 ? 0 : (py > fh ? fh : py);

            b.keyPoints.push_back(px);
            b.keyPoints.push_back(py);
            b.keyPoints.push_back(d[(5 + i * 3 + 2)* n + b.index]);
        }
        i++;
    }
}
#if USE_NEON
int YoloV8::get_max_value_neno(float* cls_conf,float &max_value ,int & max_index,int i,int nout){
    
            int m_class_num_tmp=m_class_num/4;

            float32x4_t max_value_neon = vdupq_n_f32(0);
            int32x4_t max_index_neon = vdupq_n_s32(0); // 用-1初始化索引

            for (int j = 0; j < m_class_num_tmp; j++) {
                float32_t *data_neon = &cls_conf[i * nout + j * 4];
                float32x4_t cur_value = vld1q_f32(data_neon);
                // 比较当前值和最大值，并更新最大值和索引
                uint32x4_t mask = vcgtq_f32(cur_value, max_value_neon); // 比较当前值是否大于最大值
                max_value_neon = vmaxq_f32(max_value_neon, cur_value);
                max_index_neon = vbslq_s32(mask, vdupq_n_s32(j), max_index_neon); // 条件选择索引
            }

            // 将NEON寄存器的值提取到数组中
            float32_t max_values[4];
            int32_t max_indices[4];
            vst1q_f32(max_values, max_value_neon);
            vst1q_s32(max_indices, max_index_neon);

            // 遍历数组以找到最终的最大值和索引
            for (int j = 0; j < 4; j++) {
                if (max_values[j] > max_value) {
                    max_value = max_values[j];
                    max_index = max_indices[j]*4+j;
                }
            }  
              
   
}
#endif  

void YoloV8::NMS(YoloV8BoxVec& dets, float nmsConfidence) {
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

    if (dets.size() > max_det) {
        dets.erase(dets.begin(), dets.begin() + (dets.size() - max_det));
    }
}

void YoloV8::draw_bmcv(bm_handle_t& handle,
    YoloV8Box& b,
    bm_image& frame,
    bool putScore) 
{
    if (b.score < 0.25) {
        return;
    }
    //Draw a rectangle displaying the bounding box
    bmcv_rect_t rect;
    rect.start_x = MIN(MAX(b.x1, 0), frame.width);
    rect.start_y = MIN(MAX(b.y1, 0), frame.height);
    rect.crop_w = MAX(MIN(b.x2 - b.x1, frame.width - rect.start_x), 0);
    rect.crop_h = MAX(MIN(b.y2 - b.y1, frame.height - rect.start_y), 0);
    int thickness = 2;

    bmcv_color_t color = GetBmColor();
    if(rect.crop_w <= thickness * 2 || rect.crop_h <= thickness * 2){
            std::cout << "width or height too small, this rect will not be drawed: " << 
                "[" << rect.start_x << ", "<< rect.start_y << ", " << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
        } 
    else{
        bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, color.r, color.g, color.b);
    }

    if (putScore){
        //Get the label for the class name and its confidence
        std::string label = "score: " + cv::format("%.2f", b.score < 1 ? b.score * 100 : b.score);
        bmcv_point_t org = {b.x1, b.y1 - 5};
        if (frame.width - org.x  < 200) {
            org.x = frame.width - 200;
        }
        if (org.y  < 40) {
            org.y = 40;
        }
        float fontScale = 1; 
        bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness);
    }

    int pointsNum = b.keyPoints.size() / 3;
    std::vector<bmcv_point_t> bmPointsStart, bmPointsEnd;
    if (pointsNum != 17) {
        return;
    }
    bmPointsStart.reserve(pointLinks.size());
    bmPointsEnd.reserve(pointLinks.size());

    int i = 0;
    for (auto& link : pointLinks) {
        if (b.keyPoints[link.first * 3 + 2] < 0.4 || b.keyPoints[link.second * 3 + 2] < 0.4) {
            continue;
        }

        bmPointsStart.push_back(bmcv_point_t{b.keyPoints[link.first * 3], b.keyPoints[link.first * 3 + 1]});
        bmPointsEnd.push_back(bmcv_point_t{b.keyPoints[link.second * 3], b.keyPoints[link.second * 3 + 1]});
        i++;
    }

    bmcv_image_draw_lines(handle, frame, bmPointsStart.data(), bmPointsEnd.data(), bmPointsStart.size() , color, thickness);

}
