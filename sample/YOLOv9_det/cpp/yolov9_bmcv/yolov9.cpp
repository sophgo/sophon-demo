//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov9.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "libavutil/macros.h"

#define USE_ASPECT_RATIO 1
#define USE_MULTICLASS_NMS 0
#define USE_OPT 1
#define USE_NEON 0

const std::vector<std::vector<unsigned char>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloV9::YoloV9(std::shared_ptr<BMNNContext> context) : m_bmContext(context) {
    std::cout << "YoloV9 ctor .." << std::endl;
}

YoloV9::~YoloV9() {
    std::cout << "YoloV9 dtor ..." << std::endl;
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for (int i = 0; i < max_batch; i++) {
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
}

int YoloV9::Init(float confThresh, float nmsThresh, const std::string& coco_names_file) {
    m_confThreshold = confThresh;
    m_nmsThreshold = nmsThresh;

    std::ifstream ifs(coco_names_file);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            line = line.substr(0, line.length() - 1);
            m_class_names.push_back(line);
        }
    }

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
    use_cpu_opt = (m_bmNetwork->outputTensor(0)->get_shape()->dims[1] > m_bmNetwork->outputTensor(0)->get_shape()->dims[2]);
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

void YoloV9::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int YoloV9::batch_size() {
    return max_batch;
};

int YoloV9::Detect(const std::vector<bm_image>& input_images, std::vector<YoloV9BoxVec>& boxes) {
    int ret = 0;
    // 3. preprocess
    m_ts->save("yolov9 preprocess", input_images.size());
    ret = pre_process(input_images);
    assert(ret == 0);
    m_ts->save("yolov9 preprocess", input_images.size());

    // 4. forward
    m_ts->save("yolov9 inference", input_images.size());
    ret = m_bmNetwork->forward();
    assert(ret == 0);
    m_ts->save("yolov9 inference", input_images.size());

    // 5. post process
    m_ts->save("yolov9 postprocess", input_images.size());
    if(use_cpu_opt){
#if USE_OPT
        ret = post_process_opt(input_images, boxes);
#else
        std::cout<<"NOT_USE_OPT"<<std::endl;
#endif
    }else{
        ret = post_process(input_images, boxes);
    }
    assert(ret == 0);
    m_ts->save("yolov9 postprocess", input_images.size());
    return ret;
}

int YoloV9::pre_process(const std::vector<bm_image>& images) {
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
    assert(ret == 0);

    // 3. attach to tensor
    if (image_n != max_batch)
        image_n = m_bmNetwork->get_nearest_batch(image_n);
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number

    return 0;
}

float YoloV9::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

int YoloV9::post_process(const std::vector<bm_image>& images, std::vector<YoloV9BoxVec>& detected_boxes) {
    YoloV9BoxVec yolobox_vec;
    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);

    for (int i = 0; i < output_num; i++) {
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int min_idx = 0;
        int box_num = 0;

        // Single output
        auto out_tensor = outputTensors[min_idx];
        m_class_num = out_tensor->get_shape()->dims[1] - mask_num - 4;
        int feat_num = out_tensor->get_shape()->dims[2];
        int nout = m_class_num + mask_num + 4;
        float* output_data = nullptr;
        std::vector<float> decoded_data;

        LOG_TS(m_ts, "post 1: get output");
        assert(box_num == 0 || box_num == out_tensor->get_shape()->dims[1]);
        box_num = out_tensor->get_shape()->dims[1];
        output_data = (float*)out_tensor->get_cpu_data() + batch_idx * feat_num * (m_class_num + mask_num + 4);
        LOG_TS(m_ts, "post 1: get output");

        // Candidates
        LOG_TS(m_ts, "post 2: get detections matrix nx6 (xyxy, conf, cls)");
        float* cls_conf = output_data + 4 * feat_num;
        for (int i = 0; i < feat_num; i++) {
#if USE_MULTICLASS_NMS
            // multilabel
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[i + j * feat_num];
                if (cur_value >= m_confThreshold) {
                    YoloV9Box box;
                    box.score = cur_value;
                    box.class_id = j;
                    int c = box.class_id * max_wh;
                    float centerX = output_data[i + 0 * feat_num];
                    float centerY = output_data[i + 1 * feat_num];
                    float width = output_data[i + 2 * feat_num];
                    float height = output_data[i + 3 * feat_num];

                    box.x1 = centerX - width / 2 + c;
                    box.y1 = centerY - height / 2 + c;
                    box.x2 = box.x1 + width;
                    box.y2 = box.y1 + height;

                    yolobox_vec.push_back(box);
                }
            }
#else
            // best class
            float max_value = 0.0;
            int max_index = 0;
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[i + j * feat_num];
                if (cur_value > max_value) {
                    max_value = cur_value;
                    max_index = j;
                }
            }

            if (max_value >= m_confThreshold) {
                YoloV9Box box;
                box.score = max_value;
                box.class_id = max_index;
                int c = box.class_id * max_wh;
                float centerX = output_data[i + 0 * feat_num];
                float centerY = output_data[i + 1 * feat_num];
                float width = output_data[i + 2 * feat_num];
                float height = output_data[i + 3 * feat_num];

                box.x1 = centerX - width / 2 + c;
                box.y1 = centerY - height / 2 + c;
                box.x2 = box.x1 + width;
                box.y2 = box.y1 + height;

                yolobox_vec.push_back(box);
            }
#endif
        }
        LOG_TS(m_ts, "post 2: get detections matrix nx6 (xyxy, conf, cls)");

        LOG_TS(m_ts, "post 3: nms");
        NMS(yolobox_vec, m_nmsThreshold);

        if (yolobox_vec.size() > max_det) {
            yolobox_vec.erase(yolobox_vec.begin(), yolobox_vec.begin() + (yolobox_vec.size() - max_det));
        }

        for (int i = 0; i < yolobox_vec.size(); i++) {
            int c = yolobox_vec[i].class_id * max_wh;
            yolobox_vec[i].x1 = yolobox_vec[i].x1 - c;
            yolobox_vec[i].y1 = yolobox_vec[i].y1 - c;
            yolobox_vec[i].x2 = yolobox_vec[i].x2 - c;
            yolobox_vec[i].y2 = yolobox_vec[i].y2 - c;
        }

        float tx1 = 0, ty1 = 0;
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame.width, frame.height, m_net_w, m_net_h, &isAlignWidth);
        if (isAlignWidth) {
            ty1 = (m_net_h - (float)(frame_height * ratio)) / 2;
        } else {
            tx1 = (m_net_w - (float)(frame_width * ratio)) / 2;
        }
        for (int i = 0; i < yolobox_vec.size(); i++) {
            float centerx = ((yolobox_vec[i].x2 + yolobox_vec[i].x1) / 2 - tx1) / ratio;
            float centery = ((yolobox_vec[i].y2 + yolobox_vec[i].y1) / 2 - ty1) / ratio;
            float width = (yolobox_vec[i].x2 - yolobox_vec[i].x1) / ratio;
            float height = (yolobox_vec[i].y2 - yolobox_vec[i].y1) / ratio;
            yolobox_vec[i].x1 = centerx - width / 2;
            yolobox_vec[i].y1 = centery - height / 2;
            yolobox_vec[i].x2 = centerx + width / 2;
            yolobox_vec[i].y2 = centery + height / 2;
        }
        clip_boxes(yolobox_vec, frame_width, frame_height);

        LOG_TS(m_ts, "post 3: nms");

        detected_boxes.push_back(yolobox_vec);
    }

    return 0;
}
#if USE_NEON
int YoloV9::get_max_value_neno(float* cls_conf,float &max_value ,int & max_index,int i,int nout){

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

#if USE_OPT
int YoloV9::post_process_opt(const std::vector<bm_image>& images, std::vector<YoloV9BoxVec>& detected_boxes) {
    YoloV9BoxVec yolobox_vec;
    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);

    for (int i = 0; i < output_num; i++) {
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int min_idx = 0;
        int box_num = 0;

        // Single output
        auto out_tensor = outputTensors[min_idx];
        m_class_num = out_tensor->get_shape()->dims[2] - mask_num - 4;
        int feat_num = out_tensor->get_shape()->dims[1];
        int nout = m_class_num + mask_num + 4;
        float* output_data = nullptr;
        std::vector<float> decoded_data;

        LOG_TS(m_ts, "post 1: get output");
        assert(box_num == 0 || box_num == out_tensor->get_shape()->dims[2]);
        box_num = out_tensor->get_shape()->dims[2];
        output_data = (float*)out_tensor->get_cpu_data() + batch_idx * feat_num * (m_class_num + mask_num + 4);
        LOG_TS(m_ts, "post 1: get output");

        // Candidates
        LOG_TS(m_ts, "post 2: get detections matrix nx6 (xyxy, conf, cls)");
        float* cls_conf = output_data + 4 ;
        for (int i = 0; i < feat_num; i++) {
#if USE_MULTICLASS_NMS
            // multilabel
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[i*nout + j];
                if (cur_value >= m_confThreshold) {
                    YoloV9Box box;
                    box.score = cur_value;
                    box.class_id = j;
                    int c = box.class_id * max_wh;
                    float centerX = output_data[i * nout ];
                    float centerY = output_data[i * nout + 1];
                    float width = output_data[i * nout +  2];
                    float height = output_data[i * nout +  3];

                    box.x1 = centerX - width / 2 + c;
                    box.y1 = centerY - height / 2 + c;
                    box.x2 = box.x1 + width;
                    box.y2 = box.y1 + height;

                    yolobox_vec.push_back(box);
                }
            }
#else
            // best class
            float max_value = 0.0;
            int max_index = 0;
#if USE_NEON
            get_max_value_neon(cls_conf,max_value , max_index,i,nout);
#else
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[i * nout + j];
                if (cur_value > max_value) {
                    max_value = cur_value;
                    max_index = j;
                }
            }
#endif
            if (max_value >= m_confThreshold) {
                YoloV9Box box;
                box.score = max_value;
                box.class_id = max_index;
                int c = box.class_id * max_wh;
                float centerX = output_data[i* nout ];
                float centerY = output_data[i * nout + 1 ];
                float width = output_data[i * nout + 2 ];
                float height = output_data[i * nout + 3 ];

                box.x1 = centerX - width / 2 + c;
                box.y1 = centerY - height / 2 + c;
                box.x2 = box.x1 + width;
                box.y2 = box.y1 + height;

                yolobox_vec.push_back(box);
            }
#endif
        }
        LOG_TS(m_ts, "post 2: get detections matrix nx6 (xyxy, conf, cls)");

        LOG_TS(m_ts, "post 3: nms");
        NMS(yolobox_vec, m_nmsThreshold);

        if (yolobox_vec.size() > max_det) {
            yolobox_vec.erase(yolobox_vec.begin(), yolobox_vec.begin() + (yolobox_vec.size() - max_det));
        }

        for (int i = 0; i < yolobox_vec.size(); i++) {
            int c = yolobox_vec[i].class_id * max_wh;
            yolobox_vec[i].x1 = yolobox_vec[i].x1 - c;
            yolobox_vec[i].y1 = yolobox_vec[i].y1 - c;
            yolobox_vec[i].x2 = yolobox_vec[i].x2 - c;
            yolobox_vec[i].y2 = yolobox_vec[i].y2 - c;
        }

        float tx1 = 0, ty1 = 0;
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame.width, frame.height, m_net_w, m_net_h, &isAlignWidth);
        if (isAlignWidth) {
            ty1 = (m_net_h - (float)(frame_height * ratio)) / 2;
        } else {
            tx1 = (m_net_w - (float)(frame_width * ratio)) / 2;
        }
        for (int i = 0; i < yolobox_vec.size(); i++) {
            float centerx = ((yolobox_vec[i].x2 + yolobox_vec[i].x1) / 2 - tx1) / ratio;
            float centery = ((yolobox_vec[i].y2 + yolobox_vec[i].y1) / 2 - ty1) / ratio;
            float width = (yolobox_vec[i].x2 - yolobox_vec[i].x1) / ratio;
            float height = (yolobox_vec[i].y2 - yolobox_vec[i].y1) / ratio;
            yolobox_vec[i].x1 = centerx - width / 2;
            yolobox_vec[i].y1 = centery - height / 2;
            yolobox_vec[i].x2 = centerx + width / 2;
            yolobox_vec[i].y2 = centery + height / 2;
        }
        clip_boxes(yolobox_vec, frame_width, frame_height);

        LOG_TS(m_ts, "post 3: nms");

        detected_boxes.push_back(yolobox_vec);
    }

    return 0;
}
#endif

void YoloV9::clip_boxes(YoloV9BoxVec& yolobox_vec, int src_w, int src_h) {
    for (int i = 0; i < yolobox_vec.size(); i++) {
        if (yolobox_vec[i].x1 < 0)
            yolobox_vec[i].x1 = 0;
        else if (yolobox_vec[i].x1 > src_w)
            yolobox_vec[i].x1 = src_w;
        if (yolobox_vec[i].y1 < 0)
            yolobox_vec[i].y1 = 0;
        else if (yolobox_vec[i].y1 > src_h)
            yolobox_vec[i].y1 = src_h;
        if (yolobox_vec[i].x2 < 0)
            yolobox_vec[i].x2 = 0;
        else if (yolobox_vec[i].x2 > src_w)
            yolobox_vec[i].x2 = src_w;
        if (yolobox_vec[i].y2 < 0)
            yolobox_vec[i].y2 = 0;
        else if (yolobox_vec[i].y2 > src_h)
            yolobox_vec[i].y2 = src_h;
    }
}

void YoloV9::xywh2xyxy(YoloV9BoxVec& xyxyboxes, std::vector<std::vector<float>> box) {
    for (int i = 0; i < box.size(); i++) {
        YoloV9Box tmpbox;
        tmpbox.x1 = box[i][0] - box[i][2] / 2;
        tmpbox.y1 = box[i][1] - box[i][3] / 2;
        tmpbox.x2 = box[i][0] + box[i][2] / 2;
        tmpbox.y2 = box[i][1] + box[i][3] / 2;
        xyxyboxes.push_back(tmpbox);
    }
}

void YoloV9::NMS(YoloV9BoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YoloV9Box& a, const YoloV9Box& b) { return a.score < b.score; });

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

void YoloV9::draw_bmcv(bm_handle_t& handle,
                       int classId,
                       float conf,
                       int left,
                       int top,
                       int width,
                       int height,
                       bm_image& frame,
                       bool put_text_flag)  // Draw the predicted bounding box
{
    if (conf < 0.25) return;
    int colors_num = colors.size();
    //Draw a rectangle displaying the bounding box
    bmcv_rect_t rect;
    rect.start_x = MIN(MAX(left, 0), frame.width);
    rect.start_y = MIN(MAX(top, 0), frame.height);
    rect.crop_w = MAX(MIN(width, frame.width - rect.start_x), 0);
    rect.crop_h = MAX(MIN(height, frame.height - rect.start_y), 0);
    int thickness = 2;
    if(rect.crop_w <= thickness * 2 || rect.crop_h <= thickness * 2){
            std::cout << "width or height too small, this rect will not be drawed: " <<
                "[" << rect.start_x << ", "<< rect.start_y << ", " << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
        }
    else{
        bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
    }
    if (put_text_flag){
        //Get the label for the class name and its confidence
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << conf;
        std::string label = m_class_names[classId] + ":" + stream.str();
        bmcv_point_t org = {left, top};
        bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]};
        float fontScale = 2;
        if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
        std::cout << "bmcv put text error !!!" << std::endl;
        }
    }
}
