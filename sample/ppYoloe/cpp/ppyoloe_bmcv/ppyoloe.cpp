//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ppyoloe.hpp"
#include "bmruntime_cpp.h"
#include "bmruntime_interface.h"
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

ppYoloe::ppYoloe(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
    std::cout << "ppYoloe ctor .." << std::endl;
}

ppYoloe::~ppYoloe() {
    std::cout << "ppYoloe dtor ..." << std::endl;
    bm_free_device(m_bmContext->handle(), m_input_ratio.device_mem);
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
}

int ppYoloe::Init(float confThresh, float nmsThresh, const std::string& coco_names_file) {
    m_confThreshold = confThresh;
    m_nmsThreshold  = nmsThresh;
    std::ifstream ifs(coco_names_file);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            line = line.substr(0, line.length() - 1);
            m_class_names.push_back(line);
        }
    }
    std::cout << "===============================" << std::endl;
    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    // 2. get input
    max_batch = m_bmNetwork->maxBatch();

    auto input_ratio = m_bmNetwork->inputTensor(0);
    std::shared_ptr<BMNNTensor> input_img = m_bmNetwork->inputTensor(1);

    m_net_h = input_ratio->get_shape()->dims[2];
    m_net_w = input_ratio->get_shape()->dims[3];

    bm_status_t status;
    auto net_info = bmrt_get_network_info(m_bmContext->bmrt(), m_bmContext->network_name(0).c_str());

    status = bm_malloc_device_byte(m_bmContext->handle(), &m_input_ratio.device_mem, net_info->max_input_bytes[1]);
    m_input_ratio.dtype = input_img->get_dtype();
    m_input_ratio.st_mode= BM_STORE_1N;
    m_input_ratio.shape = {2, {net_info->stages[0].input_shapes[1].dims[0], 2}};

    // 3. get output
    m_output_num = m_bmNetwork->outputTensorNum();

    // 4. Initialize bmimage
    m_resized_imgs.resize(max_batch);
    m_min_max_imgs.resize(max_batch);
    m_standard_imgs.resize(max_batch);

    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w,64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for(int i=0; i<max_batch; i++){
        auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (input_ratio->get_dtype() == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_min_max_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);
    ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_standard_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    // 5.converto
    float input_scale = input_ratio->get_scale();
    input_scale = input_scale * 1.0 / 255.f;
    min_max_scaler.alpha_0 = input_scale;
    min_max_scaler.beta_0 = 0;
    min_max_scaler.alpha_1 = input_scale;
    min_max_scaler.beta_1 = 0;
    min_max_scaler.alpha_2 = input_scale;
    min_max_scaler.beta_2 = 0;

    standard_scaler.alpha_0 = 1 / 0.229;
    standard_scaler.beta_0 =  -0.485 / 0.229;
    standard_scaler.alpha_1 = 1 / 0.224;
    standard_scaler.beta_1 = -0.456 / 0.224;
    standard_scaler.alpha_2 = 1 / 0.225;
    standard_scaler.beta_2 = -0.406 / 0.225;

    return 0;
}

void ppYoloe::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int ppYoloe::batch_size() {
    return max_batch;
};

int ppYoloe::Detect(const std::vector<bm_image>& input_images, std::vector<ppYoloeBoxVec>& boxes) {
    int ret = 0;
    // 1. preprocess
    m_ts->save("ppyoloe preprocess", max_batch);
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    m_ts->save("ppyoloe preprocess", max_batch);

    // 2. forward
    m_ts->save("ppyoloe inference", max_batch);
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("ppyoloe inference", max_batch);

    // 3. post process
    m_ts->save("ppyoloe postprocess", max_batch);
    ret = post_process(input_images, boxes);
    CV_Assert(ret == 0);
    m_ts->save("ppyoloe postprocess", max_batch);
    return ret;
}

int ppYoloe::pre_process(const std::vector<bm_image>& images) {
    std::shared_ptr<BMNNTensor> input_img = m_bmNetwork->inputTensor(0);
    std::shared_ptr<BMNNTensor> input_ratio = m_bmNetwork->inputTensor(1);
    int image_n = images.size();

    //1. resize image
    int ret = 0;
    float *ratio_input = new float[2*image_n];
    for(int i = 0; i < image_n; ++i) {
        std::vector<float> ratio = get_img_ratio(images[i].width, images[i].height, m_net_w, m_net_h);

        ratio_input[2*i] = ratio[0];
        ratio_input[2*i+1] = ratio[1];

        auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i], &m_resized_imgs[i]);

        assert(BM_SUCCESS == ret);

    #if DUMP_FILE
        cv::Mat resized_img;
        cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
        std::string fname = cv::format("resized_img_%d.jpg", i);
        cv::imwrite(fname, resized_img);
    #endif
    }

    bm_memcpy_s2d_partial(m_bmContext->handle(), m_input_ratio.device_mem,
                          (void *)ratio_input, bmrt_tensor_bytesize(&m_input_ratio));

    delete[] ratio_input;

    input_ratio->set_device_mem(&m_input_ratio.device_mem);
    input_ratio->set_shape_by_dim(0, image_n); // set real batch number

    //2. converto
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, min_max_scaler, m_resized_imgs.data(), m_min_max_imgs.data());
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, standard_scaler, m_min_max_imgs.data(), m_standard_imgs.data());
    CV_Assert(ret == 0);

    //3. attach to tensor
    if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n);
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_standard_imgs.data(), &input_dev_mem);
    input_img->set_device_mem(&input_dev_mem);
    input_img->set_shape_by_dim(0, image_n);  // set real batch number
    return 0;
}

std::vector<float> ppYoloe::get_img_ratio(int src_w, int src_h, int dst_w, int dst_h){
  float r_h = (float)dst_h / src_h;
  float r_w = (float)dst_w / src_w;
  return std::vector<float>{r_h, r_w};
}

int ppYoloe::post_process(const std::vector<bm_image>& images, std::vector<ppYoloeBoxVec>& detected_boxes) {

    // postprocess
    ppYoloeBoxVec yolobox_vec;

    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(m_output_num);

    for(int i=0; i<m_output_num; i++){
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx){
        LOG_TS(m_ts, "post 1: get outputs");
        yolobox_vec.clear();

        auto out_score = outputTensors[0];
        auto out_coordinate = outputTensors[1];
        m_class_num = out_score->get_shape()->dims[1];

        // float* output_data = nullptr;
        float *out_score_ptr = nullptr;
        float *out_coordinate_ptr = nullptr;

        // box_num = 8400
        int box_num = out_coordinate->get_shape()->dims[1];
        out_score_ptr = (float*)out_score->get_cpu_data() + batch_idx * box_num * m_class_num;
        out_coordinate_ptr = (float*)out_coordinate->get_cpu_data() + batch_idx * box_num * 4;

        LOG_TS(m_ts, "post 1: get outputs");

        LOG_TS(m_ts, "post 2: filter boxes");
        for (int i = 0; i < box_num; i++) {

            float *ptr0 = out_score_ptr + i;
            float *ptr1 = out_coordinate_ptr + i*4;

            int class_id = argmax_interval(&ptr0[0],m_class_num,box_num);

            float confidence = ptr0[class_id*box_num];

            if (confidence > m_confThreshold) {

                ppYoloeBox box;
                box.x = int(round(ptr1[0]));
                box.y = int(round(ptr1[1]));
                int x2 =int(round(ptr1[2])) ;
                int y2 =int(round(ptr1[3])) ;
                box.width = x2 -box.x ;
                box.height = y2 - box.y;

                box.class_id = class_id;
                box.score = confidence;
                yolobox_vec.push_back(box);
            }
        }
        LOG_TS(m_ts, "post 2: filter boxes");

        // NMS
        LOG_TS(m_ts, "post 3: nms");
#if USE_MULTICLASS_NMS
        std::vector<ppYoloeBoxVec> class_vec(m_class_num);
        for (auto& box : yolobox_vec) {
            class_vec[box.class_id].push_back(box);
        }
        for (auto& cls_box : class_vec) {
            NMS(cls_box, m_nmsThreshold);
        }
        yolobox_vec.clear();
        for (auto& cls_box : class_vec) {
            yolobox_vec.insert(yolobox_vec.end(), cls_box.begin(), cls_box.end());
        }
#else
        NMS(yolobox_vec, m_nmsThreshold);
#endif
        detected_boxes.push_back(yolobox_vec);
        LOG_TS(m_ts, "post 3: nms");
    }

    return 0;
}

int ppYoloe::argmax_interval(float *data, int class_num, int box_num){
    float max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < class_num; ++i) {
        float value = data[i*box_num];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }
    return max_index;
}

void ppYoloe::NMS(ppYoloeBoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const ppYoloeBox& a, const ppYoloeBox& b) { return a.score < b.score; });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        areas[i] = dets[i].width * dets[i].height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].x, dets[i].x);
            float top = std::max(dets[index].y, dets[i].y);
            float right = std::min(dets[index].x + dets[index].width, dets[i].x + dets[i].width);
            float bottom = std::min(dets[index].y + dets[index].height, dets[i].y + dets[i].height);
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

void ppYoloe::draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int width, int height, bm_image& frame, bool put_text_flag)   // Draw the predicted bounding box
{
  if (conf < 0.25) return;
  int colors_num = colors.size();
  //Draw a rectangle displaying the bounding box
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(left, 0), frame.width);
  rect.start_y = MIN(MAX(top, 0), frame.height);
  rect.crop_w = MAX(MIN(width, frame.width - left), 0);
  rect.crop_h = MAX(MIN(height, frame.height - top), 0);
  bmcv_image_draw_rectangle(handle, frame, 1, &rect, 3, colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
  if (put_text_flag){
    //Get the label for the class name and its confidence
    std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
    bmcv_point_t org = {left, top};
    bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]};
    int thickness = 2;
    float fontScale = 2;
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;
    }
  }
}