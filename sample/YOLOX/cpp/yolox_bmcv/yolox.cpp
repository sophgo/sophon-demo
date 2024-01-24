//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolox.hpp"
#include <fstream>
#include <vector>
#include <string>
#define USE_ASPECT_RATIO 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloX::YoloX(std::shared_ptr<BMNNContext> context):m_bmContext(context) {
    std::cout << "YoloX ctor .." << std::endl;
}

YoloX::~YoloX() {
    std::cout << "YoloX dtor ..." << std::endl;
}

int YoloX::Init(float confThresh, float nmsThresh, const std::string& coco_names_file) {
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
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    // 3. get output
    output_num = m_bmNetwork->outputTensorNum();
    min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;


    // 4. Initialize bmimage
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);

    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w,64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for(int i=0; i<max_batch; i++){
        auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8){
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_BGR_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    // 5.converto
    float input_scale = tensor->get_scale();
    input_scale = input_scale * 1.0 / 1.f;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = 0;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = 0;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = 0;
    return 0;
}

void YoloX::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int YoloX::batch_size() {
    return max_batch;
};

int YoloX::Detect(const std::vector<bm_image>& input_images, std::vector<YoloXBoxVec>& boxes) {
    int ret = 0;
    // 1. preprocess
    LOG_TS(m_ts, "yolox preprocess");
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolox preprocess");

    
    // 2. forward
    LOG_TS(m_ts, "yolox inference");
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolox inference");

    // 3. post process
    LOG_TS(m_ts, "yolox postprocess");
    ret = post_process(input_images, boxes, false);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolox postprocess");
    return ret;
}

int YoloX::pre_process(const std::vector<bm_image>& images) {
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    int image_n = images.size();

    //1. resize image
    int ret = 0;
    for(int i = 0; i < image_n; ++i) {
        bm_image image1 = images[i];
        bm_image image_aligned;
        bool need_copy = image1.width & (64-1);
        if(need_copy){
        int stride1[3], stride2[3];
        bm_image_get_stride(image1, stride1);
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);
        bm_image_create(m_bmContext->handle(), image1.height, image1.width,
            image1.image_format, image1.data_type, &image_aligned, stride2);

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
        padding_attr.dst_crop_h = images[i].height*ratio;
        padding_attr.dst_crop_w = m_net_w;

        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        }else{
        padding_attr.dst_crop_h = m_net_h;
        padding_attr.dst_crop_w = images[i].width*ratio;

        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        }

        bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
        auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
            &padding_attr, &crop_rect);
    #else
        auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i], &m_resized_imgs[i]);
    #endif
        assert(BM_SUCCESS == ret);
        
    #if DUMP_FILE
        cv::Mat resized_img;
        cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
        std::string fname = cv::format("resized_img_%d.jpg", i);
        cv::imwrite(fname, resized_img);
    #endif
        if(need_copy) bm_image_destroy(image_aligned);
    }
    
    //2. converto
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
    CV_Assert(ret == 0);

    //3. attach to tensor
    if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
    return 0;
}

float YoloX::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w){
    *pIsAligWidth = true;
    ratio = r_w;
  }
  else{
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

int YoloX::post_process(const std::vector<bm_image>& images, std::vector<YoloXBoxVec>& detected_boxes, bool p6 ) {
    // p6
    std::vector<int> strides = {8,16,32};
    if (p6){
        strides.push_back(64);
    }

    // init grids
    int outlen_diml = 0 ;
    for (int i =0;i<strides.size();++i){
        int layer_w = m_net_w / strides[i];
        int layer_h = m_net_h / strides[i];
        outlen_diml += layer_h * layer_w;
    }
    int* grids_x_          = new int[outlen_diml];
    int* grids_y_          = new int[outlen_diml];
    int* expanded_strides_ = new int[outlen_diml];

    int channel_len = 0;
    for (int i=0;i<strides.size();++i){
        int layer_w = m_net_w / strides[i];
        int layer_h = m_net_h / strides[i];
        for (int m = 0; m < layer_h; ++m){
            for (int n = 0; n < layer_w; ++n){
                grids_x_[channel_len + m * layer_w + n] = n;
                grids_y_[channel_len + m * layer_w + n] = m;
                expanded_strides_[channel_len + m * layer_w + n] = strides[i];
            }
        }
        channel_len += layer_w * layer_h;
    }

    // postprocess
    YoloXBoxVec yolobox_vec;
    std::vector<cv::Rect> bbox_vec;
    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
    for(int i=0; i<output_num; i++){
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }
    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx){
        LOG_TS(m_ts, "post 1: get outputs");
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width;
        int frame_height = frame.height;

        int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame.width, frame.height, m_net_w, m_net_h, &isAlignWidth);
        if (isAlignWidth) {
            ty1 = (int)((m_net_h - (int)(frame_height * ratio)) / 2);
        } else {
            tx1 = (int)((m_net_w - (int)(frame_width * ratio)) / 2);
        }
#endif

        int min_idx = 0;
        for(int i=0; i<output_num; i++){
            auto output_shape = m_bmNetwork->outputTensor(i)->get_shape();
            auto output_dims = output_shape->num_dims;
            if(min_dim > output_dims){
                min_idx = i;
                min_dim =output_dims;
            }
        }

        auto out_tensor = outputTensors[min_idx];
        int nout = out_tensor->get_shape()->dims[min_dim-1];
        m_class_num = nout - 5;

        float* output_data = nullptr;
        int box_num = out_tensor->get_shape()->dims[1];
        output_data = (float*)out_tensor->get_cpu_data() + batch_idx * box_num * nout;
        LOG_TS(m_ts, "post 1: get outputs");

        LOG_TS(m_ts, "post 2: filter boxes");
        for (int i = 0; i < box_num; i++) {
            float* ptr = output_data + i * nout;
            float score = ptr[4];
            int class_id = argmax(&ptr[5], m_class_num);
            float confidence = ptr[class_id + 5];
            if (confidence * score > m_confThreshold) {
                
                float centerX = (ptr[0] + grids_x_[i]) * expanded_strides_[i];
                float centerY = (ptr[1] + grids_y_[i]) * expanded_strides_[i];
                float width = exp(ptr[2]) * expanded_strides_[i];
                float height = exp(ptr[3]) * expanded_strides_[i];

                YoloXBox box;
                box.width = width / ratio;
                box.height = height  / ratio;

                // left,top
                box.x = (centerX) / ratio - int(box.width/2);
                box.y = (centerY) / ratio - int(box.height/2); 

                box.class_id = class_id;
                box.score = confidence * score;
                yolobox_vec.push_back(box);
            }
        }
        LOG_TS(m_ts, "post 2: filter boxes");

        // NMS
        LOG_TS(m_ts, "post 3: nms");
#if USE_MULTICLASS_NMS
        std::vector<YoloXBoxVec> class_vec(m_class_num);
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
    
    // delete
    delete[] grids_x_;
    grids_x_ = NULL;
    delete[] grids_y_;
    grids_y_ = NULL;
    delete[] expanded_strides_;
    expanded_strides_ = NULL;


    return 0;
}

int YoloX::argmax(float* data, int num) {
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


void YoloX::NMS(YoloXBoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YoloXBox& a, const YoloXBox& b) { return a.score < b.score; });

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

void YoloX::draw_bmcv(bm_handle_t &handle, int classId, float conf, int left, int top, int width, int height, bm_image& frame, bool put_text_flag)   // Draw the predicted bounding box
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
  if(width < thickness * 2 || height < thickness * 2){
        std::cout << "width or height too small, this rect will not be drawed: " << 
              "[" << rect.start_x << ", "<< rect.start_y << ", " << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
    } 
  else{
    bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
  }
  if (put_text_flag){
    //Get the label for the class name and its confidence
    std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
    bmcv_point_t org = {left, top};
    bmcv_color_t color = {colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]};
    float fontScale = 2; 
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org, color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;   
    }
  }
}

