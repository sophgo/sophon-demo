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

#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloX::YoloX(int dev_id, std::string bmodel_file) : engine() {
    engine = std::make_shared<sail::Engine>(dev_id);
    if (!engine->load(bmodel_file)) {
        std::cout << "Engine load bmodel " << bmodel_file << "failed" << endl;
        exit(0);
    }

    std::cout << "YoloX ctor .." << std::endl;
}

YoloX::~YoloX() {
    std::cout << "YoloX dtor ..." << std::endl;
}

int YoloX::Init(float confThresh, float nmsThresh, const std::string& coco_names_file) {
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
    std::cout << "===============================" << std::endl;
    // 1. Initialize bmcv
    sail::Handle handle(engine->get_device_id());
    bmcv = std::make_shared<sail::Bmcv>(handle);
    // 2. Initialize engine
    graph_names = engine->get_graph_names();
    std::string gh_info;
    for_each(graph_names.begin(), graph_names.end(), [&](std::string& s) { gh_info += "0: " + s + "; "; });
    std::cout << "grapgh name -> " << gh_info << "\n";
    if (graph_names.size() > 1) {
        std::cout << "NetworkNumError, this net only accept one network!" << std::endl;
        exit(1);
    }

    // input names of network
    input_names = engine->get_input_names(graph_names[0]);
    assert(input_names.size() > 0);
    std::string input_tensor_names;
    for_each(input_names.begin(), input_names.end(), [&](std::string& s) { input_tensor_names += "0: " + s + "; "; });
    std::cout << "net input name -> " << input_tensor_names << "\n";
    if (input_names.size() > 1) {
        std::cout << "InputNumError, yolox has only one input!" << std::endl;
        exit(1);
    }

    // output names of network
    output_names = engine->get_output_names(graph_names[0]);
    assert(output_names.size() > 0);
    std::string output_tensor_names;
    for_each(output_names.begin(), output_names.end(),
             [&](std::string& s) { output_tensor_names += "0: " + s + "; "; });
    std::cout << "net output name -> " << output_tensor_names << "\n";

    // input shape of network 0
    input_shape = engine->get_input_shape(graph_names[0], input_names[0]);
    std::string input_tensor_shape;
    for_each(input_shape.begin(), input_shape.end(), [&](int s) { input_tensor_shape += std::to_string(s) + " "; });
    std::cout << "input tensor shape -> " << input_tensor_shape << "\n";

    // output shapes of network 0
    output_shape = engine->get_output_shape(graph_names[0], output_names[0]);
    std::string output_tensor_shape;
    for_each(output_shape.begin(), output_shape.end(), [&](int s) { output_tensor_shape += std::to_string(s) + " "; });
    std::cout << "output tensor shape -> " << output_tensor_shape << "\n";

    // data type of network input.
    input_dtype = engine->get_input_dtype(graph_names[0], input_names[0]);
    std::cout << "input dtype -> " << input_dtype << ", is fp32=" << ((input_dtype == BM_FLOAT32) ? "true" : "false")
              << "\n";

    // data type of network output.
    output_dtype = engine->get_output_dtype(graph_names[0], output_names[0]);
    std::cout << "output dtype -> " << output_dtype << ", is fp32=" << ((output_dtype == BM_FLOAT32) ? "true" : "false")
              << "\n";
    std::cout << "===============================" << std::endl;

    // 3. Initialize Network IO
    input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, false, false);
    input_tensors[input_names[0]] = input_tensor.get();

    output_tensor = std::make_shared<sail::Tensor>(handle, output_shape, output_dtype, true, true);
    output_tensors[output_names[0]] = output_tensor.get();

    engine->set_io_mode(graph_names[0], sail::SYSO);

    // Initialize net utils
    max_batch = input_shape[0];
    m_net_h = input_shape[2];
    m_net_w = input_shape[3];
    float input_scale = engine->get_input_scale(graph_names[0], input_names[0]);
    input_scale = input_scale * 1.0 / 1.f;
    ab[0] = input_scale;
    ab[1] = 0;
    ab[2] = input_scale;
    ab[3] = 0;
    ab[4] = input_scale;
    ab[5] = 0;
    return 0;
}

void YoloX::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int YoloX::batch_size() {
    return max_batch;
};

int YoloX::Detect(std::vector<sail::BMImage>& input_images, std::vector<YoloXBoxVec>& boxes) {
    int ret = 0;
    // 1. preprocess
    LOG_TS(m_ts, "yolox preprocess");
    if (input_images.size() == 4 && max_batch == 4) {
        ret = pre_process<4>(input_images);
    } else if (input_images.size() == 1 && max_batch == 1) {
        ret = pre_process(input_images[0]);
    } else {
        std::cout << "unsupport batch size!" << std::endl;
        exit(1);
    }
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolox preprocess");
    auto bmimg = bmcv->tensor_to_bm_image(*input_tensors[input_names[0]]);
    // 2. forward
    LOG_TS(m_ts, "yolox inference");
    engine->process(graph_names[0], input_tensors, output_tensors);
    LOG_TS(m_ts, "yolox inference");

    // 3. post process
    LOG_TS(m_ts, "yolox postprocess");
    ret = post_process(input_images, boxes, false);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolox postprocess");
    return ret;
}

int YoloX::pre_process(sail::BMImage& input) {
    int stride1[3], stride2[3];
    bm_image_get_stride(input.data(), stride1);  // bmcv api
    stride2[0] = FFALIGN(stride1[0], 64);
    stride2[1] = FFALIGN(stride1[1], 64);
    stride2[2] = FFALIGN(stride1[2], 64);
    sail::BMImage bgr_img(engine->get_handle(), input.height(), input.width(), FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                          stride2);
    bmcv->convert_format(input, bgr_img);
    sail::BMImage convert_img(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_BGR_PLANAR,
                              bmcv->get_bm_image_data_format(input_dtype));
    // letter box
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(input.width(), input.height(), m_net_w, m_net_h, &isAlignWidth);
    sail::PaddingAtrr pad = sail::PaddingAtrr();
    pad.set_r(114);
    pad.set_g(114);
    pad.set_b(114);
    if (isAlignWidth) {
        unsigned int th = input.height() * ratio;
        pad.set_h(th);
        pad.set_w(m_net_w);
        pad.set_sty(0);
        pad.set_stx(0);
    } else {
        pad.set_h(m_net_h);
        unsigned int tw = input.width() * ratio;
        pad.set_w(tw);
        pad.set_sty(0);
        pad.set_stx(0);
    }
    // resize imgs
    sail::BMImage resized_img =
        bmcv->vpp_resize_padding(bgr_img, m_net_w, m_net_h, pad,BMCV_INTER_LINEAR);
    
    // convert
    bmcv->convert_to(
        resized_img, convert_img,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(convert_img, *input_tensor.get());

    return 0;
}

template <std::size_t N>
int YoloX::pre_process(std::vector<sail::BMImage>& input) {
    if (input.size() != N) {
        std::cout << "Unsupport batch size!" << std::endl;
        exit(1);
    }
    std::shared_ptr<sail::BMImage> resized_imgs_vec[N];
    sail::BMImageArray<N> resized_imgs;
    sail::BMImageArray<N> convert_imgs(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_BGR_PLANAR,
                                       bmcv->get_bm_image_data_format(input_dtype));
    for (size_t i = 0; i < input.size(); ++i) {
        int stride1[3], stride2[3];
        bm_image_get_stride(input[i].data(), stride1);  // bmcv api
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);
        sail::BMImage bgr_img(engine->get_handle(), input[i].height(), input[i].width(), FORMAT_BGR_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE, stride2);
        bmcv->convert_format(input[i], bgr_img);

        // letter box
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(input[i].width(), input[i].height(), m_net_w, m_net_h, &isAlignWidth);
        sail::PaddingAtrr pad = sail::PaddingAtrr();
        pad.set_r(114);
        pad.set_g(114);
        pad.set_b(114);
        if (isAlignWidth) {
            unsigned int th = input[i].height() * ratio;
            pad.set_h(th);
            pad.set_w(m_net_w);
            pad.set_sty(0);
            pad.set_stx(0);
        } else {
            pad.set_h(m_net_h);
            unsigned int tw = input[i].width() * ratio;
            pad.set_w(tw);
            pad.set_sty(0);
            pad.set_stx(0);
        }
        resized_imgs_vec[i] = std::make_shared<sail::BMImage>(engine->get_handle(), input_shape[2], input_shape[3],
                                                              FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE);

        bmcv->vpp_crop_and_resize_padding(&bgr_img.data(), &resized_imgs_vec[i].get()->data(), 0, 0, bgr_img.width(),
                                          bgr_img.height(), m_net_w, m_net_h, pad, 1, BMCV_INTER_LINEAR);

        resized_imgs.attach_from(i, *resized_imgs_vec[i].get());
    }
    bmcv->convert_to(
        resized_imgs, convert_imgs,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(convert_imgs, *input_tensor.get());
    return 0;
}

float YoloX::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

int YoloX::post_process(std::vector<sail::BMImage>& images, std::vector<YoloXBoxVec>& detected_boxes, bool p6 = false) {
    // p6
    std::vector<int> strides = {8,16,32};
    LOG_TS(m_ts, "post 1: Init Grids");
    if (p6){
        strides.push_back(64);
    }
    // get grids
    int outlen_diml = 0 ;
    int net_w = input_shape[2];
    int net_h = input_shape[3];
    for (int i =0;i<strides.size();++i){
        int layer_w = net_w / strides[i];
        int layer_h = net_h / strides[i];
        outlen_diml += layer_h * layer_w;
    }
    int* grids_x_          = new int[outlen_diml];
    int* grids_y_          = new int[outlen_diml];
    int* expanded_strides_ = new int[outlen_diml];
    int channel_len = 0;
    for (int i=0;i<strides.size();++i){
        int layer_w = net_w / strides[i];
        int layer_h = net_h / strides[i];
        for (int m = 0; m < layer_h; ++m){
            for (int n = 0; n < layer_w; ++n){
                grids_x_[channel_len + m * layer_w + n] = n;
                grids_y_[channel_len + m * layer_w + n] = m;
                expanded_strides_[channel_len + m * layer_w + n] = strides[i];
            }
        }
        channel_len += layer_w * layer_h;
    }
    LOG_TS(m_ts, "post 1: Init Grids");

    // postprocess
    YoloXBoxVec yolobox_vec;
    std::vector<cv::Rect> bbox_vec;
    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx){
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width();
        int frame_height = frame.height();
        int tx1 = 0, ty1 = 0;
        // get ration
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame.width(), frame.height(), m_net_w, m_net_h, &isAlignWidth);
        if (isAlignWidth) {
            ty1 = (int)((m_net_h - (int)(frame_height * ratio)) / 2);
        } else {
            tx1 = (int)((m_net_w - (int)(frame_width * ratio)) / 2);
        }

        auto out_tensor = output_tensor;
        int nout = out_tensor->shape()[2];
        m_class_num = nout - 5;

        float* output_data = nullptr;
        int box_num = out_tensor->shape()[1];
        output_data = reinterpret_cast<float*>(output_tensor->sys_data()) + batch_idx * box_num * nout;
        
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

                // convert to left,top
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


// Draw the predicted bounding box
void YoloX::draw_bmcv(int classId,float conf,int left,int top,int width,int height,sail::BMImage& frame,bool put_text_flag)  
{
    if (conf < 0.25)
        return;
    int colors_num = colors.size();
    // Draw a rectangle displaying the bounding box
    int start_x = MIN(MAX(left, 0), frame.width());
    int start_y = MIN(MAX(top, 0), frame.height());
    int crop_w = MAX(MIN(width, frame.width() - left), 0);
    int crop_h = MAX(MIN(height, frame.height() - top), 0);
    auto color_tuple = std::make_tuple(colors[classId % colors_num][2], colors[classId % colors_num][1],
                                       colors[classId % colors_num][0]);
    bmcv->rectangle(frame, start_x, start_y, crop_w, crop_h, color_tuple, 3);
    if (put_text_flag) {  // only support YUV420P, puttext not used here.
        std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
        if (BM_SUCCESS != bmcv->putText(frame, label.c_str(), left, top, color_tuple, 2, 2)) {
            std::cout << "bmcv put text error !!!" << std::endl;
        }
    }
}
