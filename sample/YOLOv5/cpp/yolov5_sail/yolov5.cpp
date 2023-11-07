//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov5.hpp"
#include <fstream>
#include <vector>
#include <string>
#define USE_ASPECT_RATIO 1
#define RESIZE_STRATEGY BMCV_INTER_NEAREST
#define USE_BMCV_VPP_CONVERT 1
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloV5::YoloV5(int dev_id, std::string bmodel_file, bool use_cpu_opt) : engine(), use_cpu_opt(use_cpu_opt) {
    engine = std::make_shared<sail::Engine>(dev_id);
    if (!engine->load(bmodel_file)) {
        std::cout << "Engine load bmodel " << bmodel_file << "failed" << endl;
        exit(0);
    }

    std::cout << "YoloV5 ctor .." << std::endl;
}

YoloV5::~YoloV5() {
    std::cout << "YoloV5 dtor ..." << std::endl;
}

int YoloV5::Init(float confThresh, float nmsThresh, const std::string& coco_names_file) {
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
        std::cout << "InputNumError, yolov5 has only one input!" << std::endl;
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
    output_shape.resize(output_names.size());
    for (int i = 0; i < output_names.size(); i++) {
        output_shape[i] = engine->get_output_shape(graph_names[0], output_names[i]);
        std::string output_tensor_shape;
        for_each(output_shape[i].begin(), output_shape[i].end(),
                 [&](int s) { output_tensor_shape += std::to_string(s) + " "; });
        std::cout << "output tensor " << i << " shape -> " << output_tensor_shape << "\n";
    }

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
    output_tensor.resize(output_names.size());
    for (int i = 0; i < output_names.size(); i++) {
        output_tensor[i] = std::make_shared<sail::Tensor>(handle, output_shape[i], output_dtype, true, true);
        output_tensors[output_names[i]] = output_tensor[i].get();
    }
    engine->set_io_mode(graph_names[0], sail::SYSO);

    // Initialize net utils
    max_batch = input_shape[0];
    m_net_h = input_shape[2];
    m_net_w = input_shape[3];
    min_dim = output_shape[0].size();
    float input_scale = engine->get_input_scale(graph_names[0], input_names[0]);
    input_scale = input_scale * 1.0 / 255.f;
    ab[0] = input_scale;
    ab[1] = 0;
    ab[2] = input_scale;
    ab[3] = 0;
    ab[4] = input_scale;
    ab[5] = 0;
    return 0;
}

void YoloV5::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int YoloV5::batch_size() {
    return max_batch;
};

int YoloV5::Detect(std::vector<sail::BMImage>& input_images, std::vector<YoloV5BoxVec>& boxes) {
    int ret = 0;
    // 1. preprocess
    LOG_TS(m_ts, "yolov5 preprocess");
    if (input_images.size() == 4 && max_batch == 4) {
        ret = pre_process<4>(input_images);
    } else if (input_images.size() == 1 && max_batch == 1) {
        ret = pre_process(input_images[0]);
    } else {
        std::cout << "unsupport batch size!" << std::endl;
        exit(1);
    }
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolov5 preprocess");
    auto bmimg = bmcv->tensor_to_bm_image(*input_tensors[input_names[0]]);
    // 2. forward
    LOG_TS(m_ts, "yolov5 inference");
    engine->process(graph_names[0], input_tensors, output_tensors);
    LOG_TS(m_ts, "yolov5 inference");

    // 3. post process
    LOG_TS(m_ts, "yolov5 postprocess");
    if (use_cpu_opt)
        ret = post_process_cpu_opt(input_images, boxes);
    else
        ret = post_process(input_images, boxes);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolov5 postprocess");
    return ret;
}

int YoloV5::pre_process(sail::BMImage& input) {
    int stride1[3], stride2[3];
    bm_image_get_stride(input.data(), stride1);  // bmcv api
    stride2[0] = FFALIGN(stride1[0], 64);
    stride2[1] = FFALIGN(stride1[1], 64);
    stride2[2] = FFALIGN(stride1[2], 64);
    sail::BMImage rgb_img(engine->get_handle(), input.height(), input.width(), FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                          stride2);
    bmcv->convert_format(input, rgb_img);
    sail::BMImage convert_img(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                              bmcv->get_bm_image_data_format(input_dtype));
#if USE_ASPECT_RATIO
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
        int ty1 = (int)((m_net_h - th) / 2);
        pad.set_sty(ty1);
        pad.set_stx(0);
    } else {
        pad.set_h(m_net_h);
        unsigned int tw = input.width() * ratio;
        pad.set_w(tw);

        int tx1 = (int)((m_net_w - tw) / 2);
        pad.set_sty(0);
        pad.set_stx(tx1);
    }
#if USE_BMCV_VPP_CONVERT
    // Using BMCV api, align with yolov5_bmcv.
    sail::BMImage resized_img(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE);
    bmcv_rect_t rect;
    rect.start_x = 0;
    rect.start_y = 0;
    rect.crop_w = input.width();
    rect.crop_h = input.height();
    bmcv_padding_atrr_t padding;
    padding.dst_crop_stx = pad.dst_crop_stx;
    padding.dst_crop_sty = pad.dst_crop_sty;
    padding.dst_crop_w = pad.dst_crop_w;
    padding.dst_crop_h = pad.dst_crop_h;
    padding.if_memset = 1;
    padding.padding_r = pad.padding_r;
    padding.padding_g = pad.padding_g;
    padding.padding_b = pad.padding_b;
    auto ret = bmcv_image_vpp_convert_padding(engine->get_handle().data(), 1, rgb_img.data(), &resized_img.data(),
                                              &padding, &rect, RESIZE_STRATEGY);
    assert(ret == 0);
#else
    sail::BMImage resized_img =
        bmcv->vpp_crop_and_resize_padding(rgb_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, pad);
#endif
#else
    sail::BMImage resized_img =
        bmcv->crop_and_resize(rgb_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
#endif
    bmcv->convert_to(
        resized_img, convert_img,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(convert_img, *input_tensor.get());
    return 0;
}

template <std::size_t N>
int YoloV5::pre_process(std::vector<sail::BMImage>& input) {
    if (input.size() != N) {
        std::cout << "Unsupport batch size!" << std::endl;
        exit(1);
    }
    std::shared_ptr<sail::BMImage> resized_imgs_vec[N];
    sail::BMImageArray<N> resized_imgs;
    sail::BMImageArray<N> convert_imgs(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                                       bmcv->get_bm_image_data_format(input_dtype));
    for (size_t i = 0; i < input.size(); ++i) {
        int stride1[3], stride2[3];
        bm_image_get_stride(input[i].data(), stride1);  // bmcv api
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);
        sail::BMImage rgb_img(engine->get_handle(), input[i].height(), input[i].width(), FORMAT_RGB_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE, stride2);
        bmcv->convert_format(input[i], rgb_img);

#if USE_ASPECT_RATIO
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
            int ty1 = (int)((m_net_h - th) / 2);
            pad.set_sty(ty1);
            pad.set_stx(0);
        } else {
            pad.set_h(m_net_h);
            unsigned int tw = input[i].width() * ratio;
            pad.set_w(tw);
            int tx1 = (int)((m_net_w - tw) / 2);
            pad.set_sty(0);
            pad.set_stx(tx1);
        }
        resized_imgs_vec[i] = std::make_shared<sail::BMImage>(engine->get_handle(), input_shape[2], input_shape[3],
                                                              FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
#if USE_BMCV_VPP_CONVERT
        // Using BMCV api, align with yolov5_bmcv.
        bmcv_rect_t rect;
        rect.start_x = 0;
        rect.start_y = 0;
        rect.crop_w = input[i].width();
        rect.crop_h = input[i].height();
        bmcv_padding_atrr_t padding;
        padding.dst_crop_stx = pad.dst_crop_stx;
        padding.dst_crop_sty = pad.dst_crop_sty;
        padding.dst_crop_w = pad.dst_crop_w;
        padding.dst_crop_h = pad.dst_crop_h;
        padding.if_memset = 1;
        padding.padding_r = pad.padding_r;
        padding.padding_g = pad.padding_g;
        padding.padding_b = pad.padding_b;
        auto ret = bmcv_image_vpp_convert_padding(engine->get_handle().data(), 1, rgb_img.data(),
                                                  &resized_imgs_vec[i].get()->data(), &padding, &rect);
        assert(ret == 0);
#else
        bmcv->vpp_crop_and_resize_padding(&rgb_img.data(), &resized_imgs_vec[i].get()->data(), 0, 0, rgb_img.width(),
                                          rgb_img.height(), m_net_w, m_net_h, pad);
#endif
        resized_imgs.attach_from(i, *resized_imgs_vec[i].get());
#else
        sail::BMImage resized_img =
            bmcv->crop_and_resize(rgb_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
        resized_imgs.CopyFrom(i, resized_img);
#endif
    }
    bmcv->convert_to(
        resized_imgs, convert_imgs,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(convert_imgs, *input_tensor.get());
    return 0;
}

float YoloV5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
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

int YoloV5::post_process(std::vector<sail::BMImage>& images, std::vector<YoloV5BoxVec>& detected_boxes) {
    YoloV5BoxVec yolobox_vec;
    std::vector<cv::Rect> bbox_vec;

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width();
        int frame_height = frame.height();

        int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
        bool isAlignWidth = false;
        float ratio = get_aspect_scaled_ratio(frame.width(), frame.height(), m_net_w, m_net_h, &isAlignWidth);
        if (isAlignWidth) {
            ty1 = (int)((m_net_h - (int)(frame_height * ratio)) / 2);
        } else {
            tx1 = (int)((m_net_w - (int)(frame_width * ratio)) / 2);
        }
#endif

        int min_idx = 0;
        int box_num = 0;
        for (int i = 0; i < output_names.size(); i++) {
            assert(output_shape[i].size() == 3 || output_shape[i].size() == 5);
            if (output_shape[i].size() == 5) {
                box_num += output_shape[i][1] * output_shape[i][2] * output_shape[i][3];
            }

            if (min_dim > output_shape[i].size()) {
                min_idx = i;
                min_dim = output_shape[i].size();
            }
        }

        auto out_tensor = output_tensor[min_idx];
        int nout = out_tensor->shape()[min_dim - 1];
        m_class_num = nout - 5;

        float* output_data = nullptr;
        std::vector<float> decoded_data;

        if (min_dim == 3 && output_names.size() != 1) {
            std::cout << "--> WARNING: the current bmodel has redundant outputs" << std::endl;
            std::cout << "             you can remove the redundant outputs to improve performance" << std::endl;
            std::cout << std::endl;
        }

        if (min_dim == 5) {
            LOG_TS(m_ts, "post 1: get output and decode");
            const std::vector<std::vector<std::vector<int>>> anchors{
                {{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}};
            const int anchor_num = anchors[0].size();
            assert(output_names.size() == (int)anchors.size());
            assert(box_num > 0);
            if ((int)decoded_data.size() != box_num * nout) {
                decoded_data.resize(box_num * nout);
            }
            float* dst = decoded_data.data();
            for (int tidx = 0; tidx < output_names.size(); ++tidx) {
                int feat_c = output_tensor[tidx]->shape()[1];
                int feat_h = output_tensor[tidx]->shape()[2];
                int feat_w = output_tensor[tidx]->shape()[3];
                int area = feat_h * feat_w;
                assert(feat_c == anchor_num);
                int feature_size = feat_h * feat_w * nout;
                output_data = reinterpret_cast<float*>(output_tensor[tidx]->sys_data());
                float* tensor_data = output_data + batch_idx * feat_c * area * nout;
                for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
                    float* ptr = tensor_data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
                        dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * m_net_h;
                        dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                        dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                        dst[4] = sigmoid(ptr[4]);
                        float score = dst[4];
                        if (score > m_confThreshold) {
                            for (int d = 5; d < nout; d++) {
                                dst[d] = sigmoid(ptr[d]);
                            }
                        }
                        dst += nout;
                        ptr += nout;
                    }
                }
            }
            output_data = decoded_data.data();
            LOG_TS(m_ts, "post 1: get output and decode");
        } else {
            LOG_TS(m_ts, "post 1: get output");
            assert(box_num == 0 || box_num == out_tensor->shape()[1]);
            box_num = out_tensor->shape()[1];
            output_data = reinterpret_cast<float*>(output_tensor[0]->sys_data()) + batch_idx * box_num * nout;
            LOG_TS(m_ts, "post 1: get output");
        }

        LOG_TS(m_ts, "post 2: filter boxes");
        int max_wh = 7680;
        bool agnostic = false;
        for (int i = 0; i < box_num; i++) {
            float* ptr = output_data+i*nout;
            float score = ptr[4];
            if (score > m_confThreshold) {
#if USE_MULTICLASS_NMS
                for (int j = 0; j < m_class_num; j++) {
                    float confidence = ptr[5 + j];
                    int class_id = j;
                    if (confidence * score > m_confThreshold)
                    {
                        float centerX = ptr[0];
                        float centerY = ptr[1];
                        float width = ptr[2];
                        float height = ptr[3];

                        YoloV5Box box;
                        if (!agnostic)
                            box.x = centerX - width / 2 + class_id * max_wh;
                        else
                            box.x = centerX - width / 2;
                        if (box.x < 0) box.x = 0;
                        if (!agnostic)
                            box.y = centerY - height / 2 + class_id * max_wh;
                        else
                            box.y = centerY - height / 2;
                        if (box.y < 0) box.y = 0;
                        box.width = width;
                        box.height = height;
                        box.class_id = class_id;
                        box.score = confidence * score;
                        yolobox_vec.push_back(box);
                    }
                }
#else
                int class_id = argmax(&ptr[5], m_class_num);
                float confidence = ptr[class_id + 5];
                if (confidence * score > m_confThreshold)
                {
                    float centerX = ptr[0];
                    float centerY = ptr[1];
                    float width = ptr[2];
                    float height = ptr[3];

                    YoloV5Box box;
                    if (!agnostic)
                    box.x = centerX - width / 2 + class_id * max_wh;
                    else
                    box.x = centerX - width / 2;
                    if (box.x < 0) box.x = 0;
                    if (!agnostic)
                    box.y = centerY - height / 2 + class_id * max_wh;
                    else
                    box.y = centerY - height / 2;
                    if (box.y < 0) box.y = 0;
                    box.width = width;
                    box.height = height;
                    box.class_id = class_id;
                    box.score = confidence * score;
                    yolobox_vec.push_back(box);
                }
#endif
            }
        }
        LOG_TS(m_ts, "post 2: filter boxes");

        LOG_TS(m_ts, "post 3: nms");
        NMS(yolobox_vec, m_nmsThreshold);
        if (!agnostic)
            for (auto& box : yolobox_vec){
                box.x -= box.class_id * max_wh;
                box.y -= box.class_id * max_wh;
                box.x = (box.x - tx1) / ratio;
                box.y = (box.y - ty1) / ratio;
                box.width = (box.width) / ratio;
                box.height = (box.height) / ratio;
            }
        LOG_TS(m_ts, "post 3: nms");

        detected_boxes.push_back(yolobox_vec);
    }

    return 0;
}

int YoloV5::post_process_cpu_opt(std::vector<sail::BMImage>& images, std::vector<YoloV5BoxVec>& detected_boxes) {
    YoloV5BoxVec yolobox_vec;
    std::vector<cv::Rect> bbox_vec;

    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
        yolobox_vec.clear();
        auto& frame = images[batch_idx];
        int frame_width = frame.width();
        int frame_height = frame.height();

        int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
        bool is_align_width = false;
        float ratio = get_aspect_scaled_ratio(frame.width(), frame.height(), m_net_w, m_net_h, &is_align_width);
        if (is_align_width) {
            ty1 = (int)((m_net_h - (int)(frame_height * ratio)) / 2);
        } else {
            tx1 = (int)((m_net_w - (int)(frame_width * ratio)) / 2);
        }
#endif

        int min_idx = 0;
        int box_num = 0;
        for (int i = 0; i < output_names.size(); i++) {
            assert(output_shape[i].size() == 3 || output_shape[i].size() == 5);
            if (output_shape[i].size() == 5) {
                box_num += output_shape[i][1] * output_shape[i][2] * output_shape[i][3];
            }

            if (min_dim > output_shape[i].size()) {
                min_idx = i;
                min_dim = output_shape[i].size();
            }
        }

        auto out_tensor = output_tensor[min_idx];
        int nout = out_tensor->shape()[min_dim - 1];
        m_class_num = nout - 5;
#if USE_MULTICLASS_NMS
        int out_nout = nout;
#else
        int out_nout = 7;
#endif
        float transformed_m_confThreshold = - std::log(1 / m_confThreshold - 1);

        float* output_data = nullptr;
        std::vector<float> decoded_data;

        if (min_dim == 3 && output_names.size() != 1) {
            std::cout << "--> WARNING: the current bmodel has redundant outputs" << std::endl;
            std::cout << "             you can remove the redundant outputs to improve performance" << std::endl;
            std::cout << std::endl;
        }

        if (min_dim == 5) {
            LOG_TS(m_ts, "post 1: get output and decode");
            const std::vector<std::vector<std::vector<int>>> anchors{
                {{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}};
            const int anchor_num = anchors[0].size();
            assert(output_names.size() == (int)anchors.size());
            assert(box_num > 0);
            if((int)decoded_data.size() != box_num*out_nout){
                decoded_data.resize(box_num*out_nout);
            }
            float* dst = decoded_data.data();
            for (int tidx = 0; tidx < output_names.size(); ++tidx) {
                int feat_c = output_tensor[tidx]->shape()[1];
                int feat_h = output_tensor[tidx]->shape()[2];
                int feat_w = output_tensor[tidx]->shape()[3];
                int area = feat_h * feat_w;
                assert(feat_c == anchor_num);
                int feature_size = feat_h * feat_w * nout;
                output_data = reinterpret_cast<float*>(output_tensor[tidx]->sys_data());
                float* tensor_data = output_data + batch_idx * feat_c * area * nout;
                for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
                    float* ptr = tensor_data + anchor_idx * feature_size;
                    for (int i = 0; i < area; i++) {
                        if(ptr[4] <= transformed_m_confThreshold){
                            ptr += nout;
                            continue;
                        }
                        dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
                        dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * m_net_h;
                        dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                        dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                        dst[4] = sigmoid(ptr[4]);
#if USE_MULTICLASS_NMS
                        for(int d = 5; d < nout; d++)
                            dst[d] = ptr[d];
#else
                        dst[5] = ptr[5];
                        dst[6] = 5;
                        for(int d = 6; d < nout; d++){
                            if(ptr[d] > dst[5]){
                                dst[5] = ptr[d];
                                dst[6] = d;
                            }
                        }
                        dst[6] -= 5;
#endif
                        dst += out_nout;
                        ptr += nout;
                    }
                }
            }
            output_data = decoded_data.data();
            box_num = (dst - output_data) / out_nout;
            LOG_TS(m_ts, "post 1: get output and decode");
        } else {
            LOG_TS(m_ts, "post 1: get output");
            assert(box_num == 0 || box_num == out_tensor->shape()[1]);
            box_num = out_tensor->shape()[1];
            output_data = reinterpret_cast<float*>(output_tensor[0]->sys_data()) + batch_idx * box_num * nout;
            LOG_TS(m_ts, "post 1: get output");
        }

        LOG_TS(m_ts, "post 2: filter boxes");
        int max_wh = 7680;
        bool agnostic = false;
        for (int i = 0; i < box_num; i++) {
            float* ptr = output_data + i * out_nout;
            float score = ptr[4];
            float box_transformed_m_confThreshold = - std::log(score / m_confThreshold - 1);
            if(min_dim != 5)
                box_transformed_m_confThreshold = m_confThreshold / score;
#if USE_MULTICLASS_NMS
            assert(min_dim == 5);
            float centerX = ptr[0];
            float centerY = ptr[1];
            float width = ptr[2];
            float height = ptr[3];
            for (int j = 0; j < m_class_num; j++) {
                float confidence = ptr[5 + j];
                int class_id = j;
                if (confidence > box_transformed_m_confThreshold)
                {
                    YoloV5Box box;
                    if (!agnostic)
                        box.x = centerX - width / 2 + class_id * max_wh;
                    else
                        box.x = centerX - width / 2;
                    if (box.x < 0) box.x = 0;
                    if (!agnostic)
                        box.y = centerY - height / 2 + class_id * max_wh;
                    else
                        box.y = centerY - height / 2;
                    if (box.y < 0) box.y = 0;
                    box.width = width;
                    box.height = height;
                    box.class_id = class_id;
                    box.score = sigmoid(confidence) * score;
                    yolobox_vec.push_back(box);
                }
            }
#else
            int class_id = ptr[6];
            float confidence = ptr[5];
            if(min_dim != 5){
                ptr = output_data + i * nout;
                score = ptr[4];
                class_id = argmax(&ptr[5], m_class_num);
                confidence = ptr[class_id + 5];
            }
            if (confidence > box_transformed_m_confThreshold) {
                float centerX = ptr[0];
                float centerY = ptr[1];
                float width = ptr[2];
                float height = ptr[3];

                YoloV5Box box;
                if (!agnostic)
                    box.x = centerX - width / 2 + class_id * max_wh;
                else
                    box.x = centerX - width / 2;
                if (box.x < 0) 
                    box.x = 0;
                if (!agnostic)
                    box.y = centerY - height / 2 + class_id * max_wh;
                else
                    box.y = centerY - height / 2;
                if (box.y < 0)
                    box.y = 0;
                box.width = width;
                box.height = height;
                box.class_id = class_id;
                if(min_dim == 5)
                    confidence = sigmoid(confidence);
                box.score = confidence * score;
                yolobox_vec.push_back(box);
            }
#endif
        }
        LOG_TS(m_ts, "post 2: filter boxes");

        // printf("\n --> valid boxes number = %d\n", (int)yolobox_vec.size());

        LOG_TS(m_ts, "post 3: nms");
        NMS(yolobox_vec, m_nmsThreshold);
        if (!agnostic)
            for (auto& box : yolobox_vec){
                box.x -= box.class_id * max_wh;
                box.y -= box.class_id * max_wh;
                box.x = (box.x - tx1) / ratio;
                if (box.x < 0) box.x = 0;
                box.y = (box.y - ty1) / ratio;
                if (box.y < 0) box.y = 0;
                box.width = (box.width) / ratio;
                if (box.x + box.width >= frame_width)
                    box.width = frame_width - box.x;
                box.height = (box.height) / ratio;
                if (box.y + box.height >= frame_height)
                    box.height = frame_height - box.y;
            }
        else
            for (auto& box : yolobox_vec){
                box.x = (box.x - tx1) / ratio;
                if (box.x < 0) box.x = 0;
                box.y = (box.y - ty1) / ratio;
                if (box.y < 0) box.y = 0;
                box.width = (box.width) / ratio;
                if (box.x + box.width >= frame_width)
                    box.width = frame_width - box.x;
                box.height = (box.height) / ratio;
                if (box.y + box.height >= frame_height)
                    box.height = frame_height - box.y;
            }
        LOG_TS(m_ts, "post 3: nms");

        detected_boxes.push_back(yolobox_vec);
    }

    return 0;
}

int YoloV5::argmax(float* data, int num) {
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

float YoloV5::sigmoid(float x) {
    return 1.0 / (1 + expf(-x));
}

void YoloV5::NMS(YoloV5BoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YoloV5Box& a, const YoloV5Box& b) { return a.score < b.score; });

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
            float overlap = std::max(0.0f, right - left + 0.00001f) * std::max(0.0f, bottom - top + 0.00001f);
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

void YoloV5::drawPred(int classId,
                      float conf,
                      int left,
                      int top,
                      int right,
                      int bottom,
                      cv::Mat& frame)  // Draw the predicted bounding box
{
    // Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

    // Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if ((int)m_class_names.size() >= m_class_num) {
        label = this->m_class_names[classId] + ":" + label;
    } else {
        label = std::to_string(classId) + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    // rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top +
    // baseLine), Scalar(0, 255, 0), FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
}

void YoloV5::draw_bmcv(int classId,
                       float conf,
                       int left,
                       int top,
                       int width,
                       int height,
                       sail::BMImage& frame,
                       bool put_text_flag)  // Draw the predicted bounding box
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
    int thickness = 2;
    if(width < thickness * 2 || height < thickness * 2){
        std::cout << "width or height too small, this rect will not be drawed: " << 
              "[" << start_x << ", "<< start_y << ", " << crop_w << ", " << crop_h << "]" << std::endl;
    } else{
        bmcv->rectangle(frame, start_x, start_y, crop_w, crop_h, color_tuple, thickness);
    }
    if (put_text_flag) {  // only support YUV420P, puttext not used here.
        std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
        if (BM_SUCCESS != bmcv->putText(frame, label.c_str(), left, top, color_tuple, 2, 2)) {
            std::cout << "bmcv put text error !!!" << std::endl;
        }
    }
}
