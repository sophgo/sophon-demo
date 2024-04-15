//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ppyoloe.hpp"
#include <fstream>
#include <vector>
#include <string>

#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

ppYoloe::ppYoloe(int dev_id, std::string bmodel_file) : engine() {
    engine = std::make_shared<sail::Engine>(dev_id);
    if (!engine->load(bmodel_file)) {
        std::cout << "Engine load bmodel " << bmodel_file << "failed" << std::endl;
        exit(0);
    }

    std::cout << "ppYoloe ctor .." << std::endl;
}

ppYoloe::~ppYoloe() {
    std::cout << "ppYoloe dtor ..." << std::endl;
}

int ppYoloe::Init(float confThresh, float nmsThresh, const std::string& coco_names_file) {
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
    if (input_names.size() > 2) {
        std::cout << "InputNumError, ppyoloe has only two inputs!" << std::endl;
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
    input_img_shape = engine->get_input_shape(graph_names[0], input_names[0]);
    std::string input_tensor_shape;
    for_each(input_img_shape.begin(), input_img_shape.end(), [&](int s) { input_tensor_shape += std::to_string(s) + " "; });
    std::cout << "input image shape -> " << input_tensor_shape << "\n";

    input_ratio_shape = engine->get_input_shape(graph_names[0], input_names[1]);
    input_tensor_shape.clear();
    for_each(input_ratio_shape.begin(), input_ratio_shape.end(), [&](int s) { input_tensor_shape += std::to_string(s) + " "; });
    std::cout << "input ratio shape -> " << input_tensor_shape << "\n";

    // output shapes of network 0
    output_score_shape = engine->get_output_shape(graph_names[0], output_names[0]);
    std::string output_tensor_shape;
    for_each(output_score_shape.begin(), output_score_shape.end(), [&](int s) { output_tensor_shape += std::to_string(s) + " "; });
    std::cout << "output score shape -> " << output_tensor_shape << "\n";

    output_coordinate_shape = engine->get_output_shape(graph_names[0], output_names[1]);
    output_tensor_shape.clear();
    for_each(output_coordinate_shape.begin(), output_coordinate_shape.end(), [&](int s) { output_tensor_shape += std::to_string(s) + " "; });
    std::cout << "output coordinate shape -> " << output_tensor_shape << "\n";

    // data type of network input.
    input_img_dtype = engine->get_input_dtype(graph_names[0], input_names[0]);
    std::cout << "input image dtype -> " << input_img_dtype << ", is fp32=" << ((input_img_dtype == BM_FLOAT32) ? "true" : "false")
              << "\n";

    input_ratio_dtype = engine->get_input_dtype(graph_names[0], input_names[1]);
    std::cout << "input ratio dtype -> " << input_ratio_dtype << ", is fp32=" << ((input_ratio_dtype == BM_FLOAT32) ? "true" : "false")
              << "\n";

    // data type of network output.
    output_score_dtype = engine->get_output_dtype(graph_names[0], output_names[0]);
    std::cout << "output score dtype -> " << output_score_dtype << ", is fp32=" << ((output_score_dtype == BM_FLOAT32) ? "true" : "false")
              << "\n";
    output_coordinate_dtype = engine->get_output_dtype(graph_names[0], output_names[1]);
    std::cout << "output coordinate dtype -> " << output_coordinate_dtype << ", is fp32=" << ((output_coordinate_dtype == BM_FLOAT32) ? "true" : "false")
              << "\n";
    std::cout << "===============================" << std::endl;

    // 3. Initialize Network IO
    input_img_tensor = std::make_shared<sail::Tensor>(handle, input_img_shape, input_img_dtype, false, false);
    input_ratio_tensor = std::make_shared<sail::Tensor>(handle, input_ratio_shape, input_ratio_dtype, true, true);
    input_tensors[input_names[0]] = input_img_tensor.get();
    input_tensors[input_names[1]] = input_ratio_tensor.get();

    output_score_tensor = std::make_shared<sail::Tensor>(handle, output_score_shape, output_score_dtype, true, true);
    output_coordinate_tensor = std::make_shared<sail::Tensor>(handle, output_coordinate_shape, output_coordinate_dtype, true, true);
    output_tensors[output_names[0]] = output_score_tensor.get();
    output_tensors[output_names[1]] = output_coordinate_tensor.get();

    engine->set_io_mode(graph_names[0], sail::SYSO);

    // Initialize net utils
    max_batch = input_img_shape[0];
    m_net_h = input_img_shape[2];
    m_net_w = input_img_shape[3];
    float input_scale = engine->get_input_scale(graph_names[0], input_names[0]);
    input_scale = input_scale * 1.0 / 255.f;
    norm[0] = input_scale;
    norm[1] = 0;
    norm[2] = input_scale;
    norm[3] = 0;
    norm[4] = input_scale;
    norm[5] = 0;

    ab[0] = 1 / 0.229;
    ab[1] = -0.485 / 0.229;
    ab[2] = 1 / 0.224;
    ab[3] = -0.456 / 0.224;
    ab[4] = 1 / 0.225;
    ab[5] = -0.406 / 0.225;
    return 0;
}

void ppYoloe::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

int ppYoloe::batch_size() {
    return max_batch;
};

int ppYoloe::Detect(std::vector<sail::BMImage>& input_images, std::vector<ppYoloeBoxVec>& boxes) {
    int ret = 0;
    // 1. preprocess
    m_ts->save("ppyoloe preprocess", max_batch);
    if (input_images.size() == 4 && max_batch == 4) {
        ret = pre_process<4>(input_images);
    } else if (input_images.size() == 1 && max_batch == 1) {
        ret = pre_process(input_images[0]);
    } else {
        std::cout << "unsupport batch size!" << std::endl;
        exit(1);
    }
    CV_Assert(ret == 0);
    m_ts->save("ppyoloe preprocess", max_batch);
    auto bmimg = bmcv->tensor_to_bm_image(*input_tensors[input_names[0]]);
    // 2. forward
    m_ts->save("ppyoloe inference", max_batch);
    engine->process(graph_names[0], input_tensors, output_tensors);
    m_ts->save("ppyoloe inference", max_batch);

    // 3. post process
    m_ts->save("ppyoloe postprocess", max_batch);
    ret = post_process(input_images, boxes);
    CV_Assert(ret == 0);
    m_ts->save("ppyoloe postprocess", max_batch);
    return ret;
}

int ppYoloe::pre_process(sail::BMImage& input) {
    std::vector<float> ratio = get_img_ratio(input.width(), input.height(), input_img_shape[3], input_img_shape[2]);

    float *input_ratio_tensor_ptr = reinterpret_cast<float*>(input_ratio_tensor->sys_data());

    input_ratio_tensor_ptr[0]=ratio[0];
    input_ratio_tensor_ptr[1]=ratio[1];

    input_ratio_tensor->sync_s2d();


    sail::BMImage rgb_img(engine->get_handle(), input.height(), input.width(), FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
    bmcv->convert_format(input, rgb_img);

    sail::BMImage min_max_img(engine->get_handle(), input_img_shape[2], input_img_shape[3], FORMAT_RGB_PLANAR,
                              bmcv->get_bm_image_data_format(input_img_dtype));

    sail::BMImage convert_img(engine->get_handle(), input_img_shape[2], input_img_shape[3], FORMAT_RGB_PLANAR,
                              bmcv->get_bm_image_data_format(input_img_dtype));

    sail::BMImage resized_img = bmcv->resize(rgb_img, m_net_w, m_net_h, BMCV_INTER_LINEAR);

    // convert
    bmcv->convert_to(
        resized_img, min_max_img,
        std::make_tuple(std::make_pair(norm[0], norm[1]), std::make_pair(norm[2], norm[3]), std::make_pair(norm[4], norm[5])));

    bmcv->convert_to(
        min_max_img, convert_img,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));

    bmcv->bm_image_to_tensor(convert_img, *input_img_tensor.get());

    return 0;
}

template <std::size_t N>
int ppYoloe::pre_process(std::vector<sail::BMImage>& input) {
    if (input.size() != N) {
        std::cout << "Unsupport batch size!" << std::endl;
        exit(1);
    }
    std::shared_ptr<sail::BMImage> resized_imgs_vec[N];
    sail::BMImageArray<N> resized_imgs;
    sail::BMImageArray<N> min_max_imgs(engine->get_handle(), input_img_shape[2], input_img_shape[3], FORMAT_RGB_PLANAR,
                                       bmcv->get_bm_image_data_format(input_img_dtype));
    sail::BMImageArray<N> standard_imgs(engine->get_handle(), input_img_shape[2], input_img_shape[3], FORMAT_RGB_PLANAR,
                                       bmcv->get_bm_image_data_format(input_img_dtype));
    float *input_ratio_tensor_ptr = reinterpret_cast<float*>(input_ratio_tensor->sys_data());
    for (size_t i = 0; i < input.size(); ++i) {

        std::vector<float> ratio = get_img_ratio(input[i].width(), input[i].height(), input_img_shape[3], input_img_shape[2]);
        input_ratio_tensor_ptr[2*i]=ratio[0];
        input_ratio_tensor_ptr[2*i+1]=ratio[1];

        sail::BMImage rgb_img(engine->get_handle(), input[i].height(), input[i].width(), FORMAT_RGB_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE);
        bmcv->convert_format(input[i], rgb_img);

        resized_imgs_vec[i] = std::make_shared<sail::BMImage>(engine->get_handle(), input_img_shape[2], input_img_shape[3],
                                                              FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
        bmcv->resize(rgb_img, *resized_imgs_vec[i],m_net_w, m_net_h, BMCV_INTER_LINEAR);
        resized_imgs.attach_from(i, *resized_imgs_vec[i].get());
    }
    input_ratio_tensor->sync_s2d();
    bmcv->convert_to(
        resized_imgs, min_max_imgs,
        std::make_tuple(std::make_pair(norm[0], norm[1]), std::make_pair(norm[2], norm[3]), std::make_pair(norm[4], norm[5])));

    bmcv->convert_to(
        min_max_imgs, standard_imgs,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(standard_imgs, *input_img_tensor.get());
    return 0;
}

std::vector<float> ppYoloe::get_img_ratio(int src_w, int src_h, int dst_w, int dst_h){
  float r_h = (float)dst_h / src_h;
  float r_w = (float)dst_w / src_w;
  return std::vector<float>{r_h, r_w};
}
int ppYoloe::post_process(std::vector<sail::BMImage>& images, std::vector<ppYoloeBoxVec>& detected_boxes) {
    // postprocess
    ppYoloeBoxVec yolobox_vec;
    for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx){
        LOG_TS(m_ts, "post 1: get outputs");
        yolobox_vec.clear();

        auto out_tensor0 = output_score_tensor;
        // int nout = out_tensor0->shape()[1];
        m_class_num = out_tensor0->shape()[1];
        int box_num = out_tensor0->shape()[2];

        float* output_score_data = nullptr;
        float* output_coordinate_data = nullptr;
        output_score_data = reinterpret_cast<float*>(output_score_tensor->sys_data()) + batch_idx * box_num * m_class_num;
        output_coordinate_data = reinterpret_cast<float*>(output_coordinate_tensor->sys_data()) + batch_idx * box_num * 4;

        LOG_TS(m_ts, "post 2: filter boxes");
        for (int i = 0; i < box_num; i++) {
            float* ptr0 = output_score_data + i;
            float *ptr1 = output_coordinate_data + i*4;

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


// Draw the predicted bounding box
void ppYoloe::draw_bmcv(int classId,float conf,int left,int top,int width,int height,sail::BMImage& frame,bool put_text_flag)
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
