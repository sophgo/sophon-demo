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
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},     {255, 85, 0},    {255, 170, 0},   {255, 255, 0}, {170, 255, 0}, {85, 255, 0},  {0, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 255, 255},   {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},
    {170, 0, 255},   {255, 0, 255},   {255, 0, 170},   {255, 0, 85},  {255, 0, 0},   {255, 0, 255}, {255, 85, 255},
    {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}};

YoloV5::YoloV5(int dev_id, std::string bmodel_file) : engine() {
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

int YoloV5::Init(const std::string& coco_names_file) {

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
    m_net_h = input_shape[1];
    m_net_w = input_shape[2];
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
    m_ts->save("yolov5 preprocess", input_images.size());
    if (input_images.size() == 4 && max_batch == 4) {
        ret = pre_process<4>(input_images);
    } else if (input_images.size() == 1 && max_batch == 1) {
        ret = pre_process(input_images[0]);
    } else {
        std::cout << "unsupport batch size!" << std::endl;
        exit(1);
    }
    CV_Assert(ret == 0);
    m_ts->save("yolov5 preprocess", input_images.size());

    // output_tensor[0]->zeros();

    // 2. forward
    m_ts->save("yolov5 inference", input_images.size());
    engine->process(graph_names[0], input_tensors, output_tensors);
    m_ts->save("yolov5 inference", input_images.size());

    // 3. post process
    m_ts->save("yolov5 postprocess", input_images.size());
    ret = get_result(input_images, boxes);
    CV_Assert(ret == 0);
    m_ts->save("yolov5 postprocess", input_images.size());
    return ret;
}

int YoloV5::pre_process(sail::BMImage& input) {
    int ret = 0;
    sail::BMImage resized_img(engine->get_handle(), input.height(), input.width(), FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE);
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

    ret = bmcv->vpp_crop_and_resize_padding(input, resized_img, 0, 0, 
            input.width(), input.height(), m_net_w, m_net_h, pad, RESIZE_STRATEGY);
    CV_Assert(ret == 0);

#else
    ret = bmcv->crop_and_resize(input, resized_img, 0, 0, input.width(), input.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
    CV_Assert(ret == 0);
#endif

    bmcv->bm_image_to_tensor(resized_img, *input_tensor.get());

    return 0;
}

template <std::size_t N>
int YoloV5::pre_process(std::vector<sail::BMImage>& input) {
    int ret = 0;
    if (input.size() != N) {
        std::cout << "Unsupport batch size!" << std::endl;
        exit(1);
    }
    sail::BMImageArray<N> resized_imgs = sail::BMImageArray<N>(engine->get_handle(), m_net_h, m_net_w, FORMAT_BGR_PACKED,
                              DATA_TYPE_EXT_1N_BYTE);

    
    for (size_t i = 0; i < input.size(); ++i) {
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


        ret = bmcv->vpp_crop_and_resize_padding(&input[i].data(), &resized_imgs[i], 0, 0, input[i].width(),
                                          input[i].height(), m_net_w, m_net_h, pad, 1, RESIZE_STRATEGY);
        assert(ret == 0);

#else
        sail::BMImage resized_img(engine->get_handle(), m_net_h, m_net_w, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE);
        bmcv->crop_and_resize(input[i], resized_img, 0, 0, input[i].width(), input[i].height(), m_net_w, m_net_h, RESIZE_STRATEGY);
        resized_imgs.copy_from(i, resized_img);
#endif
        
    }

    bmcv->bm_image_to_tensor(resized_imgs, *input_tensor.get());
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

int YoloV5::get_result(const std::vector<sail::BMImage>& images, std::vector<YoloV5BoxVec>& detected_boxes) {
  YoloV5BoxVec yolobox_vec;
  auto shape = engine->get_output_shape(graph_names[0], output_names[0]);
  int box_num = shape[2];
//   printf("box_num %d \n",box_num);
  int box_id = 0;

  auto output_data = reinterpret_cast<float*>(output_tensor[0]->sys_data());
  auto cout = 0;

  for(int batch_idx = 0; batch_idx < images.size(); ++ batch_idx){
    yolobox_vec.clear();
    auto& frame = images[batch_idx];
    int frame_width = frame.width();
    int frame_height = frame.height();

    int tx1 = 0, ty1 = 0;
    float rx = float(frame.width()) / m_net_w;
    float ry = float(frame.height()) / m_net_h;
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(frame.width(), frame.height(), m_net_w, m_net_h, &isAlignWidth);
    if (isAlignWidth) {
      ty1 = (int)((m_net_h - (int)(frame_height*ratio)) / 2);
    }else{
      tx1 = (int)((m_net_w - (int)(frame_width*ratio)) / 2);
    }
    rx = 1.0 / ratio;
    ry = 1.0 / ratio;
#endif
    while(*(output_data+cout) == float(batch_idx) && box_id < box_num){
      YoloV5Box box;
      // init box data
      float centerX = *(output_data+cout+3);
      float centerY = *(output_data+cout+4);
      float width = *(output_data+cout+5);
      float height = *(output_data+cout+6);
      // get bbox
      box.x = int((centerX - width / 2 - tx1) * rx);
      if (box.x < 0) box.x = 0;
      box.y = int((centerY - height / 2  - ty1) * ry);
      if (box.y < 0) box.y = 0;
      box.width = width * rx;
      box.height = height * ry;

      box.class_id = int(*(output_data+cout+1));
      box.score = *(output_data+cout+2);
      cout += 7;
    
      box_id++;
      yolobox_vec.emplace_back(box);
    }
    detected_boxes.emplace_back(yolobox_vec);
  }
  return 0;
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
    int crop_w = MAX(MIN(width, frame.width() - start_x), 0);
    int crop_h = MAX(MIN(height, frame.height() - start_y), 0);
    auto color_tuple = std::make_tuple(colors[classId % colors_num][2], colors[classId % colors_num][1],
                                       colors[classId % colors_num][0]);
    int thickness = 2;
    if(crop_w <= thickness * 2 || crop_h <= thickness * 2){
        std::cout << "width or height too small, this rect will not be drawed: " << 
              "[" << start_x << ", "<< start_y << ", " << crop_w << ", " << crop_h << "]" << std::endl;
    } else{
        bmcv->rectangle(frame, start_x, start_y, crop_w, crop_h, color_tuple, thickness);
    }
    if (put_text_flag) {  // only support YUV420P, puttext not used here.
        std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
        if (BM_SUCCESS != bmcv->putText(frame, label.c_str(), start_x, start_y, color_tuple, 2, 2)) {
            std::cout << "bmcv put text error !!!" << std::endl;
        }
    }
}
