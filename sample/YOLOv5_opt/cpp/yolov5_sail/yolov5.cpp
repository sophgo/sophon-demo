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

int YoloV5::Init(float confThresh, float nmsThresh, const std::string& tpu_kernel_module_path, const std::string& coco_names_file) {
    m_confThreshold = MAX(0.1, confThresh);
    m_nmsThreshold = MAX(0.1, nmsThresh);
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
    engine->set_io_mode(graph_names[0], sail::DEVIO);

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

    // Initialize tpukernel postprocession.
    tpukernel_api = std::make_shared<sail::tpu_kernel_api_yolov5_detect_out>(
                    engine->get_device_id(), 
                    output_shape, m_net_w, m_net_h, 
                    tpu_kernel_module_path);
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
    ret = post_process_tpu_kernel(input_images, boxes);
    CV_Assert(ret == 0);
    LOG_TS(m_ts, "yolov5 postprocess");
    return ret;
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
                                              &padding, &rect);
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

int YoloV5::post_process_tpu_kernel(std::vector<sail::BMImage>& images, std::vector<YoloV5BoxVec>& detected_boxes) {
  std::vector<sail::TensorPTRWithName> tpukernel_input;
  std::vector<std::vector<int>> shapes; 
  int output_num = output_names.size();
  sail::TensorPTRWithName* tensor_map[output_num]; 

  for(int i = 0; i < output_num; i++) {
    tensor_map[i] = new sail::TensorPTRWithName(std::to_string(i), output_tensor[i].get());
    tpukernel_input.emplace_back(*tensor_map[i]);
  }
  std::vector<std::vector<sail::DeteObjRect>> out_doxs;  
  tpukernel_api->process(tpukernel_input, out_doxs, m_confThreshold, m_nmsThreshold);
  for(int i = 0; i < max_batch; i++){
    int tx1 = 0, ty1 = 0;
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(images[i].width(), images[i].height(), m_net_w, m_net_h, &isAlignWidth);
    if (isAlignWidth) {
      ty1 = (int)((m_net_h - (int)(images[i].height() * ratio)) / 2);
    } else {
      tx1 = (int)((m_net_w - (int)(images[i].width() * ratio)) / 2);
  }
#endif

    YoloV5BoxVec vec;
    detected_boxes.push_back(vec);
    for (int bid = 0; bid < out_doxs[i].size(); bid++) {
      YoloV5Box temp_bbox;
      temp_bbox.class_id = out_doxs[i][bid].class_id;
      if (temp_bbox.class_id == -1) {
        continue;
      }
      temp_bbox.score = out_doxs[i][bid].score;
      temp_bbox.width = (out_doxs[i][bid].width+ 0.5) / ratio;
      temp_bbox.height =(out_doxs[i][bid].height+ 0.5) / ratio;
      float centerX = ((out_doxs[i][bid].left + out_doxs[i][bid].right) / 2 + 1 - tx1) / ratio - 1;
      float centerY = ((out_doxs[i][bid].top + out_doxs[i][bid].bottom) / 2 + 1 - ty1) / ratio - 1;
      temp_bbox.x = MAX(int(centerX - temp_bbox.width / 2), 0);
      temp_bbox.y = MAX(int(centerY - temp_bbox.height / 2), 0);
      detected_boxes[i].push_back(temp_bbox);  // 0
    }   
  }
  for(int i = 0; i < output_num; i++) {
    delete tensor_map[i];
  }
  return 0;
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
    bmcv->rectangle(frame, start_x, start_y, crop_w, crop_h, color_tuple, 3);
    if (put_text_flag) {  // only support YUV420P, puttext not used here.
        std::string label = m_class_names[classId] + ":" + cv::format("%.2f", conf);
        if (BM_SUCCESS != bmcv->putText(frame, label.c_str(), left, top, color_tuple, 2, 2)) {
            std::cout << "bmcv put text error !!!" << std::endl;
        }
    }
}
