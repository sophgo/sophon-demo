//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "scrfd.hpp"

#include <fstream>
#include <string>
#define USE_ASPECT_RATIO 1
#define RESIZE_STRATEGY BMCV_INTER_NEAREST
#define USE_BMCV_VPP_CONVERT 1
#define DUMP_FILE 0

Scrfd::Scrfd(int dev_id, std::string bmodel_file) : engine() {
  engine = std::make_shared<sail::Engine>(dev_id);
  if (!engine->load(bmodel_file)) {
    std::cout << "Engine load bmodel " << bmodel_file << "failed" << std::endl;
    exit(0);
  }

  std::cout << "Scrfd ctor .." << std::endl;
}

Scrfd::~Scrfd() { std::cout << "Scrfd ctor .." << std::endl; }

int Scrfd::Init(float confThresh, float nmsThresh) {
  m_confThreshold = confThresh;
  m_nmsThreshold = nmsThresh;

  std::cout << "===============================" << std::endl;

  // 1. Initialize bmcv
  sail::Handle handle(engine->get_device_id());
  bmcv = std::make_shared<sail::Bmcv>(handle);

  // 2. Initialize engine
  graph_names = engine->get_graph_names();
  std::string gh_info;
  for_each(graph_names.begin(), graph_names.end(),
           [&](std::string& s) { gh_info += "0: " + s + "; "; });
  std::cout << "grapgh name -> " << gh_info << "\n";
  if (graph_names.size() > 1) {
    std::cout << "NetworkNumError, this net only accept one network!"
              << std::endl;
    exit(1);
  }

  // input names of network
  input_names = engine->get_input_names(graph_names[0]);
  assert(input_names.size() > 0);
  std::string input_tensor_names;
  for_each(input_names.begin(), input_names.end(),
           [&](std::string& s) { input_tensor_names += "0: " + s + "; "; });
  std::cout << "net input name -> " << input_tensor_names << "\n";
  if (input_names.size() > 1) {
    std::cout << "InputNumError, Scrfd has only one inputs!" << std::endl;
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
  for_each(input_shape.begin(), input_shape.end(),
           [&](int s) { input_tensor_shape += std::to_string(s) + " "; });
  std::cout << "input tensor shape -> " << input_tensor_shape << "\n";

  // output shapes of network 0
  output_shape.resize(output_names.size());
  for (int i = 0; i < output_names.size(); i++) {
    output_shape[i] = engine->get_output_shape(graph_names[0], output_names[i]);
    std::string output_tensor_shape;
    for_each(output_shape[i].begin(), output_shape[i].end(),
             [&](int s) { output_tensor_shape += std::to_string(s) + " "; });
    std::cout << "output tensor " << i << " shape -> " << output_tensor_shape
              << "\n";
  }

  // data type of network input.
  input_dtype = engine->get_input_dtype(graph_names[0], input_names[0]);
  std::cout << "input dtype -> " << input_dtype
            << ", is fp32=" << ((input_dtype == BM_FLOAT32) ? "true" : "false")
            << "\n";

  // data type of network output.
  output_dtype = engine->get_output_dtype(graph_names[0], output_names[0]);
  std::cout << "output dtype -> " << output_dtype
            << ", is fp32=" << ((output_dtype == BM_FLOAT32) ? "true" : "false")
            << "\n";
  std::cout << "===============================" << std::endl;

  // 3. Initialize Network IO
  input_tensor = std::make_shared<sail::Tensor>(handle, input_shape,
                                                input_dtype, false, false);
  input_tensors[input_names[0]] = input_tensor.get();
  output_tensor.resize(output_names.size());
  for (int i = 0; i < output_names.size(); i++) {
    output_tensor[i] = std::make_shared<sail::Tensor>(handle, output_shape[i],
                                                      output_dtype, true, true);
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
  ab[0] = 1.0 * input_scale;    // scale
  ab[1] = 127.5 * input_scale;  //  mean
  ab[2] = 1.0 * input_scale;
  ab[3] = 127.5 * input_scale;
  ab[4] = 1.0 * input_scale;
  ab[5] = 127.5 * input_scale;
  return 0;
}

void Scrfd::enableProfile(TimeStamp* ts) { m_ts = ts; }

int Scrfd::batch_size() { return max_batch; };

int Scrfd::Detect(std::vector<sail::BMImage>& input_images,
                  std::vector<ScrfdBoxVec>& boxes) {
  int ret = 0;
  // 1. preprocess
  m_ts->save("scrfd preprocess", input_images.size());
  if (input_images.size() == 4 && max_batch == 4) {
    ret = pre_process<4>(input_images);
  } else if (input_images.size() == 1 && max_batch == 1) {
    ret = pre_process(input_images[0]);
  } else {
    std::cout << "unsupport batch size!" << std::endl;
    exit(1);
  }
  CV_Assert(ret == 0);
  m_ts->save("scrfd preprocess", input_images.size());
  auto bmimg = bmcv->tensor_to_bm_image(*input_tensors[input_names[0]]);
  
  // 2. forward
  m_ts->save("scrfd inference", input_images.size());
  engine->process(graph_names[0], input_tensors, output_tensors);
  m_ts->save("scrfd inference", input_images.size());

  // 3. post process
  m_ts->save("scrfd postprocess", input_images.size());
  ret = post_process(input_images, boxes);
  CV_Assert(ret == 0);
  m_ts->save("scrfd postprocess", input_images.size());
  return ret;
}

int Scrfd::pre_process(sail::BMImage& input) {
  int ret = 0;
    sail::BMImage rgb_img(engine->get_handle(), input.height(), input.width(), FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
    rgb_img.align();
    bmcv->convert_format(input, rgb_img);
    sail::BMImage convert_img(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                              bmcv->get_bm_image_data_format(input_dtype));
    sail::BMImage resized_img(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                               DATA_TYPE_EXT_1N_BYTE);
    resized_img.align();
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

    ret = bmcv->vpp_crop_and_resize_padding(rgb_img, resized_img, 0, 0, 
            rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, pad, RESIZE_STRATEGY);
    CV_Assert(ret == 0);

#else
    ret = bmcv->crop_and_resize(rgb_img, resized_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
    CV_Assert(ret == 0);
#endif
    bmcv->convert_to(
        resized_img, convert_img,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(convert_img, *input_tensor.get());
    return 0;
}

template <std::size_t N>
int Scrfd::pre_process(std::vector<sail::BMImage>& input) {
  int ret = 0;
    if (input.size() != N) {
        std::cout << "Unsupport batch size!" << std::endl;
        exit(1);
    }
    std::shared_ptr<sail::BMImage> resized_imgs_vec[N];
    sail::BMImageArray<N> resized_imgs;
    sail::BMImageArray<N> convert_imgs(engine->get_handle(), input_shape[2], input_shape[3], FORMAT_RGB_PLANAR,
                                       bmcv->get_bm_image_data_format(input_dtype));
    
    for (size_t i = 0; i < input.size(); ++i) {
        sail::BMImage rgb_img(engine->get_handle(), input[i].height(), input[i].width(), FORMAT_RGB_PLANAR,
                              DATA_TYPE_EXT_1N_BYTE);
        rgb_img.align();
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
        resized_imgs_vec[i]->align();

        ret = bmcv->vpp_crop_and_resize_padding(&rgb_img.data(), &resized_imgs_vec[i].get()->data(), 0, 0, rgb_img.width(),
                                          rgb_img.height(), m_net_w, m_net_h, pad, 1, RESIZE_STRATEGY);
        assert(ret == 0);

        resized_imgs.attach_from(i, *resized_imgs_vec[i].get());
#else
        sail::BMImage resized_img =
            bmcv->crop_and_resize(rgb_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
        resized_imgs.copy_from(i, resized_img);
#endif
    }
    bmcv->convert_to(
        resized_imgs, convert_imgs,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(convert_imgs, *input_tensor.get());
    return 0;
}

float Scrfd::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h,
                                     bool* pIsAligWidth) {
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

int Scrfd::post_process(std::vector<sail::BMImage>& images,
                        std::vector<ScrfdBoxVec>& detected_boxes) {
  ScrfdBoxVec scrfdbox_vec;
  for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
    scrfdbox_vec.clear();
    auto& frame = images[batch_idx];
    int frame_width = frame.width();
    int frame_height = frame.height();
    compute_pad_resize_param(frame_height, frame_width, m_net_h, m_net_w, rescale_params);
    std::vector<cvai_face_info_t> res =
        outputParser(m_net_w, m_net_h, rescale_params, batch_idx);
    scrfdbox_vec = res;
    detected_boxes.push_back(scrfdbox_vec);
  };
  return 0;
};

void Scrfd::draw_opencv(std::vector<cvai_face_info_t>& res, cv::Mat& frame) {
  for (uint32_t i = 0; i < res.size(); i++) {
    cvai_bbox_t bbox = res[i].bbox;
    cvai_pts_t pti = res[i].pts;
    int x1 = (int)bbox.x1;
    int y1 = (int)bbox.y1;
    int x2 = (int)bbox.x2;
    int y2 = (int)bbox.y2;
    cv::Rect box(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
    cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
    for (uint32_t j = 0; j < pti.size; j++) {
      int x = (int)pti.x[j];
      int y = (int)pti.y[j];
      cv::circle(frame, cv::Point(pti.x[j], pti.y[j]), 2, cv::Scalar(0, 0, 255),
                 2);
    }
    string label = cv::format("%.3f", bbox.score);
    cv::putText(frame, label, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(0, 255, 0), 1);
  }
};

void Scrfd::draw_bmcv(cvai_pts_t five_point, float conf, int left, int top,
                      int width, int height, sail::BMImage& frame,
                      bool put_text_flag, bool draw_point_flag) {
  if (conf < 0.25) return;
  int start_x_top = MIN(MAX(left, 0), frame.width());
  int start_y_top = MIN(MAX(top, 0), frame.height());
  int crop_w = MAX(MIN(width, frame.width() - left), 0);
  int crop_h = MAX(MIN(height, frame.height() - top), 0);
  auto color_tuple = std::make_tuple(0, 0, 255);
  int thickness = 2;
  if (draw_point_flag) {
    cvai_pts_t pti = five_point;
    for (int j = 0; j < pti.size; j++) {
      int x = (int)pti.x[j];
      int y = (int)pti.y[j];
      auto center = std::make_pair(x, y);
      bmcv->drawPoint(frame, center, color_tuple, 2);
    }
  }
  if (width < thickness * 2 || height < thickness * 2) {
    std::cout << "width or height too small, this rect will not be drawed: "
              << "[" << start_x_top << ", " << start_y_top << ", " << crop_w
              << ", " << crop_h << "]" << std::endl;
  } else {
    bmcv->rectangle(frame, start_x_top, start_y_top, crop_w, crop_h,
                    color_tuple, thickness);
  }
  if (put_text_flag) {  // only support YUV420P, puttext not used here.
    std::string label = cv::format("%.3f", conf);
    if (BM_SUCCESS !=
        bmcv->putText(frame, label.c_str(), left, top, color_tuple, 2, 2)) {
      std::cout << "bmcv put text error !!!" << std::endl;
    }
  }
};

std::vector<std::vector<float>> Scrfd::generate_mmdet_base_anchors(
    float base_size, float center_offset, const std::vector<float>& ratios,
    const std::vector<int>& scales) {
  std::vector<std::vector<float>> base_anchors;
  float x_center = base_size * center_offset;
  float y_center = base_size * center_offset;

  for (size_t i = 0; i < ratios.size(); i++) {
    float h_ratio = sqrt(ratios[i]);
    float w_ratio = 1 / h_ratio;
    for (size_t j = 0; j < scales.size(); j++) {
      float halfw = base_size * w_ratio * scales[j] / 2;
      float halfh = base_size * h_ratio * scales[j] / 2;
      // x1,y1,x2,y2
      std::vector<float> base_anchor = {x_center - halfw, y_center - halfh,
                                        x_center + halfw, y_center + halfh};
      base_anchors.emplace_back(base_anchor);
    }
  }
  return base_anchors;
}

std::vector<std::vector<float>> Scrfd::generate_mmdet_grid_anchors(
    int feat_w, int feat_h, int stride,
    std::vector<std::vector<float>>& base_anchors) {
  std::vector<std::vector<float>> grid_anchors;
  for (int ih = 0; ih < feat_h; ih++) {
    int sh = ih * stride;
    for (int iw = 0; iw < feat_w; iw++) {
      int sw = iw * stride;
      for (size_t k = 0; k < base_anchors.size(); k++) {
        auto& base_anchor = base_anchors[k];
        std::vector<float> grid_anchor = {
            base_anchor[0] + sw, base_anchor[1] + sh, base_anchor[2] + sw,
            base_anchor[3] + sh};
        grid_anchors.emplace_back(grid_anchor);
      }
    }
  }
  return grid_anchors;
}

template <typename T>
void Scrfd::NonMaximumSuppression(std::vector<T>& bboxes,
                                  std::vector<T>& bboxes_nms,
                                  const float threshold, const char method) {
  std::sort(bboxes.begin(), bboxes.end(),
            [](cvai_face_info_t& a, cvai_face_info_t& b) {
              return a.bbox.score > b.bbox.score;
            });

  int select_idx = 0;
  int num_bbox = bboxes.size();
  std::vector<int> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    bboxes_nms.emplace_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;

    cvai_bbox_t select_bbox = bboxes[select_idx].bbox;
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
                                     (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      cvai_bbox_t& bbox_i(bboxes[i].bbox);
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0) {
        continue;
      }

      float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) *
                                       (bbox_i.y2 - bbox_i.y1 + 1));
      float area_intersect = w * h;
      if (method == 'u' && static_cast<float>(area_intersect) /
                                   (area1 + area2 - area_intersect) >
                               threshold) {
        mask_merged[i] = 1;
        continue;
      }
      if (method == 'm' &&
          static_cast<float>(area_intersect) / std::min(area1, area2) >
              threshold) {
        mask_merged[i] = 1;
      }
    }
  }
}

void Scrfd::clip_boxes(int width, int height, cvai_bbox_t& box) {
  if (box.x1 < 0) {
    box.x1 = 0;
  }
  if (box.y1 < 0) {
    box.y1 = 0;
  }
  if (box.x2 > width - 1) {
    box.x2 = width - 1;
  }
  if (box.y2 > height - 1) {
    box.y2 = height - 1;
  }
}

bool Scrfd::compute_pad_resize_param(int src_height, int src_width,
                                     int dst_height, int dst_width,
                                     std::vector<float>& rescale_params) {
  rescale_params.clear();
  float src_w = src_width;
  float src_h = src_height;
  float ratio_w = src_w / dst_width;
  float ratio_h = src_h / dst_height;
  float ratio = std::max(ratio_w, ratio_h);
  rescale_params.push_back(ratio);
  rescale_params.push_back(ratio);
  if (ratio_w != ratio_h) {
    int src_resized_w = lrint(src_w / ratio);
    int src_resized_h = lrint(src_h / ratio);
    int roi_x = (dst_width - src_resized_w + 1) / 2;
    int roi_y = (dst_height - src_resized_h + 1) / 2;
    rescale_params.push_back(roi_x);
    rescale_params.push_back(roi_y);
    return true;
  } else {
    rescale_params.push_back(0.0f);
    rescale_params.push_back(0.0f);
    return false;
  }
}

std::vector<cvai_face_info_t> Scrfd::outputParser(
    int frame_width, int frame_height, const std::vector<float>& rescale_param,
    int batch_idx) {
  std::vector<anchor_cfg> cfg;
  anchor_cfg tmp;

  std::vector<int> m_feat_stride_fpn = {8, 16, 32};

  tmp.SCALES = {1, 2};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 8;
  cfg.push_back(tmp);

  tmp.SCALES = {4, 8};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 16;
  cfg.push_back(tmp);

  tmp.SCALES = {16, 32};
  tmp.BASE_SIZE = 16;
  tmp.RATIOS = {1.0};
  tmp.ALLOWED_BORDER = 9999;
  tmp.STRIDE = 32;
  cfg.push_back(tmp);

  float im_scale_w = rescale_param[0];
  float im_scale_h = rescale_param[1];
  float pad_x = rescale_param[2];
  float pad_y = rescale_param[3];

  std::map<std::string, std::vector<anchor_box>> anchors_fpn_map;
  std::map<int, std::vector<std::vector<float>>> fpn_anchors_;

  for (size_t i = 0; i < cfg.size(); i++) {
    std::vector<std::vector<float>> base_anchors = generate_mmdet_base_anchors(
        cfg[i].BASE_SIZE, 0, cfg[i].RATIOS, cfg[i].SCALES);
    int stride = cfg[i].STRIDE;
    int input_w = frame_width;
    int input_h = frame_height;
    int feat_w = int(input_w / float(stride) + 0.5);
    int feat_h = int(input_h / float(stride) + 0.5);
    fpn_anchors_[stride] =
        generate_mmdet_grid_anchors(feat_w, feat_h, stride, base_anchors);
  }

  m_feat_stride_fpn = {8, 16, 32};
  std::vector<cvai_face_info_t> vec_bbox;
  std::vector<cvai_face_info_t> vec_bbox_nms;

  float* score_data = nullptr;
  float* bbox_data = nullptr;
  float* landmark_data = nullptr;

  for (size_t i = 0; i < m_feat_stride_fpn.size(); i++) {
    int stride = m_feat_stride_fpn[i];

    auto score_tensor = output_tensor[i];
    int feat_score_0 = output_tensor[i]->shape()[0];
    int feat_score_1 = output_tensor[i]->shape()[1];
    int feat_score_2 = output_tensor[i]->shape()[2];
    float* score_data = reinterpret_cast<float*>(output_tensor[i]->sys_data());
    float* tensor_data_score =  score_data + batch_idx * feat_score_1 * feat_score_2;

    auto bbox_tensor = output_tensor[i + 3];
    int feat_bbox_0 = output_tensor[i + 3]->shape()[0];
    int feat_bbox_1 = output_tensor[i + 3]->shape()[1];
    int feat_bbox_2 = output_tensor[i + 3]->shape()[2];
    float* bbox_data = reinterpret_cast<float*>(output_tensor[i + 3]->sys_data());
    float* tensor_data_bbox= bbox_data + batch_idx * feat_bbox_1 * feat_bbox_2;
    auto landmark_tensor = output_tensor[i + 6];
    int feat_landmark_0 = output_tensor[i + 6]->shape()[0];
    int feat_landmark_1 = output_tensor[i + 6]->shape()[1];
    int feat_landmark_2 = output_tensor[i + 6]->shape()[2];
    float* landmark_data = reinterpret_cast<float*>(output_tensor[i + 6]->sys_data());
    float* tensor_data_landmark = landmark_data + batch_idx * feat_landmark_1 * feat_landmark_2;

    std::vector<std::vector<float>>& fpn_grids = fpn_anchors_[stride];
    for (size_t num = 0; num < score_tensor->shape()[1]; num++) {                    // anchor index
      float conf = tensor_data_score[num];  // conf
      if (conf <= m_confThreshold) {
        continue;
      }
      std::vector<float>& grid = fpn_grids[num];
      float grid_cx = (grid[0] + grid[2]) / 2;
      float grid_cy = (grid[1] + grid[3]) / 2;

      cvai_face_info_t box;
      memset(&box, 0, sizeof(box));
      box.pts.size = 5;
      box.pts.x = (float*)malloc(sizeof(float) * box.pts.size);
      box.pts.y = (float*)malloc(sizeof(float) * box.pts.size);
      box.bbox.score = conf;

      box.bbox.x1 = grid_cx - tensor_data_bbox[num * 4 + 0] * stride;
      box.bbox.y1 = grid_cy - tensor_data_bbox[num * 4 + 1] * stride;
      box.bbox.x2 = grid_cx + tensor_data_bbox[num * 4 + 2] * stride;
      box.bbox.y2 = grid_cy + tensor_data_bbox[num * 4 + 3] * stride;

      for (size_t k = 0; k < box.pts.size; k++) {
        box.pts.x[k] = tensor_data_landmark[num * 10 + k * 2] * stride + grid_cx;
        box.pts.y[k] = tensor_data_landmark[num * 10 + k * 2 + 1] * stride + grid_cy;
      }
      vec_bbox.push_back(box);
    }
  }
  vec_bbox_nms.clear();

  NonMaximumSuppression(vec_bbox, vec_bbox_nms, m_nmsThreshold, 'u');
  for (uint32_t i = 0; i < vec_bbox_nms.size(); ++i) {
    clip_boxes(frame_width, frame_height, vec_bbox_nms[i].bbox);
    cvai_bbox_t& bbox = vec_bbox_nms[i].bbox;
    bbox.x1 = (bbox.x1 - pad_x) * im_scale_w;
    bbox.y1 = (bbox.y1 - pad_y) * im_scale_h;
    bbox.x2 = (bbox.x2 - pad_x) * im_scale_w;
    bbox.y2 = (bbox.y2 - pad_y) * im_scale_h;
    cvai_pts_t& pts = vec_bbox_nms[i].pts;
    for (int k = 0; k < pts.size; k++) {
      pts.x[k] = (pts.x[k] - pad_x) * im_scale_w;
      pts.y[k] = (pts.y[k] - pad_y) * im_scale_h;
    }
  }
  return vec_bbox_nms;
}

void Scrfd::readDirectory(const std::string& directory,
                          std::vector<std::string>& files_vector,
                          bool recursive) {
  DIR* pDir = opendir(directory.c_str());
  if (pDir == nullptr) {
    std::cerr << "Unable to open directory: " << directory << std::endl;
    return;
  }

  struct dirent* ptr;
  while ((ptr = readdir(pDir)) != nullptr) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      std::string fullPath = directory + "/" + ptr->d_name;
      if (ptr->d_type == DT_DIR && recursive) {
        // 如果是目录，并且允许递归，则读取子目录
        readDirectory(fullPath, files_vector, false);  // 只递归一层
      } else {
        // 如果是文件，添加到向量中
        files_vector.push_back(fullPath);
      }
    }
  }
  closedir(pDir);
}