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
#define DUMP_FILE 0
#define USE_MULTICLASS_NMS 1

const std::vector<std::vector<int>> colors = {
    {255, 0, 0},    {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {170, 255, 0},  {85, 255, 0},    {0, 255, 0},     {0, 255, 85},
    {0, 255, 170},  {0, 255, 255},   {0, 170, 255},   {0, 85, 255},
    {0, 0, 255},    {85, 0, 255},    {170, 0, 255},   {255, 0, 255},
    {255, 0, 170},  {255, 0, 85},    {255, 0, 0},     {255, 0, 255},
    {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255},
    {85, 255, 255}};

Scrfd::Scrfd(std::shared_ptr<BMNNContext> context) : m_bmContext(context) {
  std::cout << "Scrfd ctor .." << std::endl;
}

Scrfd::~Scrfd() {
  std::cout << "Scrfd dtor ..." << std::endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for (int i = 0; i < max_batch; i++) {
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int Scrfd::Init(float confThresh, float nmsThresh) {
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
  assert(output_num == 9);
  min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

  // 4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for (int i = 0; i < max_batch; i++) {
    auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w,
                               FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                               &m_resized_imgs[i], strides);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8) {
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w,
                                   FORMAT_RGB_PLANAR, img_dtype,
                                   m_converto_imgs.data(), max_batch);
  assert(BM_SUCCESS == ret);

  // 5.converto
  float input_scale = tensor->get_scale();
  input_scale = input_scale * 1.0 / 255.f;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = 127.5 * input_scale;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = 127.5 * input_scale;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = 127.5 * input_scale;

  return 0;
}

void Scrfd::enableProfile(TimeStamp* ts) { m_ts = ts; }

int Scrfd::batch_size() { return max_batch; };

int Scrfd::Detect(const std::vector<bm_image>& input_images,
                  std::vector<ScrfdBoxVec>& boxes) {
  int ret = 0;
  // 3. preprocess
  m_ts->save("scrfd preprocess", input_images.size());
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  m_ts->save("scrfd preprocess", input_images.size());

  // 4. forward
  m_ts->save("scrfd inference", input_images.size());
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  m_ts->save("scrfd inference", input_images.size());

  // 5. post process
  m_ts->save("scrfd postprocess", input_images.size());
  ret = post_process(input_images, boxes);
  CV_Assert(ret == 0);
  m_ts->save("scrfd postprocess", input_images.size());
  return ret;
}

int Scrfd::pre_process(const std::vector<bm_image>& images) {
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();
  // 1. resize image
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
      bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                      image1.image_format, image1.data_type, &image_aligned,
                      stride2);

      bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;
      bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1,
                         image_aligned);
    } else {
      image_aligned = image1;
    }
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height,
                                          m_net_w, m_net_h, &isAlignWidth);
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

      int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
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
    auto ret = bmcv_image_vpp_convert_padding(
        m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
        &padding_attr, &crop_rect, BMCV_INTER_NEAREST);
#else
    auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i],
                                      &m_resized_imgs[i]);
#endif
    assert(BM_SUCCESS == ret);

#if DUMP_FILE
    cv::Mat resized_img;
    cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
    std::string fname = cv::format("resized_img_%d.jpg", i);
    cv::imwrite(fname, resized_img);
#endif
    if (need_copy) bm_image_destroy(image_aligned);
  }

  // 2. converto
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr,
                              m_resized_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);

  // 3. attach to tensor
  if (image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n);
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(),
                                     &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
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

int Scrfd::post_process(const std::vector<bm_image>& images,
                        std::vector<ScrfdBoxVec>& detected_boxes) {
  ScrfdBoxVec scrfdbox_vec;
  for (int batch_idx = 0; batch_idx < images.size(); ++batch_idx) {
    scrfdbox_vec.clear();
    auto& frame = images[batch_idx];
    int frame_width = frame.width;
    int frame_height = frame.height;
    compute_pad_resize_param(frame_height, frame_width, m_net_h, m_net_w,
                             rescale_params);
    std::vector<cvai_face_info_t> res =
        outputParser(m_net_w, m_net_h, rescale_params, batch_idx);
    scrfdbox_vec = res;
    detected_boxes.push_back(scrfdbox_vec);
  };
  return 0;
}

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
  std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
  for (int i = 0; i < output_num; i++) {
    outputTensors[i] = m_bmNetwork->outputTensor(i);
  }
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

  for (size_t i = 0; i < m_feat_stride_fpn.size(); i++) {
    int stride = m_feat_stride_fpn[i];

    auto score_tensor = outputTensors[i];
    int feat_score_1 = score_tensor->get_shape()->dims[1];
    int feat_score_2 = score_tensor->get_shape()->dims[2];
    float* score_data = (float*)score_tensor->get_cpu_data() +
                        batch_idx * feat_score_1 * feat_score_2;

    auto bbox_tensor = outputTensors[i + 3];
    int feat_bbox_1 = bbox_tensor->get_shape()->dims[1];
    int feat_bbox_2 = bbox_tensor->get_shape()->dims[2];
    float* bbox_data = (float*)bbox_tensor->get_cpu_data() +
                       batch_idx * feat_bbox_1 * feat_bbox_2;

    auto landmark_tensor = outputTensors[i + 6];
    int feat_landmark_1 = landmark_tensor->get_shape()->dims[1];
    int feat_landmark_2 = landmark_tensor->get_shape()->dims[2];
    float* landmark_data = (float*)landmark_tensor->get_cpu_data() +
                           batch_idx * feat_landmark_1 * feat_landmark_2;

    std::vector<std::vector<float>>& fpn_grids = fpn_anchors_[stride];
    for (size_t num = 0; num < score_tensor->get_shape()->dims[1];
         num++) {                    // anchor index
      float conf = score_data[num];  // conf
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

      box.bbox.x1 = grid_cx - bbox_data[num * 4 + 0] * stride;
      box.bbox.y1 = grid_cy - bbox_data[num * 4 + 1] * stride;
      box.bbox.x2 = grid_cx + bbox_data[num * 4 + 2] * stride;
      box.bbox.y2 = grid_cy + bbox_data[num * 4 + 3] * stride;

      for (size_t k = 0; k < box.pts.size; k++) {
        box.pts.x[k] = landmark_data[num * 10 + k * 2] * stride + grid_cx;
        box.pts.y[k] = landmark_data[num * 10 + k * 2 + 1] * stride + grid_cy;
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

void Scrfd::drawPred(std::vector<cvai_face_info_t>& res,
                     cv::Mat& frame)  // Draw the predicted bounding box
{
  // Draw a rectangle displaying the bounding box
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
    std::string label = cv::format("%.3f", bbox.score);
    cv::putText(frame, label, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(0, 255, 0), 1);
  }
}

void Scrfd::draw_bmcv(bm_handle_t& handle, cvai_pts_t five_point, float conf,
                      int left, int top, int width, int height, bm_image& frame,
                      bool put_text_flag,
                      bool draw_point_flag)  // Draw the predicted bounding box
{
  if (conf < 0.25) return;
  bmcv_rect_t rect;
  rect.start_x = MIN(MAX(left, 0), frame.width);
  rect.start_y = MIN(MAX(top, 0), frame.height);
  rect.crop_w = MAX(MIN(width, frame.width - rect.start_x), 0);
  rect.crop_h = MAX(MIN(height, frame.height - rect.start_y), 0);
  auto color_tuple = std::make_tuple(0, 0, 255);
  int thickness = 2;
  if (draw_point_flag) {
    cvai_pts_t pti = five_point;
    for (int j = 0; j < pti.size; j++) {
      int x = (int)pti.x[j];
      int y = (int)pti.y[j];
      auto center = std::make_pair(x, y);
      // bmcv_image_draw_point(handle,frame, center, color_tuple, 2);
    }
  }
  if (rect.crop_w <= thickness * 2 || rect.crop_h <= thickness * 2) {
    std::cout << "width or height too small, this rect will not be drawed: "
              << "[" << rect.start_x << ", " << rect.start_y << ", "
              << rect.crop_w << ", " << rect.crop_h << "]" << std::endl;
  } else {
    bmcv_image_draw_rectangle(handle, frame, 1, &rect, thickness, colors[0][0],
                              colors[0][1], colors[0][2]);
  }
  if (put_text_flag) {  // only support YUV420P, puttext not used here.
    std::string label = cv::format("%.3f", conf);
    bmcv_point_t org = {rect.start_x, rect.start_y};
    bmcv_color_t color = {colors[0][0], colors[0][1], colors[0][2]};
    float fontScale = 2;
    if (BM_SUCCESS != bmcv_image_put_text(handle, frame, label.c_str(), org,
                                          color, fontScale, thickness)) {
      std::cout << "bmcv put text error !!!" << std::endl;
    }
  }
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