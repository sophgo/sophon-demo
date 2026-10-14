//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef DFINE_H
#define DFINE_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

struct DFineBox {
  float x1, y1, x2, y2;
  float score;
  int class_id;
};

using DFineBoxVec = std::vector<DFineBox>;

class DFine {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  //configuration
  float m_confThreshold = 0.4;

  std::vector<std::string> m_class_names;
  int m_net_h, m_net_w;
  int max_batch;
  int output_num;
  bmcv_convert_to_attr converto_attr;

  // orig_size input tensor device memory ([batch, 2] int32)
  bm_device_mem_t m_orig_size_mem;
  bool m_orig_size_mem_allocated;

  // output tensor indices (labels, boxes, scores)
  int m_labels_idx, m_boxes_idx, m_scores_idx;

  TimeStamp *m_ts = nullptr;

  private:
  int pre_process(const std::vector<bm_image>& images);
  int post_process(const std::vector<bm_image>& images, std::vector<DFineBoxVec>& boxes);

  public:
  DFine(std::shared_ptr<BMNNContext> context);
  virtual ~DFine();
  int Init(float confThresh = 0.4, const std::string& coco_names_file = "");
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Detect(const std::vector<bm_image>& images, std::vector<DFineBoxVec>& boxes);
  void draw_bmcv(bm_handle_t &handle, int classId, float conf, int x1, int y1, int x2, int y2, bm_image& frame);
};

#endif //!DFINE_H
