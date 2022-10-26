//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
//  face_detection.hpp
//  NelivaSDK
//
//  Created by Bitmain on 2020/11/5.
//  Copyright © 2020年 AnBaolei. All rights reserved.
//

#ifndef face_detection_hpp
#define face_detection_hpp

#include <cstring>
#include <memory>
#include <opencv2/opencv.hpp>
#include "bmodel_base.hpp"
#include "retinaface_post.hpp"

class FaceDetection : public BmodelBase{

public:
  FaceDetection(const std::string bmodel, int device_id);
  ~FaceDetection();

  bool run(std::vector<cv::Mat>& input_imgs,
                            std::vector<std::vector<stFaceRect> >& results);
  void set_max_face_count(int max_face_count);
  void set_score_threshold(float score_threshold);

private:
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
  std::vector<float> preprocess(const std::vector<bm_image>& input_imgs);
  void postprocess(std::vector<std::vector<stFaceRect> >& results, std::vector<float>ratios, std::vector<cv::Mat>& input_imgs);

private:
  std::shared_ptr<RetinaFacePostProcess> post_process_;
  int max_face_count_ = 50;
  float score_threshold_ = 0.5f;
};

#endif /* face_detection_hpp */
