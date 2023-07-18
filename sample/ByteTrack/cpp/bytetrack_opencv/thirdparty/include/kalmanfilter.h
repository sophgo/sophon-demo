//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef KALMANFILTER_H
#define KALMANFILTER_H
#include <opencv2/opencv.hpp>
#include <iostream>

class KalmanFilter {
 public:
  static const double chi2inv95[10];
  KalmanFilter();
  ~KalmanFilter();
  std::pair<cv::Mat, cv::Mat> initiate(const cv::Mat& measurement);
  std::pair<cv::Mat, cv::Mat> predict(const cv::Mat& mean,
                                      const cv::Mat& covariance);
  std::pair<cv::Mat, cv::Mat> update(const cv::Mat& mean,
                                     const cv::Mat& covariance,
                                     const cv::Mat& measurement);
  cv::Mat gating_distance(const cv::Mat& mean, const cv::Mat& covariance,
                          const std::vector<cv::Mat>& measurements,
                          bool only_position = false);

 private:
  std::unique_ptr<cv::KalmanFilter> opencv_kf;
  float _std_weight_position;
  float _std_weight_velocity;
};

#endif  // KALMANFILTER_H
