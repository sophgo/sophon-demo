//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "kalmanfilter.h"

void Cholesky(const cv::Mat& A, cv::Mat& S) {
  S = A.clone();
  cv::Cholesky((float*)S.ptr(), S.step, S.rows, NULL, 0, 0);
  S = S.t();
  for (int i = 1; i < S.rows; i++) {
    for (int j = 0; j < i; j++) {
      S.at<float>(i, j) = 0;
    }
  }
}

// sisyphus
const double KalmanFilter::chi2inv95[10] = {
    0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};

KalmanFilter::KalmanFilter() {
  this->_std_weight_position = 1. / 20;
  this->_std_weight_velocity = 1. / 160;

  opencv_kf = std::make_unique<cv::KalmanFilter>(8, 4);
  // 设置状态转移矩阵
  opencv_kf->transitionMatrix =
      (cv::Mat_<float>(8, 8) << 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1);

  // 设置测量矩阵
  opencv_kf->measurementMatrix =
      (cv::Mat_<float>(4, 8) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0);
}

KalmanFilter::~KalmanFilter() {}

std::pair<cv::Mat, cv::Mat> KalmanFilter::initiate(const cv::Mat& measurement) {
  cv::Mat mean_pos = measurement.clone();
  cv::Mat mean_vel = cv::Mat::zeros(1, 4, CV_32F);

  cv::Mat mean(1, 8, CV_32F);
  for (int i = 0; i < 8; i++) {
    if (i < 4)
      mean.at<float>(0, i) = mean_pos.at<float>(0, i);
    else
      mean.at<float>(0, i) = mean_vel.at<float>(0, i - 4);
  }

  cv::Mat std(1, 8, CV_32F);
  std.at<float>(0) = 2 * _std_weight_position * measurement.at<float>(0, 3);
  std.at<float>(1) = 2 * _std_weight_position * measurement.at<float>(0, 3);
  std.at<float>(2) = 1e-2;
  std.at<float>(3) = 2 * _std_weight_position * measurement.at<float>(0, 3);
  std.at<float>(4) = 10 * _std_weight_velocity * measurement.at<float>(0, 3);
  std.at<float>(5) = 10 * _std_weight_velocity * measurement.at<float>(0, 3);
  std.at<float>(6) = 1e-5;
  std.at<float>(7) = 10 * _std_weight_velocity * measurement.at<float>(0, 3);

  cv::Mat tmp = std.mul(std);
  cv::Mat var = cv::Mat::diag(tmp);

  return std::make_pair(mean, var);
}

std::pair<cv::Mat, cv::Mat> KalmanFilter::predict(const cv::Mat& mean,
                                                  const cv::Mat& covariance) {
  float std_pos = _std_weight_position * mean.at<float>(3) *
                  _std_weight_position * mean.at<float>(3);
  float std_vel = _std_weight_velocity * mean.at<float>(3) *
                  _std_weight_velocity * mean.at<float>(3);
  opencv_kf->processNoiseCov =
      (cv::Mat_<float>(8, 8) << std_pos, 0, 0, 0, 0, 0, 0, 0, 0, std_pos, 0, 0,
       0, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 0, 0, 0, std_pos, 0, 0, 0, 0, 0,
       0, 0, 0, std_vel, 0, 0, 0, 0, 0, 0, 0, 0, std_vel, 0, 0, 0, 0, 0, 0, 0,
       0, 1e-10, 0, 0, 0, 0, 0, 0, 0, 0, std_vel);
  opencv_kf->statePost = mean.t();
  opencv_kf->errorCovPost = covariance;

  opencv_kf->predict();

  return std::make_pair(opencv_kf->statePost.t(), opencv_kf->errorCovPost);
}

std::pair<cv::Mat, cv::Mat> KalmanFilter::update(const cv::Mat& mean,
                                                 const cv::Mat& covariance,
                                                 const cv::Mat& measurement) {
  opencv_kf->statePre = mean.t();
  opencv_kf->errorCovPre = covariance;
  float std_pos = _std_weight_position * mean.at<float>(3) *
                  _std_weight_position * mean.at<float>(3);
  opencv_kf->measurementNoiseCov =
      (cv::Mat_<float>(4, 4) << std_pos, 0, 0, 0, 0, std_pos, 0, 0, 0, 0, 1e-2,
       0, 0, 0, 0, std_pos);

  opencv_kf->correct(measurement.t());

  return std::make_pair(opencv_kf->statePost.t(), opencv_kf->errorCovPost);
}

cv::Mat KalmanFilter::gating_distance(const cv::Mat& mean,
                                      const cv::Mat& covariance,
                                      const std::vector<cv::Mat>& measurements,
                                      bool only_position) {
  if (only_position) {
    printf("not implement!");
    exit(0);
  }

  cv::Mat std(1, 4, CV_32F);
  std.at<float>(0) = _std_weight_position * mean.at<float>(3);
  std.at<float>(1) = _std_weight_position * mean.at<float>(3);
  std.at<float>(2) = 1e-1;
  std.at<float>(3) = _std_weight_position * mean.at<float>(3);

  cv::Mat mean1 = opencv_kf->measurementMatrix * mean.t();
  cv::Mat covariance1 = opencv_kf->measurementMatrix * covariance *
                        opencv_kf->measurementMatrix.t();

  cv::Mat diag = cv::Mat::zeros(4, 4, CV_32F);
  diag.at<float>(0, 0) = std.at<float>(0) * std.at<float>(0);
  diag.at<float>(1, 1) = std.at<float>(1) * std.at<float>(1);
  diag.at<float>(2, 2) = std.at<float>(2) * std.at<float>(2);
  diag.at<float>(3, 3) = std.at<float>(3) * std.at<float>(3);

  covariance1 += diag;

  cv::Mat d(measurements.size(), 4, CV_32F);
  int pos = 0;
  for (const auto& box : measurements) {
    cv::Mat diff = box - mean1.t();
    diff.copyTo(d.row(pos++));
  }

  cv::Mat cvCovariance = covariance1;
  cv::Mat factor;
  Cholesky(cvCovariance, factor);

  cv::Mat cvD = d;
  cv::Mat cvZ = factor.inv(cv::DECOMP_CHOLESKY) * cvD.t();
  cv::Mat cvZZ = cvZ.mul(cvZ);
  cv::Mat cvSquareMaha = cv::Mat::zeros(1, cvZZ.cols, CV_32F);
  cv::reduce(cvZZ, cvSquareMaha, 0, cv::REDUCE_SUM);

  return cvSquareMaha;
}
