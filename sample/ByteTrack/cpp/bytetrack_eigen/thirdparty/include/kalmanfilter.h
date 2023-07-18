//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;
// typedef std::vector<FEATURE> FEATURESS;

// Kalmanfilter
// typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

// main
using RESULT_DATA = std::pair<int, DETECTBOX>;

// tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t {
  std::vector<MATCH_DATA> matches;
  std::vector<int> unmatched_tracks;
  std::vector<int> unmatched_detections;
} TRACHER_MATCHD;

// linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

class KalmanFilter {
 public:
  static const double chi2inv95[10];
  KalmanFilter();
  KAL_DATA initiate(const DETECTBOX& measurement);
  void predict(KAL_MEAN& mean, KAL_COVA& covariance);
  KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
  KAL_DATA update(const KAL_MEAN& mean, const KAL_COVA& covariance,
                  const DETECTBOX& measurement);

  Eigen::Matrix<float, 1, -1> gating_distance(
      const KAL_MEAN& mean, const KAL_COVA& covariance,
      const std::vector<DETECTBOX>& measurements, bool only_position = false);

 private:
  Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
  Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
  float _std_weight_position;
  float _std_weight_velocity;
};
