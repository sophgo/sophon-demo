// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "profiler.h"
#include <opencv2/opencv.hpp>

FpsProfiler::FpsProfiler() {}

FpsProfiler::FpsProfiler(const std::string& name, int summary_cond_cnts)
    : name_(name), summary_cond_cnts_(summary_cond_cnts) {
  tmp_fps_ = 0;
  avg_fps_ = 0;
  cnts_ = 0;
  start_ts_ = cv::getTickCount() / cv::getTickFrequency();
}

FpsProfiler::~FpsProfiler() {}

void FpsProfiler::config(const std::string& name, int summary_cond_cnts) {
  name_ = name;
  summary_cond_cnts_ = summary_cond_cnts;
  tmp_fps_ = 0;
  avg_fps_ = 0;
  cnts_ = 0;
  start_ts_ = cv::getTickCount() / cv::getTickFrequency();
}

void FpsProfiler::add(int cnts) {
  std::lock_guard<std::mutex> lock(mutex_);
  cnts_ += cnts;
  if ((summary_cond_cnts_ > 0 && cnts_ > summary_cond_cnts_) ||
      (elapse() > print_step_ && print_step_ != 0)) {
    summary();
  }
  double ts = cv::getTickCount() / cv::getTickFrequency();
  if (ts - last_print_ts_ > print_step_ && print_step_ != 0) {
    std::cout<<"["<<name_<<"] tmp_fps: "<<tmp_fps_<<", avg_fps: "<<avg_fps_<<std::endl;
    last_print_ts_ = ts;
  }
}

float FpsProfiler::getTmpFps() {
  return tmp_fps_;
}

float FpsProfiler::elapse() {
  end_ts_ = cv::getTickCount() / cv::getTickFrequency();
  return end_ts_ - start_ts_;
}

void FpsProfiler::summary() {
  tmp_fps_ = cnts_ / elapse();
  // use moving avg
  avg_fps_ = avg_fps_ * 0.9 + tmp_fps_ * 0.1;

  cnts_ = 0;
  start_ts_ = cv::getTickCount() / cv::getTickFrequency();
}