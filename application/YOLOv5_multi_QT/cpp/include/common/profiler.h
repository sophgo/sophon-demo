// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===


#ifndef SOPHON_STREAM_COMMON_PROFILER_H_
#define SOPHON_STREAM_COMMON_PROFILER_H_

#include <sys/time.h>

#include <iostream>
#include <mutex>

class FpsProfiler {
public:
  FpsProfiler();
  FpsProfiler(const std::string& name, int summary_cond_cnts = -1);
  ~FpsProfiler();

  void config(const std::string& name, int summary_cond_times);
  void add(int cnts = 1);
  float getTmpFps();

private:
  float elapse();
  void summary();

  std::mutex mutex_;

  std::string name_;

  double start_ts_;
  double end_ts_;

  int cnts_;
  int summary_cond_cnts_;

  float tmp_fps_;
  float avg_fps_;

  double last_print_ts_ = 0;

  // 10ms is too long for resnet, so there may be no tmp_fps.
  float print_step_ = 1;
};

#endif  // SOPHON_STREAM_COMMON_PROFILER_H_
