//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef BYTETRACK_H
#define BYTETRACK_H

#include "lapjv.h"
#include "strack.h"
#include "yolov5.hpp"

struct bytetrack_params {
  // detector:
  float conf_thresh;
  float nms_thresh;
  // tracker:
  float track_thresh;
  float match_thresh;
  int frame_rate;
  int track_buffer;
  int min_box_area;
};

class BYTETracker {
 public:
  BYTETracker(const bytetrack_params& params);
  ~BYTETracker();

  TimeStamp* m_ts;
  void enableProfile(TimeStamp* ts);

  void update(STracks& output_stracks, const std::vector<YoloV5Box>& objects);

 private:
  void joint_stracks(STracks& tlista, STracks& tlistb, STracks& results);

  void sub_stracks(STracks& tlista, STracks& tlistb);

  void remove_duplicate_stracks(STracks& resa, STracks& resb, STracks& stracksa,
                                STracks& stracksb);

  void linear_assignment(std::vector<std::vector<float>>& cost_matrix,
                         int cost_matrix_size, int cost_matrix_size_size,
                         float thresh, std::vector<std::vector<int>>& matches,
                         std::vector<int>& unmatched_a,
                         std::vector<int>& unmatched_b);

  void iou_distance(const STracks& atracks, const STracks& btracks,
                    std::vector<std::vector<float>>& cost_matrix);

  void ious(std::vector<std::vector<float>>& atlbrs,
            std::vector<std::vector<float>>& btlbrs,
            std::vector<std::vector<float>>& results);

  void lapjv(const std::vector<std::vector<float>>& cost,
             std::vector<int>& rowsol, std::vector<int>& colsol,
             bool extend_cost = false, float cost_limit = LONG_MAX,
             bool return_cost = true);

 private:
  float track_thresh;
  float match_thresh;
  int frame_rate;
  int track_buffer;
  int min_box_area;
  int frame_id;
  int max_time_lost;

  STracks tracked_stracks;
  STracks lost_stracks;
  STracks removed_stracks;

  std::shared_ptr<KalmanFilter> kalman_filter;
};

#endif  // BYTETRACK_H