//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "STrack.h"
#include "yolov5.hpp"

struct SaveResult
{
	int frame_id;
	int track_id;
	vector<float> tlwh;
};

class BYTETracker
{
public:
	BYTETracker(int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	vector<STrack> update(const vector<YoloV5Box> &objects);
	Scalar get_color(int idx);
	std::tuple<int, int, int> get_color_tuple(int idx);
	TimeStamp *m_ts;
  void enableProfile(TimeStamp *ts);

private:
	vector<STrack *> joint_stracks(vector<STrack *> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void linear_assignment(vector<vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
												 vector<vector<int>> &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float>> iou_distance(vector<STrack *> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float>> iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float>> ious(vector<vector<float>> &atlbrs, vector<vector<float>> &btlbrs);

	double lapjv(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol,
							 bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};