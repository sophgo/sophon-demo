//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "strack.h"

STrack::STrack(std::vector<float> tlwh_, float score, int class_id) {
  this->frame_id = 0;
  this->tracklet_len = 0;
  this->score = score;
  this->class_id = class_id;
  this->start_frame = 0;
  this->is_activated = false;
  this->track_id = 0;
  this->state = TrackState::New;

  _tlwh.resize(4);
  _tlwh.assign(tlwh_.begin(), tlwh_.end());
  tlwh.resize(4);
  tlbr.resize(4);
  static_tlwh();
  static_tlbr();
}

STrack::~STrack() {}

void STrack::activate(std::shared_ptr<KalmanFilter> kalman_filter,
                      int frame_id) {
  this->track_id = this->next_id();

  std::vector<float> _tlwh_tmp(4);
  _tlwh_tmp[0] = this->_tlwh[0];
  _tlwh_tmp[1] = this->_tlwh[1];
  _tlwh_tmp[2] = this->_tlwh[2];
  _tlwh_tmp[3] = this->_tlwh[3];
  std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
  cv::Mat xyah_box(1, 4, CV_32F);
  xyah_box.at<float>(0) = xyah[0];
  xyah_box.at<float>(1) = xyah[1];
  xyah_box.at<float>(2) = xyah[2];
  xyah_box.at<float>(3) = xyah[3];
  auto mc = kalman_filter->initiate(xyah_box);
  this->mean = mc.first.clone();
  this->covariance = mc.second.clone();

  static_tlwh();
  static_tlbr();

  this->tracklet_len = 0;
  this->state = TrackState::Tracked;
  if (frame_id == 1) {
    this->is_activated = true;
  }
  // this->is_activated = true;
  this->frame_id = frame_id;
  this->start_frame = frame_id;
}

void STrack::re_activate(std::shared_ptr<KalmanFilter> kalman_filter,
                         std::shared_ptr<STrack> new_track, int frame_id,
                         bool new_id) {
  std::vector<float> xyah = tlwh_to_xyah(new_track->tlwh);
  cv::Mat xyah_box(1, 4, CV_32F);
  xyah_box.at<float>(0) = xyah[0];
  xyah_box.at<float>(1) = xyah[1];
  xyah_box.at<float>(2) = xyah[2];
  xyah_box.at<float>(3) = xyah[3];
  auto mc = kalman_filter->update(this->mean, this->covariance, xyah_box);
  this->mean = mc.first.clone();
  this->covariance = mc.second.clone();

  static_tlwh();
  static_tlbr();

  this->tracklet_len = 0;
  this->state = TrackState::Tracked;
  this->is_activated = true;
  this->frame_id = frame_id;
  this->score = new_track->score;
  if (new_id) this->track_id = next_id();
}

void STrack::update(std::shared_ptr<KalmanFilter> kalman_filter,
                    std::shared_ptr<STrack> new_track, int frame_id) {
  this->frame_id = frame_id;
  this->tracklet_len++;

  std::vector<float> xyah = tlwh_to_xyah(new_track->tlwh);
  cv::Mat xyah_box(1, 4, CV_32F);
  xyah_box.at<float>(0) = xyah[0];
  xyah_box.at<float>(1) = xyah[1];
  xyah_box.at<float>(2) = xyah[2];
  xyah_box.at<float>(3) = xyah[3];

  auto mc = kalman_filter->update(this->mean, this->covariance, xyah_box);
  this->mean = mc.first.clone();
  this->covariance = mc.second.clone();

  static_tlwh();
  static_tlbr();

  this->state = TrackState::Tracked;
  this->is_activated = true;
  this->score = new_track->score;
}

void STrack::static_tlwh() {
  if (this->state == TrackState::New) {
    tlwh[0] = _tlwh[0];
    tlwh[1] = _tlwh[1];
    tlwh[2] = _tlwh[2];
    tlwh[3] = _tlwh[3];
    return;
  }

  tlwh[0] = mean.at<float>(0);
  tlwh[1] = mean.at<float>(1);
  tlwh[2] = mean.at<float>(2);
  tlwh[3] = mean.at<float>(3);

  tlwh[2] *= tlwh[3];
  tlwh[0] -= tlwh[2] / 2;
  tlwh[1] -= tlwh[3] / 2;
}

void STrack::static_tlbr() {
  tlbr.clear();
  tlbr.assign(tlwh.begin(), tlwh.end());
  tlbr[2] += tlbr[0];
  tlbr[3] += tlbr[1];
}

std::vector<float> STrack::tlwh_to_xyah(std::vector<float> tlwh_tmp) {
  std::vector<float> tlwh_output = tlwh_tmp;
  tlwh_output[0] += tlwh_output[2] / 2;
  tlwh_output[1] += tlwh_output[3] / 2;
  tlwh_output[2] /= tlwh_output[3];
  return tlwh_output;
}

std::vector<float> STrack::to_xyah() { return tlwh_to_xyah(tlwh); }

std::vector<float> STrack::tlbr_to_tlwh(std::vector<float>& tlbr) {
  tlbr[2] -= tlbr[0];
  tlbr[3] -= tlbr[1];
  return tlbr;
}

void STrack::mark_lost() { state = TrackState::Lost; }

void STrack::mark_removed() { state = TrackState::Removed; }

int STrack::next_id() {
  static int _count = 0;
  _count++;
  return _count;
}

int STrack::end_frame() { return this->frame_id; }

void STrack::multi_predict(std::vector<std::shared_ptr<STrack>>& stracks,
                           std::shared_ptr<KalmanFilter> kalman_filter) {
  for (int i = 0; i < stracks.size(); i++) {
    if (stracks[i]->state != TrackState::Tracked) {
      stracks[i]->mean.at<float>(7) = 0;
    }
    auto mc = kalman_filter->predict(stracks[i]->mean, stracks[i]->covariance);
    stracks[i]->mean = mc.first.clone();
    stracks[i]->covariance = mc.second.clone();
  }
}