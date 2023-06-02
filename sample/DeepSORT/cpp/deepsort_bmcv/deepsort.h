//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <iostream>
#include "FeatureExtractor.h"
#include "dataType.h"
#include "model.h"
#include "tracker.h"
#include "yolov5.hpp"
using std::vector;
struct TrackBox: YoloV5Box{
    TrackBox(float x = 0,
              float y = 0,
              float w = 0,
              float h = 0,
              float score = 0,
              float class_id = -1,
              float track_id = -1) {
        this->x = x;
        this->y = y;
        this->width = w;
        this->height = h;
        this->score = score;
        this->class_id = class_id;
        this->track_id = track_id;
    }
    int track_id;
};

struct deepsort_params{
    // detector:
    float conf_thresh;
    float nms_thresh;
    // extractor:
    float max_dist;
    float min_confidence;
    float max_iou_distance;
    float max_age;
    float n_init;
    float nn_budget;
};

class DeepSort {
   public:
    DeepSort(std::shared_ptr<BMNNContext> context, const deepsort_params& params);
    virtual ~DeepSort();
    void sort(bm_image& frame, vector<YoloV5Box>& dets, vector<TrackBox>& track_boxs, int frame_id);

    TimeStamp* m_ts;
    void enableProfile(TimeStamp* ts) {
        m_ts = ts;
        featureExtractor->m_ts = ts;
    }

   private:
    FeatureExtractor* featureExtractor;
    tracker* objTracker;
};

#endif  // deepsort.h

