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

class DeepSort {
   public:
    DeepSort(std::shared_ptr<BMNNContext> context);
    virtual ~DeepSort();
    void sort(bm_image& frame, vector<YoloV5Box>& dets, int frame_id);

    TimeStamp* m_ts;
    void enableProfile(TimeStamp* ts) {
        m_ts = ts;
        featureExtractor->m_ts = ts;
    }

   private:
    // void sort(bm_image& frame, DETECTIONSV2& detectionsv2);
    vector<RESULT_DATA> result;
    FeatureExtractor* featureExtractor;
    // vector<std::pair<CLSCONF, DETECTBOX>> results;
    tracker* objTracker;
    vector<int> respond_clss;
};

#endif  // deepsort.h
