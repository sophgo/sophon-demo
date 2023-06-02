//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "deepsort.h"
DeepSort::DeepSort(std::shared_ptr<BMNNContext> context,const deepsort_params& params) {
    std::cout << "deepsort ctor .." << std::endl;
    featureExtractor = new FeatureExtractor(context);
    featureExtractor->Init();
    objTracker = new tracker(params.max_dist, 
                             params.nn_budget, 
                             featureExtractor->k_feature_dim,
                             params.max_iou_distance,
                             params.max_age,
                             params.n_init);
}

DeepSort::~DeepSort() {
    delete objTracker;
    delete featureExtractor;
}

void DeepSort::sort(bm_image& frame, vector<YoloV5Box>& dets, vector<TrackBox>& track_boxs, int frame_id) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    for (YoloV5Box i : dets) {
        /*very important, if not this code, you will suffer from segmentation fault.*/
        auto start_x = MIN(MAX(int(i.x), 0), frame.width - 16);
        auto start_y = MIN(MAX(int(i.y), 0), frame.height - 16);
        auto crop_w = MAX(MIN(int(i.width), frame.width - int(i.x)), 16); // vpp resize support width >= 16 
        auto crop_h = MAX(MIN(int(i.height), frame.height - int(i.y)), 16); // vpp resize support height >= 16 

        DETECTBOX box(start_x, start_y, crop_w, crop_h);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.score;
        d.class_id = i.class_id;
        detections.push_back(d);
    }
    track_boxs.clear();
    if (detections.size() > 0) {
        LOG_TS(m_ts, "extractor time");
        bool flag = featureExtractor->getRectsFeature(frame, detections);
        LOG_TS(m_ts, "extractor time");
        LOG_TS(m_ts, "deepsort postprocess");
        if (flag) {
            objTracker->predict();
            objTracker->update(detections);
            // std::cout << "track num: " << objTracker.tracks.size() << std::endl;
            for (Track& track : objTracker->tracks) {
                if ((!track.is_confirmed() || track.time_since_update > 1) && frame_id > 2) { //when frame_id < 2, there is no track.
                    continue;
                }
                DETECTBOX k = track.to_tlwh();
                TrackBox tmp(k(0), k(1), k(2), k(3), 1., track.class_id, track.track_id);
                track_boxs.push_back(tmp);
            }
        }
        LOG_TS(m_ts, "deepsort postprocess");
    }
}
