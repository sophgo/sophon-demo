//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#ifndef POSE_POSTPROCESS_H
#define POSE_POSTPROCESS_H

#include "openpose.hpp"
#include "bmnn_utils.h"
#include "pose_postprocess.hpp"

class PoseBlob : public NoCopyable, public std::enable_shared_from_this<PoseBlob> {
    int m_count;
    int m_n, m_c, m_h, m_w;
    float *m_data;
public:
    PoseBlob(int n, int c, int h, int w):m_n(n),m_c(c), m_h(h), m_w(w) {
        m_count = n*c*w*h;
        m_data = new float_t[m_count];
    }

    ~PoseBlob() {
        delete[] m_data;
    }

    std::shared_ptr<PoseBlob> getPtr() {
        return shared_from_this();
    }

    int height() { return m_h;}
    int width() { return m_w;}
    int channels() {return m_c;}
    int num() { return m_n;}

    float* data() {
        return m_data;
    }
};
using PoseBlobPtr = std::shared_ptr<PoseBlob>;

class OpenPosePostProcess {
public:
    OpenPosePostProcess();

    ~OpenPosePostProcess();

    static int Nms(PoseBlobPtr bottom_blob, PoseBlobPtr top_blob, float threshold);

    static void connectBodyPartsCpu(std::vector<float> &poseKeypoints, const float *const heatMapPtr,
                                    const float *const peaksPtr, const cv::Size &heatMapSize, const int maxPeaks,
                                    const int interMinAboveThreshold, const float interThreshold,
                                    const int minSubsetCnt,
                                    const float minSubsetScore, const float scaleFactor,
                                    std::vector<int> &keypointShape, PoseKeyPoints::EModelType model_type);

    static void renderKeypointsCpu(cv::Mat &frame, const std::vector<float> &keypoints, std::vector<int> keyshape,
                                   const std::vector<unsigned int> &pairs, const std::vector<float> colors,
                                   const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                                   const float threshold, float scale);

    static void renderPoseKeypointsCpu(cv::Mat &frame, const std::vector<float> &poseKeypoints, std::vector<int> keyshape,
                           const float renderThreshold, float scale, PoseKeyPoints::EModelType modelType, const bool blendOriginalFrame = true);
    
    static void renderKeypointsBmcv(bm_handle_t &handle, bm_image &frame, const std::vector<float> &keypoints, std::vector<int> keyshape,
                                   const std::vector<unsigned int> &pairs, const std::vector<float> colors,
                                   const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                                   const float threshold, float scale);

    static bm_image renderPoseKeypointsBmcv(bm_handle_t &handle, bm_image &frame, const std::vector<float> &poseKeypoints, std::vector<int> keyshape,
                           const float renderThreshold, float scale, PoseKeyPoints::EModelType modelType, const bool blendOriginalFrame = true);

    static int getKeyPoints(std::shared_ptr<BMNNTensor> tensorPtr, const std::vector<bm_image> &images,
                            std::vector<PoseKeyPoints> &body_keypoints, PoseKeyPoints::EModelType model_type, 
                            float nms_threshold);

    static std::vector<unsigned int> getPosePairs(PoseKeyPoints::EModelType model_type);

    static std::vector<unsigned int> getPoseMapIdx(PoseKeyPoints::EModelType model_type);

    static int getNumberBodyParts(PoseKeyPoints::EModelType model_type);

};
#endif 
