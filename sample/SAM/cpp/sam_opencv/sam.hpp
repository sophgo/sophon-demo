//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SAM_HPP
#define SAM_HPP

#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "json.hpp"

#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "bmnn_utils.h"

struct SamMask {
    cv::Mat mask;   // binary mask (CV_8UC1, 0 or 255), same size as original image
    float score;    // IoU prediction score
};

class Sam {
    // Two separate bmodel contexts: embedding and decode
    std::shared_ptr<BMNNContext> m_embedContext;
    std::shared_ptr<BMNNContext> m_decodeContext;
    std::shared_ptr<BMNNNetwork> m_embedNetwork;
    std::shared_ptr<BMNNNetwork> m_decodeNetwork;

    // Embedding input tensor + device memory
    std::shared_ptr<BMNNTensor> m_embedInputTensor;
    bm_tensor_t  m_embedInputDev;
    float*       m_inputBuf = nullptr;   // [3 * 1024 * 1024]

    // Decode input tensors (6 inputs) + their device memory
    std::shared_ptr<BMNNTensor> m_decInputs[6];
    bm_tensor_t  m_decInputDevs[6];
    bool m_tensorsAllocated = false;

    // Decoder output indices (identified by name in Init)
    int m_iouIdx   = -1;   // iou_predictions output [1,4]
    int m_masksIdx  = -1;  // /Slice_9 masks output [1,4,H,W]

    // SAM constants
    static constexpr int IMG_SIZE = 1024;
    float m_pixelMean[3] = {123.675f, 116.28f, 103.53f};
    float m_pixelStd[3]  = {58.395f,  57.12f,  57.375f};

    int        m_devId;
    TimeStamp* m_ts = nullptr;

public:
    Sam(std::shared_ptr<BMNNHandle> handle,
        const std::string& embedBmodel,
        const std::string& decodeBmodel,
        int devId);
    ~Sam();

    int  Init();
    void enableProfile(TimeStamp* ts) { m_ts = ts; }

    // Point prompt: input_point "x,y". Returns up to 4 masks.
    int processPoint(const cv::Mat& bgrImage, int px, int py,
                     std::vector<SamMask>& masks);
    // Box prompt: box (x1,y1,x2,y2). Returns up to 4 masks.
    int processBox(const cv::Mat& bgrImage, int x1, int y1, int x2, int y2,
                   std::vector<SamMask>& masks);

    // Drawing helpers
    void drawMask(cv::Mat& frame, const cv::Mat& mask,
                  const cv::Scalar& color, double alpha = 0.5);
    void drawPoint(cv::Mat& frame, int x, int y,
                   const cv::Scalar& color = cv::Scalar(0, 255, 0));
    void drawBox(cv::Mat& frame, int x1, int y1, int x2, int y2,
                 const cv::Scalar& color = cv::Scalar(0, 255, 0));

private:
    void getPreprocessShape(int oldh, int oldw, int longSide,
                             int& newh, int& neww);
    // Embedding: preprocess + forward. Fills m_inputBuf, runs embedding,
    // copies output to host. Returns embedding in `embedding` (host float vector).
    int  embedImage(const cv::Mat& bgrImage, int origH, int origW,
                    int& newH, int& newW,
                    std::vector<float>& embedding);
    // Run decoder given embedding + point/box coords.
    int  runDecoder(const std::vector<float>& embedding,
                    float pointCoords[4], float pointLabels[2],
                    int origH, int origW, int newH, int newW,
                    std::vector<SamMask>& masks);
    // Copy an output tensor to host as float32 (handles F32/F16).
    // `useEmbedNetwork` selects which network's output to read from:
    //   true  → embedding network (for embedImage), false → decode network.
    int  retrieveOutput(int outputIdx, std::vector<float>& out,
                        bm_shape_t& outShape, bool useEmbedNetwork = false);
};

#endif // SAM_HPP
