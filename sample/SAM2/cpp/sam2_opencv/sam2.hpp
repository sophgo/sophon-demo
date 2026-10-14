//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SAM2_HPP
#define SAM2_HPP

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

struct Sam2Mask {
    cv::Mat mask;   // binary mask (CV_8UC1, 0 or 255), same size as original image
    float score;    // IoU prediction score
};

class Sam2 {
    // Two separate bmodel contexts: encoder and decoder
    std::shared_ptr<BMNNContext> m_encContext;
    std::shared_ptr<BMNNContext> m_decContext;
    std::shared_ptr<BMNNNetwork> m_encNetwork;
    std::shared_ptr<BMNNNetwork> m_decNetwork;

    // Encoder input tensor + device memory
    std::shared_ptr<BMNNTensor> m_encInputTensor;
    bm_tensor_t  m_encInputDev;
    float*       m_inputBuf = nullptr;   // [3 * 1024 * 1024]

    // Decoder input tensors (7 inputs) + their device memory
    std::shared_ptr<BMNNTensor> m_decInputs[7];
    bm_tensor_t  m_decInputDevs[7];
    bool m_tensorsAllocated = false;

    // Encoder output indices for the 3 outputs:
    //   image_embed [1,256,64,64], high_res_0 [1,32,256,256], high_res_1 [1,64,128,128]
    int m_embedIdx    = -1;
    int m_highRes0Idx = -1;
    int m_highRes1Idx = -1;

    // Decoder output indices (identified by name in Init)
    int m_iouIdx   = -1;   // iou_predictions output [1,3]
    int m_masksIdx  = -1;  // masks output [1,3,256,256]

    // SAM2 constants
    static constexpr int IMG_SIZE = 1024;
    float m_pixelMean[3] = {0.485f, 0.456f, 0.406f};
    float m_pixelStd[3]  = {0.229f, 0.224f, 0.225f};

    int        m_devId;
    TimeStamp* m_ts = nullptr;

public:
    Sam2(std::shared_ptr<BMNNHandle> handle,
         const std::string& encoderBmodel,
         const std::string& decoderBmodel,
         int devId);
    ~Sam2();

    int  Init();
    void enableProfile(TimeStamp* ts) { m_ts = ts; }

    // Point prompt: input_point "x,y". Returns up to 3 masks.
    int processPoint(const cv::Mat& bgrImage, int px, int py,
                     std::vector<Sam2Mask>& masks);
    // Box prompt: box (x1,y1,x2,y2). Returns up to 3 masks.
    int processBox(const cv::Mat& bgrImage, int x1, int y1, int x2, int y2,
                   std::vector<Sam2Mask>& masks);

    // Drawing helpers
    void drawMask(cv::Mat& frame, const cv::Mat& mask,
                  const cv::Scalar& color, double alpha = 0.5);
    void drawPoint(cv::Mat& frame, int x, int y,
                   const cv::Scalar& color = cv::Scalar(0, 255, 0));
    void drawBox(cv::Mat& frame, int x1, int y1, int x2, int y2,
                 const cv::Scalar& color = cv::Scalar(0, 255, 0));

private:
    // Encoding: preprocess + forward. Returns the 3 encoder outputs as
    // host float vectors (image_embed, high_res_0, high_res_1).
    int  encodeImage(const cv::Mat& bgrImage, int origH, int origW,
                     std::vector<float>& imageEmbed,
                     std::vector<float>& highRes0,
                     std::vector<float>& highRes1);
    // Run decoder given encoder outputs + point/box coords.
    int  runDecoder(const std::vector<float>& imageEmbed,
                    const std::vector<float>& highRes0,
                    const std::vector<float>& highRes1,
                    float pointCoords[4], float pointLabels[2],
                    int origH, int origW,
                    std::vector<Sam2Mask>& masks);
    // Copy an output tensor to host as float32 (handles F32/F16).
    // useEncoder=true → read from encoder network; false → decoder network.
    int  retrieveOutput(int outputIdx, std::vector<float>& out,
                        bm_shape_t& outShape, bool useEncoder);
};

#endif // SAM2_HPP
