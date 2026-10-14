//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "sam2.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// fp16 ↔ fp32 conversion (bmodel outputs may be F16; BMNNTensor::get_cpu_data()
// does not handle BM_FLOAT16, so we copy raw bytes and convert ourselves)
// ---------------------------------------------------------------------------
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t frac =  h         & 0x3ffu;
    uint32_t f;

    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;
        } else {
            int e = 1;
            do { frac <<= 1; e--; } while ((frac & 0x400u) == 0);
            frac &= 0x3ffu;
            f = (sign << 31) | ((uint32_t)(127 + e - 15) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xffu << 23) | (frac << 13);
    } else {
        f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (frac << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(uint32_t));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t frac =  x & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        frac |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint16_t h = (uint16_t)(sign | (frac >> shift));
        if ((frac >> (shift - 1)) & 1) h++;
        return h;
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (frac >> 13));
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
Sam2::Sam2(std::shared_ptr<BMNNHandle> handle,
           const std::string& encoderBmodel,
           const std::string& decoderBmodel,
           int devId)
    : m_devId(devId) {
    m_encContext = std::make_shared<BMNNContext>(handle, encoderBmodel.c_str());
    m_decContext = std::make_shared<BMNNContext>(handle, decoderBmodel.c_str());
}

Sam2::~Sam2() {
    if (m_inputBuf) delete[] m_inputBuf;
    if (m_tensorsAllocated) {
        bm_handle_t h = m_encContext->handle();
        bm_free_device(h, m_encInputDev.device_mem);
        bm_handle_t hd = m_decContext->handle();
        for (int i = 0; i < 7; i++)
            bm_free_device(hd, m_decInputDevs[i].device_mem);
    }
}

// ---------------------------------------------------------------------------
// Init — load both networks, identify I/O, allocate tensors
// ---------------------------------------------------------------------------
int Sam2::Init() {
    // --- Encoder network ---
    m_encNetwork = m_encContext->network(0);
    m_encInputTensor = m_encNetwork->inputTensor(0);
    auto encInShape = m_encInputTensor->get_shape();
    assert(encInShape->num_dims == 4);
    assert(encInShape->dims[2] == IMG_SIZE && encInShape->dims[3] == IMG_SIZE);

    m_inputBuf = new float[3 * IMG_SIZE * IMG_SIZE];

    bmrt_tensor(&m_encInputDev, m_encContext->bmrt(),
                m_encInputTensor->get_dtype(), *encInShape);
    m_encInputTensor->set_device_mem(&m_encInputDev.device_mem);

    // Identify encoder output indices by name:
    //   image_embed [1,256,64,64], high_res_0 [1,32,256,256], high_res_1 [1,64,128,128]
    int nEncOutputs = m_encNetwork->outputTensorNum();
    for (int i = 0; i < nEncOutputs; i++) {
        std::string name = m_encNetwork->m_netinfo->output_names[i];
        if (name.find("image_embed") != std::string::npos)
            m_embedIdx = i;
        if (name.find("conv_s0") != std::string::npos || name.find("high_res_feats_0") != std::string::npos)
            m_highRes0Idx = i;
        if (name.find("conv_s1") != std::string::npos || name.find("high_res_feats_1") != std::string::npos)
            m_highRes1Idx = i;
    }
    // Fallback by shape if names didn't match
    if (m_embedIdx < 0 || m_highRes0Idx < 0 || m_highRes1Idx < 0) {
        for (int i = 0; i < nEncOutputs; i++) {
            auto t = m_encNetwork->outputTensor(i);
            auto s = t->get_shape();
            // image_embed [1,256,64,64], high_res_0 [1,32,256,256], high_res_1 [1,64,128,128]
            if (s->num_dims == 4 && s->dims[1] == 256 && s->dims[2] == 64)
                m_embedIdx = (m_embedIdx < 0) ? i : m_embedIdx;
            else if (s->num_dims == 4 && s->dims[1] == 32)
                m_highRes0Idx = (m_highRes0Idx < 0) ? i : m_highRes0Idx;
            else if (s->num_dims == 4 && s->dims[1] == 64 && s->dims[2] == 128)
                m_highRes1Idx = (m_highRes1Idx < 0) ? i : m_highRes1Idx;
        }
    }
    std::cout << "[SAM2] enc: embedIdx=" << m_embedIdx
              << " highRes0Idx=" << m_highRes0Idx
              << " highRes1Idx=" << m_highRes1Idx << std::endl;
    assert(m_embedIdx >= 0 && m_highRes0Idx >= 0 && m_highRes1Idx >= 0);

    // --- Decoder network ---
    m_decNetwork = m_decContext->network(0);
    int nInputs = m_decNetwork->m_netinfo->input_num;
    assert(nInputs == 7);
    for (int i = 0; i < 7; i++) {
        m_decInputs[i] = m_decNetwork->inputTensor(i);
        auto shp = m_decInputs[i]->get_shape();
        bmrt_tensor(&m_decInputDevs[i], m_decContext->bmrt(),
                    m_decInputs[i]->get_dtype(), *shp);
        m_decInputs[i]->set_device_mem(&m_decInputDevs[i].device_mem);
    }

    // Identify decoder output indices by name
    int nOutputs = m_decNetwork->outputTensorNum();
    for (int i = 0; i < nOutputs; i++) {
        std::string name = m_decNetwork->m_netinfo->output_names[i];
        if (name.find("iou") != std::string::npos)
            m_iouIdx = i;
        if (name.find("mask") != std::string::npos)
            m_masksIdx = i;
    }
    std::cout << "[SAM2] dec: iouIdx=" << m_iouIdx
              << " masksIdx=" << m_masksIdx << std::endl;
    assert(m_iouIdx >= 0 && m_masksIdx >= 0);

    m_tensorsAllocated = true;
    return 0;
}

// ---------------------------------------------------------------------------
// encodeImage — preprocess (resize + normalize) + encoder forward
//   BGR input → RGB → resize to (1024,1024) INTER_NEAREST → normalize → forward
//   SAM2 encoder does NOT pad; it resizes directly to 1024×1024.
// ---------------------------------------------------------------------------
int Sam2::encodeImage(const cv::Mat& bgrImage, int origH, int origW,
                       std::vector<float>& imageEmbed,
                       std::vector<float>& highRes0,
                       std::vector<float>& highRes1) {
    // BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(bgrImage, rgb, cv::COLOR_BGR2RGB);

    // Resize to 1024×1024 using INTER_NEAREST (matches Python sam2_image_opencv.py)
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, cv::INTER_NEAREST);

    // HWC → CHW, normalize: (x/255 - mean) / std
    int C = 3;
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                cv::Vec3b px = resized.at<cv::Vec3b>(y, x);
                float val = (float)px[c] / 255.0f;
                m_inputBuf[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] =
                    (val - m_pixelMean[c]) / m_pixelStd[c];
            }
        }
    }

    // Copy to device
    auto dtype = m_encInputTensor->get_dtype();
    if (dtype == BM_FLOAT32) {
        bm_memcpy_s2d(m_encContext->handle(), m_encInputDev.device_mem,
                      (void*)m_inputBuf);
    } else if (dtype == BM_FLOAT16) {
        size_t count = (size_t)3 * IMG_SIZE * IMG_SIZE;
        std::vector<uint16_t> h16(count);
        for (size_t i = 0; i < count; i++)
            h16[i] = fp32_to_fp16(m_inputBuf[i]);
        bm_memcpy_s2d(m_encContext->handle(), m_encInputDev.device_mem,
                      (void*)h16.data());
    } else {
        std::cerr << "[SAM2] unsupported encoder input dtype: " << dtype << std::endl;
        return -1;
    }

    // Forward
    int ret = m_encNetwork->forward();
    assert(ret == 0);

    // Retrieve 3 outputs from the ENCODER network
    bm_shape_t shape;
    ret = retrieveOutput(m_embedIdx, imageEmbed, shape, /*useEncoder=*/true);
    assert(ret == 0);
    ret = retrieveOutput(m_highRes0Idx, highRes0, shape, /*useEncoder=*/true);
    assert(ret == 0);
    ret = retrieveOutput(m_highRes1Idx, highRes1, shape, /*useEncoder=*/true);
    assert(ret == 0);

    return 0;
}

// ---------------------------------------------------------------------------
// runDecoder — fill 7 decoder inputs, forward, retrieve masks + iou
//   pointCoords: 4 floats [x1,y1,x2,y2] (for point mode, [x,y,0,0])
//   pointLabels: 2 floats [l1,l2]        (for point mode, [1,-1])
// ---------------------------------------------------------------------------
int Sam2::runDecoder(const std::vector<float>& imageEmbed,
                     const std::vector<float>& highRes0,
                     const std::vector<float>& highRes1,
                     float pointCoords[4], float pointLabels[2],
                     int origH, int origW,
                     std::vector<Sam2Mask>& masks) {
    bm_handle_t hd = m_decContext->handle();

    // Helper lambda to copy a float vector to a decoder input device tensor
    auto copyToDevice = [&](int inputIdx, const std::vector<float>& data) {
        auto dtype = m_decInputs[inputIdx]->get_dtype();
        if (dtype == BM_FLOAT32) {
            bm_memcpy_s2d(hd, m_decInputDevs[inputIdx].device_mem,
                          (void*)data.data());
        } else if (dtype == BM_FLOAT16) {
            std::vector<uint16_t> h16(data.size());
            for (size_t i = 0; i < data.size(); i++)
                h16[i] = fp32_to_fp16(data[i]);
            bm_memcpy_s2d(hd, m_decInputDevs[inputIdx].device_mem, (void*)h16.data());
        }
    };

    // --- Input 0: image_embed [1,256,64,64] ---
    copyToDevice(0, imageEmbed);
    // --- Input 1: high_res_feats_0 [1,32,256,256] ---
    copyToDevice(1, highRes0);
    // --- Input 2: high_res_feats_1 [1,64,128,128] ---
    copyToDevice(2, highRes1);

    // --- Input 3: point_coords [1,2,2] ---
    // SAM2 normalizes coords: x /= origW * 1024, y /= origH * 1024
    float coordData[4] = {
        pointCoords[0], pointCoords[1],
        pointCoords[2], pointCoords[3]
    };
    auto coordDtype = m_decInputs[3]->get_dtype();
    if (coordDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[3].device_mem, (void*)coordData);
    } else if (coordDtype == BM_FLOAT16) {
        uint16_t h16[4];
        for (int i = 0; i < 4; i++) h16[i] = fp32_to_fp16(coordData[i]);
        bm_memcpy_s2d(hd, m_decInputDevs[3].device_mem, (void*)h16);
    }

    // --- Input 4: point_labels [1,2] ---
    auto labelDtype = m_decInputs[4]->get_dtype();
    if (labelDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[4].device_mem, (void*)pointLabels);
    } else if (labelDtype == BM_FLOAT16) {
        uint16_t h16[2];
        for (int i = 0; i < 2; i++) h16[i] = fp32_to_fp16(pointLabels[i]);
        bm_memcpy_s2d(hd, m_decInputDevs[4].device_mem, (void*)h16);
    }

    // --- Input 5: mask_input [1,1,256,256] = zeros ---
    auto maskDtype = m_decInputs[5]->get_dtype();
    auto maskShape = m_decInputs[5]->get_shape();
    size_t maskCount = bmrt_shape_count(maskShape);
    if (maskDtype == BM_FLOAT32) {
        std::vector<float> zeros(maskCount, 0.0f);
        bm_memcpy_s2d(hd, m_decInputDevs[5].device_mem, (void*)zeros.data());
    } else if (maskDtype == BM_FLOAT16) {
        std::vector<uint16_t> zeros(maskCount, 0);
        bm_memcpy_s2d(hd, m_decInputDevs[5].device_mem, (void*)zeros.data());
    }

    // --- Input 6: has_mask_input [1] = 0 ---
    float hasMaskVal = 0.0f;
    auto hmDtype = m_decInputs[6]->get_dtype();
    if (hmDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[6].device_mem, (void*)&hasMaskVal);
    } else if (hmDtype == BM_FLOAT16) {
        uint16_t h = fp32_to_fp16(0.0f);
        bm_memcpy_s2d(hd, m_decInputDevs[6].device_mem, (void*)&h);
    }

    // --- Forward ---
    int ret = m_decNetwork->forward();
    assert(ret == 0);

    // --- Retrieve masks output [1,3,256,256] ---
    std::vector<float> masksBuf;
    bm_shape_t masksShape;
    ret = retrieveOutput(m_masksIdx, masksBuf, masksShape, /*useEncoder=*/false);
    assert(ret == 0);
    int nMasks = masksShape.dims[1];
    int mh = masksShape.dims[2];
    int mw = masksShape.dims[3];

    // --- Retrieve iou scores [1,3] ---
    std::vector<float> iouBuf;
    bm_shape_t iouShape;
    ret = retrieveOutput(m_iouIdx, iouBuf, iouShape, /*useEncoder=*/false);
    assert(ret == 0);

    // --- Postprocess: resize each mask to (origW, origH), threshold > 0.0 ---
    masks.clear();
    for (int i = 0; i < nMasks; i++) {
        cv::Mat singleMask(mh, mw, CV_32FC1);
        for (int y = 0; y < mh; y++) {
            for (int x = 0; x < mw; x++) {
                singleMask.at<float>(y, x) =
                    masksBuf[(size_t)i * mh * mw + y * mw + x];
            }
        }
        cv::Mat upscaled;
        cv::resize(singleMask, upscaled, cv::Size(origW, origH), 0, 0,
                   cv::INTER_LINEAR);
        cv::Mat binary(origH, origW, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < origH; y++) {
            const float* row = upscaled.ptr<float>(y);
            uint8_t* outRow = binary.ptr<uint8_t>(y);
            for (int x = 0; x < origW; x++) {
                outRow[x] = (row[x] > 0.0f) ? 255 : 0;
            }
        }

        Sam2Mask sm;
        sm.mask = binary;
        sm.score = (i < (int)iouBuf.size()) ? iouBuf[i] : 0.0f;
        masks.push_back(sm);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// retrieveOutput — copy an output tensor to host as F32 (handles F32/F16)
//   useEncoder=true  → read from the encoder network's output
//   useEncoder=false → read from the decoder network's output
// ---------------------------------------------------------------------------
int Sam2::retrieveOutput(int outputIdx, std::vector<float>& out,
                         bm_shape_t& outShape, bool useEncoder) {
    auto network = useEncoder ? m_encNetwork : m_decNetwork;
    auto tensor = network->outputTensor(outputIdx);
    outShape = *tensor->get_shape();
    int count = bmrt_shape_count(&outShape);
    auto dtype = tensor->get_dtype();
    const bm_device_mem_t* dmem = tensor->get_device_mem();

    out.resize(count);
    bm_handle_t h = (useEncoder ? m_encContext : m_decContext)->handle();

    if (dtype == BM_FLOAT32) {
        bm_memcpy_d2s_partial(h, out.data(), *dmem, count * sizeof(float));
    } else if (dtype == BM_FLOAT16) {
        std::vector<uint16_t> h16(count);
        bm_memcpy_d2s_partial(h, h16.data(), *dmem, count * sizeof(uint16_t));
        for (int i = 0; i < count; i++) out[i] = fp16_to_fp32(h16[i]);
    } else {
        std::cerr << "[SAM2] unsupported output dtype: " << dtype << std::endl;
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// processPoint — point prompt
//   Matches Python: point_coords normalized by origW/origH to 1024.
//   point_coords=[[[x,y],[0,0]]], point_labels=[1,-1]
// ---------------------------------------------------------------------------
int Sam2::processPoint(const cv::Mat& bgrImage, int px, int py,
                       std::vector<Sam2Mask>& masks) {
    int origH = bgrImage.rows, origW = bgrImage.cols;

    LOG_TS(m_ts, "sam2 encoding");
    std::vector<float> imageEmbed, highRes0, highRes1;
    int ret = encodeImage(bgrImage, origH, origW, imageEmbed, highRes0, highRes1);
    if (ret != 0) return ret;
    LOG_TS(m_ts, "sam2 encoding");

    // SAM2 normalizes coords: x /= origW * 1024, y /= origH * 1024
    float pointCoords[4] = {
        (float)(px) / origW * IMG_SIZE,
        (float)(py) / origH * IMG_SIZE,
        0.0f, 0.0f
    };
    float pointLabels[2] = {1.0f, -1.0f};

    LOG_TS(m_ts, "sam2 decode");
    ret = runDecoder(imageEmbed, highRes0, highRes1,
                     pointCoords, pointLabels, origH, origW, masks);
    LOG_TS(m_ts, "sam2 decode");

    return ret;
}

// ---------------------------------------------------------------------------
// processBox — box prompt
//   Matches Python: point_coords = [[x1,y1],[x2,y2]], point_labels = [2,3]
// ---------------------------------------------------------------------------
int Sam2::processBox(const cv::Mat& bgrImage, int x1, int y1, int x2, int y2,
                     std::vector<Sam2Mask>& masks) {
    int origH = bgrImage.rows, origW = bgrImage.cols;

    LOG_TS(m_ts, "sam2 encoding");
    std::vector<float> imageEmbed, highRes0, highRes1;
    int ret = encodeImage(bgrImage, origH, origW, imageEmbed, highRes0, highRes1);
    if (ret != 0) return ret;
    LOG_TS(m_ts, "sam2 encoding");

    float pointCoords[4] = {
        (float)(x1) / origW * IMG_SIZE,
        (float)(y1) / origH * IMG_SIZE,
        (float)(x2) / origW * IMG_SIZE,
        (float)(y2) / origH * IMG_SIZE
    };
    float pointLabels[2] = {2.0f, 3.0f};

    LOG_TS(m_ts, "sam2 decode");
    ret = runDecoder(imageEmbed, highRes0, highRes1,
                     pointCoords, pointLabels, origH, origW, masks);
    LOG_TS(m_ts, "sam2 decode");

    return ret;
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
void Sam2::drawMask(cv::Mat& frame, const cv::Mat& mask,
                    const cv::Scalar& color, double alpha) {
    cv::Mat overlay = frame.clone();
    overlay.setTo(color, mask);
    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);
}

void Sam2::drawPoint(cv::Mat& frame, int x, int y,
                     const cv::Scalar& color) {
    cv::circle(frame, cv::Point(x, y), 10, color, -1, cv::LINE_AA);
}

void Sam2::drawBox(cv::Mat& frame, int x1, int y1, int x2, int y2,
                   const cv::Scalar& color) {
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
}
