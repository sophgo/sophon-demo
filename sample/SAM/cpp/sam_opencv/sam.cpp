//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "sam.hpp"
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
Sam::Sam(std::shared_ptr<BMNNHandle> handle,
         const std::string& embedBmodel,
         const std::string& decodeBmodel,
         int devId)
    : m_devId(devId) {
    m_embedContext = std::make_shared<BMNNContext>(handle, embedBmodel.c_str());
    m_decodeContext = std::make_shared<BMNNContext>(handle, decodeBmodel.c_str());
}

Sam::~Sam() {
    if (m_inputBuf) delete[] m_inputBuf;
    if (m_tensorsAllocated) {
        bm_handle_t h = m_embedContext->handle();
        bm_free_device(h, m_embedInputDev.device_mem);
        bm_handle_t hd = m_decodeContext->handle();
        for (int i = 0; i < 6; i++)
            bm_free_device(hd, m_decInputDevs[i].device_mem);
    }
}

// ---------------------------------------------------------------------------
// getPreprocessShape — matches Python ResizeLongestSide.get_preprocess_shape
//   scale = 1024.0 / max(oldh, oldw)
//   newh = int(oldh*scale + 0.5), neww = int(oldw*scale + 0.5)
// ---------------------------------------------------------------------------
void Sam::getPreprocessShape(int oldh, int oldw, int longSide,
                              int& newh, int& neww) {
    double scale = (double)longSide / std::max(oldh, oldw);
    double dh = oldh * scale;
    double dw = oldw * scale;
    neww = (int)(dw + 0.5);
    newh = (int)(dh + 0.5);
}

// ---------------------------------------------------------------------------
// Init — load both networks, identify I/O, allocate tensors
// ---------------------------------------------------------------------------
int Sam::Init() {
    // --- Embedding network ---
    m_embedNetwork = m_embedContext->network(0);
    m_embedInputTensor = m_embedNetwork->inputTensor(0);
    auto embedInShape = m_embedInputTensor->get_shape();
    assert(embedInShape->num_dims == 4);
    assert(embedInShape->dims[2] == IMG_SIZE && embedInShape->dims[3] == IMG_SIZE);

    m_inputBuf = new float[3 * IMG_SIZE * IMG_SIZE];

    bmrt_tensor(&m_embedInputDev, m_embedContext->bmrt(),
                m_embedInputTensor->get_dtype(), *embedInShape);
    m_embedInputTensor->set_device_mem(&m_embedInputDev.device_mem);

    // --- Decode network ---
    m_decodeNetwork = m_decodeContext->network(0);
    int nInputs = m_decodeNetwork->m_netinfo->input_num;
    assert(nInputs == 6);
    for (int i = 0; i < 6; i++) {
        m_decInputs[i] = m_decodeNetwork->inputTensor(i);
        auto shp = m_decInputs[i]->get_shape();
        bmrt_tensor(&m_decInputDevs[i], m_decodeContext->bmrt(),
                    m_decInputs[i]->get_dtype(), *shp);
        m_decInputs[i]->set_device_mem(&m_decInputDevs[i].device_mem);
    }

    // Identify output indices by name (DO NOT rely on index order = items() order)
    // bmodel declared order: [0]=iou_predictions, [1]=low_res_masks,
    //   [2]=/Slice_9 (masks at 1024 long-side), [3]=/Concat (orig size etc)
    int nOutputs = m_decodeNetwork->outputTensorNum();
    for (int i = 0; i < nOutputs; i++) {
        std::string name = m_decodeNetwork->m_netinfo->output_names[i];
        // iou_predictions output: name contains "iou_predictions"
        if (name.find("iou_predictions") != std::string::npos)
            m_iouIdx = i;
        // masks output: name contains "Slice_9" (the upscaled-to-1024 masks)
        if (name.find("Slice") != std::string::npos)
            m_masksIdx = i;
    }
    // Fallback: if names didn't match, use heuristics by shape
    if (m_iouIdx < 0 || m_masksIdx < 0) {
        for (int i = 0; i < nOutputs; i++) {
            auto t = m_decodeNetwork->outputTensor(i);
            auto s = t->get_shape();
            // iou is [1,4] (2 dims), masks is [1,4,*,*] (4 dims)
            if (m_iouIdx < 0 && s->num_dims == 2 && s->dims[1] == 4)
                m_iouIdx = i;
            else if (m_masksIdx < 0 && s->num_dims == 4)
                m_masksIdx = i;
        }
    }
    std::cout << "[SAM] iouIdx=" << m_iouIdx << " masksIdx=" << m_masksIdx << std::endl;
    assert(m_iouIdx >= 0 && m_masksIdx >= 0);

    m_tensorsAllocated = true;
    return 0;
}

// ---------------------------------------------------------------------------
// embedImage — preprocess (resize + normalize + pad) + embedding forward
//   BGR input → RGB → resize to (newh,neww) → normalize → pad to 1024×1024
//   Then run embedding bmodel, retrieve [1,256,64,64] output.
// ---------------------------------------------------------------------------
int Sam::embedImage(const cv::Mat& bgrImage, int origH, int origW,
                     int& newH, int& newW,
                     std::vector<float>& embedding) {
    // 1. ResizeLongestSide: resize to longest side = 1024
    getPreprocessShape(origH, origW, IMG_SIZE, newH, newW);

    // BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(bgrImage, rgb, cv::COLOR_BGR2RGB);

    // PIL bilinear resize (PIL uses BILINEAR with antialias-like resampling).
    // cv::INTER_LINEAR is a close match and is what sam_opencv.py effectively
    // compares against in practice (the bmodel embedding is FP16, so small
    // differences in resize are within model precision).
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // 2. HWC → CHW, normalize: (x - mean) / std, then pad to 1024×1024 with 0
    // m_inputBuf is [3 * 1024 * 1024]
    int C = 3;
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
                cv::Vec3b px = resized.at<cv::Vec3b>(y, x);
                float val = (float)px[c];
                m_inputBuf[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] =
                    (val - m_pixelMean[c]) / m_pixelStd[c];
            }
        }
        // padding region (bottom and right) stays 0 (already zeroed? not necessarily)
        for (int y = newH; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                m_inputBuf[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = 0.0f;
            }
        }
        for (int y = 0; y < newH; y++) {
            for (int x = newW; x < IMG_SIZE; x++) {
                m_inputBuf[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] = 0.0f;
            }
        }
    }

    // 3. Copy to device
    auto dtype = m_embedInputTensor->get_dtype();
    if (dtype == BM_FLOAT32) {
        bm_memcpy_s2d(m_embedContext->handle(), m_embedInputDev.device_mem,
                      (void*)m_inputBuf);
    } else if (dtype == BM_FLOAT16) {
        size_t count = (size_t)3 * IMG_SIZE * IMG_SIZE;
        std::vector<uint16_t> h16(count);
        for (size_t i = 0; i < count; i++)
            h16[i] = fp32_to_fp16(m_inputBuf[i]);
        bm_memcpy_s2d(m_embedContext->handle(), m_embedInputDev.device_mem,
                      (void*)h16.data());
    } else {
        std::cerr << "[SAM] unsupported embedding input dtype: " << dtype << std::endl;
        return -1;
    }

    // 4. Forward
    int ret = m_embedNetwork->forward();
    assert(ret == 0);

    // 5. Retrieve output [1,256,64,64] from the EMBEDDING network (not decode)
    bm_shape_t outShape;
    return retrieveOutput(0, embedding, outShape, /*useEmbedNetwork=*/true);
}

// ---------------------------------------------------------------------------
// runDecoder — fill 6 decoder inputs, forward, retrieve masks + iou
//   pointCoords: 4 floats [x1,y1,x2,y2] (for point mode, [x,y,0,0])
//   pointLabels: 2 floats [l1,l2]        (for point mode, [1,-1])
// ---------------------------------------------------------------------------
int Sam::runDecoder(const std::vector<float>& embedding,
                     float pointCoords[4], float pointLabels[2],
                     int origH, int origW, int newH, int newW,
                     std::vector<SamMask>& masks) {
    bm_handle_t hd = m_decodeContext->handle();

    // --- Input 0: image_embeddings [1,256,64,64] ---
    auto embShape = m_decInputs[0]->get_shape();
    auto embDtype = m_decInputs[0]->get_dtype();
    if (embDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[0].device_mem,
                      (void*)embedding.data());
    } else if (embDtype == BM_FLOAT16) {
        size_t cnt = embedding.size();
        std::vector<uint16_t> h16(cnt);
        for (size_t i = 0; i < cnt; i++)
            h16[i] = fp32_to_fp16(embedding[i]);
        bm_memcpy_s2d(hd, m_decInputDevs[0].device_mem, (void*)h16.data());
    } else {
        std::cerr << "[SAM] unsupported embedding dtype: " << embDtype << std::endl;
        return -1;
    }

    // --- Input 1: point_coords [1,2,2] ---
    float coordData[4] = {pointCoords[0], pointCoords[1],
                           pointCoords[2], pointCoords[3]};
    auto coordDtype = m_decInputs[1]->get_dtype();
    if (coordDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[1].device_mem, (void*)coordData);
    } else if (coordDtype == BM_FLOAT16) {
        uint16_t h16[4];
        for (int i = 0; i < 4; i++) h16[i] = fp32_to_fp16(coordData[i]);
        bm_memcpy_s2d(hd, m_decInputDevs[1].device_mem, (void*)h16);
    }

    // --- Input 2: point_labels [1,2] ---
    auto labelDtype = m_decInputs[2]->get_dtype();
    if (labelDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[2].device_mem, (void*)pointLabels);
    } else if (labelDtype == BM_FLOAT16) {
        uint16_t h16[2];
        for (int i = 0; i < 2; i++) h16[i] = fp32_to_fp16(pointLabels[i]);
        bm_memcpy_s2d(hd, m_decInputDevs[2].device_mem, (void*)h16);
    }

    // --- Input 3: mask_input [1,1,256,256] = zeros ---
    auto maskDtype = m_decInputs[3]->get_dtype();
    auto maskShape = m_decInputs[3]->get_shape();
    size_t maskCount = bmrt_shape_count(maskShape);
    if (maskDtype == BM_FLOAT32) {
        std::vector<float> zeros(maskCount, 0.0f);
        bm_memcpy_s2d(hd, m_decInputDevs[3].device_mem, (void*)zeros.data());
    } else if (maskDtype == BM_FLOAT16) {
        std::vector<uint16_t> zeros(maskCount, 0);
        bm_memcpy_s2d(hd, m_decInputDevs[3].device_mem, (void*)zeros.data());
    }

    // --- Input 4: has_mask_input [1] = 0 ---
    float hasMaskVal = 0.0f;
    auto hmDtype = m_decInputs[4]->get_dtype();
    if (hmDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[4].device_mem, (void*)&hasMaskVal);
    } else if (hmDtype == BM_FLOAT16) {
        uint16_t h = fp32_to_fp16(0.0f);
        bm_memcpy_s2d(hd, m_decInputDevs[4].device_mem, (void*)&h);
    }

    // --- Input 5: orig_im_size [2] = [origH, origW] ---
    float origSizeData[2] = {(float)origH, (float)origW};
    auto osDtype = m_decInputs[5]->get_dtype();
    if (osDtype == BM_FLOAT32) {
        bm_memcpy_s2d(hd, m_decInputDevs[5].device_mem, (void*)origSizeData);
    } else if (osDtype == BM_FLOAT16) {
        uint16_t h16[2];
        for (int i = 0; i < 2; i++) h16[i] = fp32_to_fp16(origSizeData[i]);
        bm_memcpy_s2d(hd, m_decInputDevs[5].device_mem, (void*)h16);
    }

    // --- Forward ---
    int ret = m_decodeNetwork->forward();
    assert(ret == 0);

    // --- Retrieve masks output [1,4,H_m,1024] ---
    std::vector<float> masksBuf;
    bm_shape_t masksShape;
    ret = retrieveOutput(m_masksIdx, masksBuf, masksShape);
    assert(ret == 0);
    // masksShape: [1,4,H_m,W_m] where H_m = orig-resized, W_m = 1024
    int nMasks = masksShape.dims[1];
    int mh = masksShape.dims[2];
    int mw = masksShape.dims[3];

    // --- Retrieve iou scores [1,4] ---
    std::vector<float> iouBuf;
    bm_shape_t iouShape;
    ret = retrieveOutput(m_iouIdx, iouBuf, iouShape);
    assert(ret == 0);

    // --- Postprocess: resize each mask to (origW, origH), threshold > 0.0 ---
    masks.clear();
    for (int i = 0; i < nMasks; i++) {
        // Extract single mask [mh, mw]
        cv::Mat singleMask(mh, mw, CV_32FC1);
        for (int y = 0; y < mh; y++) {
            for (int x = 0; x < mw; x++) {
                singleMask.at<float>(y, x) =
                    masksBuf[(size_t)i * mh * mw + y * mw + x];
            }
        }
        // Resize to original image size (origW, origH)
        cv::Mat upscaled;
        cv::resize(singleMask, upscaled, cv::Size(origW, origH), 0, 0,
                   cv::INTER_LINEAR);
        // Threshold > 0.0
        cv::Mat binary(origH, origW, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < origH; y++) {
            const float* row = upscaled.ptr<float>(y);
            uint8_t* outRow = binary.ptr<uint8_t>(y);
            for (int x = 0; x < origW; x++) {
                outRow[x] = (row[x] > 0.0f) ? 255 : 0;
            }
        }

        SamMask sm;
        sm.mask = binary;
        sm.score = (i < (int)iouBuf.size()) ? iouBuf[i] : 0.0f;
        masks.push_back(sm);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// retrieveOutput — copy an output tensor to host as F32 (handles F32/F16)
//   useEmbedNetwork=true  → read from the embedding network's output
//   useEmbedNetwork=false → read from the decode network's output
// ---------------------------------------------------------------------------
int Sam::retrieveOutput(int outputIdx, std::vector<float>& out,
                         bm_shape_t& outShape, bool useEmbedNetwork) {
    auto network = useEmbedNetwork ? m_embedNetwork : m_decodeNetwork;
    auto tensor = network->outputTensor(outputIdx);
    outShape = *tensor->get_shape();
    int count = bmrt_shape_count(&outShape);
    auto dtype = tensor->get_dtype();
    const bm_device_mem_t* dmem = tensor->get_device_mem();

    out.resize(count);
    bm_handle_t h = (useEmbedNetwork ? m_embedContext : m_decodeContext)->handle();

    if (dtype == BM_FLOAT32) {
        bm_memcpy_d2s_partial(h, out.data(), *dmem, count * sizeof(float));
    } else if (dtype == BM_FLOAT16) {
        std::vector<uint16_t> h16(count);
        bm_memcpy_d2s_partial(h, h16.data(), *dmem, count * sizeof(uint16_t));
        for (int i = 0; i < count; i++) out[i] = fp16_to_fp32(h16[i]);
    } else {
        std::cerr << "[SAM] unsupported output dtype: " << dtype << std::endl;
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// processPoint — point prompt
//   Matches Python: ori_coord = [[x,y],[0,0]], ori_label = [1,-1]
//   then apply_coords: x *= new_w/old_w, y *= new_h/old_h
// ---------------------------------------------------------------------------
int Sam::processPoint(const cv::Mat& bgrImage, int px, int py,
                       std::vector<SamMask>& masks) {
    int origH = bgrImage.rows, origW = bgrImage.cols;

    LOG_TS(m_ts, "sam embedding");
    int newH, newW;
    std::vector<float> embedding;
    int ret = embedImage(bgrImage, origH, origW, newH, newW, embedding);
    if (ret != 0) return ret;
    LOG_TS(m_ts, "sam embedding");

    // Transform coords: x *= new_w/old_w, y *= new_h/old_h
    double sx = (double)newW / origW;
    double sy = (double)newH / origH;
    float pointCoords[4] = {
        (float)(px * sx), (float)(py * sy),
        0.0f, 0.0f
    };
    float pointLabels[2] = {1.0f, -1.0f};

    LOG_TS(m_ts, "sam decode");
    ret = runDecoder(embedding, pointCoords, pointLabels,
                     origH, origW, newH, newW, masks);
    LOG_TS(m_ts, "sam decode");

    return ret;
}

// ---------------------------------------------------------------------------
// processBox — box prompt
//   Matches Python: input_point = reshape(2,2), input_label = [2,3]
//   point_coords = [[x1,y1],[x2,y2]], point_labels = [2,3]
// ---------------------------------------------------------------------------
int Sam::processBox(const cv::Mat& bgrImage, int x1, int y1, int x2, int y2,
                     std::vector<SamMask>& masks) {
    int origH = bgrImage.rows, origW = bgrImage.cols;

    LOG_TS(m_ts, "sam embedding");
    int newH, newW;
    std::vector<float> embedding;
    int ret = embedImage(bgrImage, origH, origW, newH, newW, embedding);
    if (ret != 0) return ret;
    LOG_TS(m_ts, "sam embedding");

    double sx = (double)newW / origW;
    double sy = (double)newH / origH;
    float pointCoords[4] = {
        (float)(x1 * sx), (float)(y1 * sy),
        (float)(x2 * sx), (float)(y2 * sy)
    };
    float pointLabels[2] = {2.0f, 3.0f};

    LOG_TS(m_ts, "sam decode");
    ret = runDecoder(embedding, pointCoords, pointLabels,
                     origH, origW, newH, newW, masks);
    LOG_TS(m_ts, "sam decode");

    return ret;
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
void Sam::drawMask(cv::Mat& frame, const cv::Mat& mask,
                    const cv::Scalar& color, double alpha) {
    cv::Mat overlay = frame.clone();
    overlay.setTo(color, mask);
    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);
}

void Sam::drawPoint(cv::Mat& frame, int x, int y,
                     const cv::Scalar& color) {
    cv::circle(frame, cv::Point(x, y), 10, color, -1, cv::LINE_AA);
}

void Sam::drawBox(cv::Mat& frame, int x1, int y1, int x2, int y2,
                   const cv::Scalar& color) {
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
}
