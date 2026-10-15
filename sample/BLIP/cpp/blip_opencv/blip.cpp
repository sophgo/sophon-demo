//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "blip.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// fp16 ↔ fp32 conversion (bmodel outputs may be F16; BMNNTensor::get_cpu_data()
// does not handle BM_FLOAT16, so we copy raw bytes and convert ourselves).
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
// Shared image preprocessing:
//   BGR image → RGB → resize (BICUBIC) to imageSize×imageSize →
//   (x/255 - mean) / std → HWC→CHW.
//   Matches Python blip.preprocess() / torchvision BICUBIC + CLIP norm.
// ---------------------------------------------------------------------------
void blip_preprocess(const cv::Mat& bgr, int imageSize, float* out) {
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(imageSize, imageSize), 0, 0, cv::INTER_CUBIC);

    // HWC → CHW, normalize: (x/255 - mean) / std
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                cv::Vec3b px = resized.at<cv::Vec3b>(y, x);
                float val = (float)px[c] / 255.0f;
                out[c * imageSize * imageSize + y * imageSize + x] =
                    (val - mean[c]) / std[c];
            }
        }
    }
}

// Copy a float buffer (image / attention_mask / embeds) to a device tensor,
// converting to F16 if the tensor dtype is F16 (BLIP bmodels on CV84X6 use F16;
// BM1684X/BM1688 use F32).
static void copyToDevice(bm_handle_t h, bm_tensor_t& t, const float* data,
                         size_t count) {
    // dtype is read from the tensor; BLIP bmodels use F32 or F16 inputs.
    auto dtype = t.dtype;
    if (dtype == BM_FLOAT32) {
        bm_memcpy_s2d_partial(h, t.device_mem, (void*)data,
                              count * sizeof(float));
    } else if (dtype == BM_FLOAT16) {
        std::vector<uint16_t> h16(count);
        for (size_t i = 0; i < count; i++) h16[i] = fp32_to_fp16(data[i]);
        bm_memcpy_s2d_partial(h, t.device_mem, (void*)h16.data(),
                              count * sizeof(uint16_t));
    } else {
        std::cerr << "[BLIP] unsupported input dtype: " << dtype << std::endl;
    }
}

// Copy an int buffer (token ids) to a device tensor. The input_ids tensor is
// INT32 on BLIP bmodels; if a platform exports it as F32 we convert on the fly.
static void copyToDevice(bm_handle_t h, bm_tensor_t& t, const int* data,
                         size_t count) {
    auto dtype = t.dtype;
    if (dtype == BM_INT32) {
        bm_memcpy_s2d_partial(h, t.device_mem, (void*)data,
                              count * sizeof(int));
    } else if (dtype == BM_FLOAT32) {
        std::vector<float> f(count);
        for (size_t i = 0; i < count; i++) f[i] = (float)data[i];
        bm_memcpy_s2d_partial(h, t.device_mem, (void*)f.data(),
                              count * sizeof(float));
    } else {
        std::cerr << "[BLIP] unsupported input_id dtype: " << dtype << std::endl;
    }
}

// Read an output tensor to a host float vector (handles F32 / F16).
static int retrieveOutput(const std::shared_ptr<BMNNNetwork>& net,
                         bm_handle_t h, int outIdx,
                         std::vector<float>& out, bm_shape_t& outShape) {
    auto tensor = net->outputTensor(outIdx);
    outShape = *tensor->get_shape();
    int count = bmrt_shape_count(&outShape);
    auto dtype = tensor->get_dtype();
    const bm_device_mem_t* dmem = tensor->get_device_mem();

    out.resize(count);
    if (dtype == BM_FLOAT32) {
        bm_memcpy_d2s_partial(h, out.data(), *dmem, count * sizeof(float));
    } else if (dtype == BM_FLOAT16) {
        std::vector<uint16_t> h16(count);
        bm_memcpy_d2s_partial(h, h16.data(), *dmem, count * sizeof(uint16_t));
        for (int i = 0; i < count; i++) out[i] = fp16_to_fp32(h16[i]);
    } else {
        std::cerr << "[BLIP] unsupported output dtype: " << dtype << std::endl;
        return -1;
    }
    return 0;
}

// Locate the single output of a network by trying name candidates, then
// falling back to index 0. BLIP bmodels export one output whose name varies
// between "output_Gather_f32" / "output_Gather" (itm), "output_Concat_f32" /
// "output_Concat" (cap / vqa tdec), and "output_LayerNormalization_f32" /
// "output_LayerNormalization" (vqa venc / tenc).
static int findOutputIndex(const std::shared_ptr<BMNNNetwork>& net,
                           const std::vector<std::string>& candidates) {
    int n = net->outputTensorNum();
    for (int i = 0; i < n; i++) {
        std::string name = net->m_netinfo->output_names[i];
        for (auto& cand : candidates) {
            if (name.find(cand) != std::string::npos) return i;
        }
    }
    return 0;  // fallback to the first output
}

// ===========================================================================
// BlipItm
// ===========================================================================
BlipItm::BlipItm(BMNNHandlePtr handle, const std::string& bmodel, int devId)
    : m_handlePtr(handle), m_devId(devId) {
    m_ctx = std::make_shared<BMNNContext>(handle, bmodel.c_str());
    m_handle = m_ctx->handle();
    m_bmrt = m_ctx->bmrt();
}

BlipItm::~BlipItm() {
    if (m_alloced) {
        bm_free_device(m_handle, m_inDevs[0].device_mem);
        bm_free_device(m_handle, m_inDevs[1].device_mem);
        bm_free_device(m_handle, m_inDevs[2].device_mem);
    }
}

int BlipItm::Init() {
    m_net = m_ctx->network("blip_itm");
    // Allocate device memory for the 3 inputs and wire it into the tensors.
    for (int i = 0; i < 3; i++) {
        auto t = m_net->inputTensor(i);
        auto shp = t->get_shape();
        bmrt_tensor(&m_inDevs[i], m_bmrt, t->get_dtype(), *shp);
        t->set_device_mem(&m_inDevs[i].device_mem);
    }
    m_alloced = true;
    return 0;
}

int BlipItm::predict(const cv::Mat& bgr, const std::vector<std::vector<int>>& textIds,
                     const std::vector<std::vector<int>>& textMasks,
                     std::vector<float>& scores) {
    // --- Preprocess image once ---
    std::vector<float> img(3 * m_imageSize * m_imageSize);
    blip_preprocess(bgr, m_imageSize, img.data());
    copyToDevice(m_handle, m_inDevs[0], img.data(), img.size());

    scores.clear();
    scores.reserve(textIds.size());

    // --- For each text: copy ids + mask, forward, read softmax[:,1] ---
    for (size_t ti = 0; ti < textIds.size(); ti++) {
        copyToDevice(m_handle, m_inDevs[1], textIds[ti].data(), m_maxLen);
        copyToDevice(m_handle, m_inDevs[2], textMasks[ti].data(), m_maxLen);

        if (m_ts) m_ts->save("blip_itm inference");
        int ret = m_net->forward();
        if (ret != 0) return -1;
        if (m_ts) m_ts->save("blip_itm inference");

        int outIdx = findOutputIndex(
            m_net, {"output_Gather_f32", "output_Gather"});
        std::vector<float> out;
        bm_shape_t shape;
        retrieveOutput(m_net, m_handle, outIdx, out, shape);
        // bmodel output is already softmax(dim=1)[:,1] (see export_itm.py)
        scores.push_back(out.empty() ? 0.0f : out[0]);
    }
    return 0;
}

// ===========================================================================
// BlipCap
// ===========================================================================
BlipCap::BlipCap(BMNNHandlePtr handle, const std::string& bmodel, int devId)
    : m_handlePtr(handle), m_devId(devId) {
    m_ctx = std::make_shared<BMNNContext>(handle, bmodel.c_str());
    m_handle = m_ctx->handle();
    m_bmrt = m_ctx->bmrt();
}

BlipCap::~BlipCap() {
    if (m_alloced) {
        bm_free_device(m_handle, m_inDevs[0].device_mem);
    }
}

int BlipCap::Init() {
    m_net = m_ctx->network("blip_cap");
    auto t = m_net->inputTensor(0);
    auto shp = t->get_shape();
    bmrt_tensor(&m_inDevs[0], m_bmrt, t->get_dtype(), *shp);
    t->set_device_mem(&m_inDevs[0].device_mem);
    m_alloced = true;
    return 0;
}

int BlipCap::predict(const cv::Mat& bgr, std::vector<int>& outIds) {
    // --- Preprocess image ---
    std::vector<float> img(3 * m_imageSize * m_imageSize);
    blip_preprocess(bgr, m_imageSize, img.data());
    copyToDevice(m_handle, m_inDevs[0], img.data(), img.size());

    // --- Forward ---
    if (m_ts) m_ts->save("blip_cap inference");
    int ret = m_net->forward();
    if (ret != 0) return -1;
    if (m_ts) m_ts->save("blip_cap inference");

    // --- Output token ids [1,20] (caller detokenizes + strips "a picture of ") ---
    int outIdx = findOutputIndex(
        m_net, {"output_Concat_f32", "output_Concat"});
    std::vector<float> out;
    bm_shape_t shape;
    retrieveOutput(m_net, m_handle, outIdx, out, shape);
    int n = shape.num_dims >= 2 ? shape.dims[1] : (int)out.size();

    outIds.resize(n);
    for (int i = 0; i < n; i++) outIds[i] = (int)std::lround(out[i]);
    return 0;
}

// ===========================================================================
// BlipVqa
// ===========================================================================
BlipVqa::BlipVqa(BMNNHandlePtr handle, const std::string& vencBmodel,
                 const std::string& tencBmodel, const std::string& tdecBmodel,
                 int devId)
    : m_handlePtr(handle), m_devId(devId) {
    m_vencCtx = std::make_shared<BMNNContext>(handle, vencBmodel.c_str());
    m_tencCtx = std::make_shared<BMNNContext>(handle, tencBmodel.c_str());
    m_tdecCtx = std::make_shared<BMNNContext>(handle, tdecBmodel.c_str());
    m_handle = m_vencCtx->handle();
}

BlipVqa::~BlipVqa() {
    if (m_alloced) {
        bm_free_device(m_handle, m_vencInDev.device_mem);
        for (int i = 0; i < 3; i++)
            bm_free_device(m_handle, m_tencInDevs[i].device_mem);
        bm_free_device(m_handle, m_tdecInDev.device_mem);
    }
}

int BlipVqa::Init(int encTokenId) {
    // [ENC] = vocab_size + 1 (init_tokenizer adds [DEC] at vocab_size, then
    // [ENC] as the next additional_special_token).
    m_encTokenId = encTokenId;

    // --- venc: pixel_values [1,3,480,480] → image_embeds [1,901,768] ---
    m_vencNet = m_vencCtx->network("blip_vqa_venc");
    {
        auto t = m_vencNet->inputTensor(0);
        auto shp = t->get_shape();
        bmrt_tensor(&m_vencInDev, m_vencCtx->bmrt(), t->get_dtype(), *shp);
        t->set_device_mem(&m_vencInDev.device_mem);
    }

    // --- tenc: image_embeds [1,901,768], input_ids [1,35], attention_mask [1,35]
    //         → question_states [1,35,768] ---
    m_tencNet = m_tencCtx->network("blip_vqa_tenc");
    for (int i = 0; i < 3; i++) {
        auto t = m_tencNet->inputTensor(i);
        auto shp = t->get_shape();
        bmrt_tensor(&m_tencInDevs[i], m_tencCtx->bmrt(), t->get_dtype(), *shp);
        t->set_device_mem(&m_tencInDevs[i].device_mem);
    }

    // --- tdec: question_states [1,35,768] → answer token ids [1,10] ---
    m_tdecNet = m_tdecCtx->network("blip_vqa_tdec");
    {
        auto t = m_tdecNet->inputTensor(0);
        auto shp = t->get_shape();
        bmrt_tensor(&m_tdecInDev, m_tdecCtx->bmrt(), t->get_dtype(), *shp);
        t->set_device_mem(&m_tdecInDev.device_mem);
    }

    m_alloced = true;
    return 0;
}

int BlipVqa::imageEncode(const cv::Mat& bgr) {
    // --- Preprocess image (480×480) ---
    std::vector<float> img(3 * m_imageSize * m_imageSize);
    blip_preprocess(bgr, m_imageSize, img.data());
    copyToDevice(m_handle, m_vencInDev, img.data(), img.size());

    // --- venc forward ---
    if (m_ts) m_ts->save("blip_vqa venc");
    int ret = m_vencNet->forward();
    if (ret != 0) return -1;
    if (m_ts) m_ts->save("blip_vqa venc");

    // --- Retrieve image_embeds [1,901,768] ---
    int outIdx = findOutputIndex(
        m_vencNet, {"output_LayerNormalization_f32", "output_LayerNormalization"});
    bm_shape_t shape;
    retrieveOutput(m_vencNet, m_handle, outIdx, m_imageEmbeds, shape);
    m_embedCount = 1;
    for (int i = 1; i < shape.num_dims; i++) m_embedCount *= shape.dims[i];
    return 0;
}

int BlipVqa::predict(const std::vector<int>& inputIds,
                     const std::vector<int>& attnMask,
                     std::vector<int>& outIds) {
    // --- tenc: copy image_embeds + input_ids + attention_mask ---
    copyToDevice(m_handle, m_tencInDevs[0], m_imageEmbeds.data(),
                 m_imageEmbeds.size());
    copyToDevice(m_handle, m_tencInDevs[1], inputIds.data(), m_maxLen);
    copyToDevice(m_handle, m_tencInDevs[2], attnMask.data(), m_maxLen);

    if (m_ts) m_ts->save("blip_vqa tenc+tdec");
    int ret = m_tencNet->forward();
    if (ret != 0) return -1;

    // --- Retrieve question_states [1,35,768] → feed into tdec ---
    int tencOutIdx = findOutputIndex(
        m_tencNet, {"output_LayerNormalization_f32", "output_LayerNormalization"});
    std::vector<float> questionStates;
    bm_shape_t qsShape;
    retrieveOutput(m_tencNet, m_handle, tencOutIdx, questionStates, qsShape);
    copyToDevice(m_handle, m_tdecInDev, questionStates.data(),
                 questionStates.size());

    ret = m_tdecNet->forward();
    if (ret != 0) return -1;
    if (m_ts) m_ts->save("blip_vqa tenc+tdec");

    // --- Detokenize answer token ids [1,10] ---
    int tdecOutIdx = findOutputIndex(
        m_tdecNet, {"output_Concat_f32", "output_Concat"});
    std::vector<float> out;
    bm_shape_t shape;
    retrieveOutput(m_tdecNet, m_handle, tdecOutIdx, out, shape);
    int n = shape.num_dims >= 2 ? shape.dims[1] : (int)out.size();

    outIds.resize(n);
    for (int i = 0; i < n; i++) outIds[i] = (int)std::lround(out[i]);
    return 0;
}
