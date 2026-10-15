//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef BLIP_HPP
#define BLIP_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"
#include "bmnn_utils.h"

// Tokenization (Bert WordPiece) is done in main.cpp, which includes
// tokenizer/bert_tokenizer.hpp; the model classes below take/return raw
// token-id vectors to keep bert_tokenizer.hpp (which has non-inline
// definitions) included in a single translation unit.

// CLIP-style normalization constants (shared by all BLIP sub-tasks).
struct BlipMeanStd {
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
};

// Shared image preprocessing:
//   BGR image → RGB → resize (BICUBIC) to imageSize×imageSize →
//   (x/255 - mean) / std → HWC→CHW, written into out[3*imageSize*imageSize].
void blip_preprocess(const cv::Mat& bgr, int imageSize, float* out);

// ---------------------------------------------------------------------------
// Image-Text Matching (blip_itm): 1 bmodel.
//   inputs : pixel_values [1,3,384,384], input_ids [1,35], attention_mask [1,35]
//   output : 2-class ITM logits [1,2]  → softmax(dim=1)[:,1] = probability
// ---------------------------------------------------------------------------
class BlipItm {
    BMNNHandlePtr            m_handlePtr;
    std::shared_ptr<BMNNContext>  m_ctx;
    std::shared_ptr<BMNNNetwork> m_net;
    bm_handle_t             m_handle;
    void*                   m_bmrt;
    // Pre-allocated input device memory (reused across calls).
    bm_tensor_t             m_inDevs[3];
    bool                    m_alloced = false;
    int                     m_devId;
    int                     m_imageSize = 384;
    int                     m_maxLen    = 35;
    TimeStamp*              m_ts = nullptr;
public:
    BlipItm(BMNNHandlePtr handle, const std::string& bmodel, int devId);
    ~BlipItm();
    int  Init();
    void enableProfile(TimeStamp* ts) { m_ts = ts; }
    // For each (text) entry, returns softmax[:,1] matching probability.
    //   textIds : pre-tokenized input_ids [maxLen], already [CLS]...[SEP] padded.
    //   textMasks: attention_mask [maxLen], 1 where id != 0 (INT32 on BLIP bmodels).
    int  predict(const cv::Mat& bgr, const std::vector<std::vector<int>>& textIds,
                 const std::vector<std::vector<int>>& textMasks,
                 std::vector<float>& scores);
};

// ---------------------------------------------------------------------------
// Image Captioning (blip_cap): 1 bmodel.
//   inputs : pixel_values [1,3,384,384]
//   output : generated token ids [1,20]  → detokenize, strip "a picture of "
// ---------------------------------------------------------------------------
class BlipCap {
    BMNNHandlePtr            m_handlePtr;
    std::shared_ptr<BMNNContext>  m_ctx;
    std::shared_ptr<BMNNNetwork> m_net;
    bm_handle_t             m_handle;
    void*                   m_bmrt;
    bm_tensor_t             m_inDevs[1];
    bool                    m_alloced = false;
    int                     m_devId;
    int                     m_imageSize = 384;
    TimeStamp*              m_ts = nullptr;
public:
    BlipCap(BMNNHandlePtr handle, const std::string& bmodel, int devId);
    ~BlipCap();
    int  Init();
    void enableProfile(TimeStamp* ts) { m_ts = ts; }
    // Returns the generated caption token ids [N] (caller detokenizes).
    int  predict(const cv::Mat& bgr, std::vector<int>& outIds);
};

// ---------------------------------------------------------------------------
// Visual Question Answering (blip_vqa): 3 chained bmodels.
//   venc: pixel_values [1,3,480,480]                 → image_embeds [1,901,768]
//   tenc: image_embeds + input_ids[1,35] + mask[1,35] → question_states [1,35,768]
//   tdec: question_states [1,35,768]                 → answer token ids [1,10]
//   The text encoder expects the [ENC] token id (= vocab_size + 1) at input_ids[0].
// ---------------------------------------------------------------------------
class BlipVqa {
    BMNNHandlePtr            m_handlePtr;
    std::shared_ptr<BMNNContext>  m_vencCtx, m_tencCtx, m_tdecCtx;
    std::shared_ptr<BMNNNetwork> m_vencNet, m_tencNet, m_tdecNet;
    bm_handle_t             m_handle;
    int                     m_devId;
    int                     m_imageSize   = 480;
    int                     m_maxLen      = 35;
    int                     m_encTokenId  = -1;   // [ENC] = vocab_size + 1
    TimeStamp*              m_ts = nullptr;
    // Pre-allocated input device memory (reused across calls).
    bm_tensor_t             m_vencInDev;          // pixel_values [1,3,480,480]
    bm_tensor_t             m_tencInDevs[3];      // image_embeds, input_ids, attention_mask
    bm_tensor_t             m_tdecInDev;          // question_states [1,35,768]
    bool                    m_alloced = false;
    // venc output (host), reused across questions.
    std::vector<float>      m_imageEmbeds;
    int                     m_embedCount = 0;
public:
    BlipVqa(BMNNHandlePtr handle, const std::string& vencBmodel,
            const std::string& tencBmodel, const std::string& tdecBmodel, int devId);
    ~BlipVqa();
    int  Init(int encTokenId);
    void enableProfile(TimeStamp* ts) { m_ts = ts; }
    // Encode the image once; must be called before predict().
    int  imageEncode(const cv::Mat& bgr);
    // Answer one question (imageEncode must have been called).
    //   inputIds : pre-tokenized [maxLen], with inputIds[0] = [ENC] token id.
    //   attnMask : attention_mask [maxLen], 1 where id != 0 (INT32 on BLIP bmodels).
    //   outIds   : generated answer token ids.
    int  predict(const std::vector<int>& inputIds,
                 const std::vector<int>& attnMask,
                 std::vector<int>& outIds);
};

#endif // BLIP_HPP
