//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef SAM3_HPP
#define SAM3_HPP

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bmnn_utils.h"
#include "npz_loader.hpp"
#include "tokenizer/sam3_tokenizer.hpp"
#include "utils.hpp"

// SAM3 504 F16 text-prompted detection + segmentation, C++ port of
// python/sam3_infer.py. Pipeline:
//
//   preprocess (504x504, x/127.5-1)
//   -> ViT part0..part4 (5 bmodels, [1,36,36,1024] chain)
//   -> transpose to [1,1024,36,36] -> neck (fpn s4/s2/s1)
//   -> per prompt: text encoder (BPE ids -> [32,1,256])
//      -> grounding encoder/decoder (TPU)
//      -> scoring + box refinement (CPU, post_process_weights.npz)
//      -> mask decoder (CPU, seg_head_weights.npz): cross-attend + pixel
//         decoder convs + instance head + mask embed, best query only
//
// Device memory: the 4 big ViT parts hold ~450MB coefficients each. With
// streaming=true they are loaded/freed per frame (peak ~1.4G, fits the
// default SE7 npu heap); with streaming=false everything stays resident
// (~2.8G, needs an enlarged heap, faster per frame).
struct Sam3Result {
  std::string prompt;
  float score = 0.f;      // best query score (sigmoid)
  int best_idx = -1;      // best query index (of 200)
  float box_norm[4];      // cx, cy, w, h (normalized to input image)
  int box_xyxy[4];        // pixels
  cv::Mat mask;           // CV_8UC1, input image size (best query, logit > 0)
  int mask_pixels = 0;
};

class Sam3 {
 public:
  Sam3(std::shared_ptr<BMNNHandle> handle, const std::string& modelDir,
       const std::string& bpePath, const std::string& postNpzPath,
       const std::string& segNpzPath, int devId);
  ~Sam3();

  int Init(bool streaming);

  void enableProfile(TimeStamp* ts) { m_ts = ts; }

  // Run the full pipeline for one image and a list of text prompts.
  int predict(const cv::Mat& bgr, const std::vector<std::string>& prompts,
              float scoreThresh, std::vector<Sam3Result>& results);

  // Draw green box + mask overlay + label for every result (python parity).
  static void drawResults(cv::Mat& image, const std::vector<Sam3Result>& results);

 private:
  struct PartCtx {  // one ViT part bmodel context + its input device tensor
    std::shared_ptr<BMNNContext> ctx;
    std::shared_ptr<BMNNNetwork> net;
    bm_tensor_t in_dev{};
    bool has_dev = false;
  };

  // --- stage helpers ---
  int runVit(const float* input);           // [1,3,504,504] -> m_vitFeat [1024,36,36]
  int runNeck(const float* vitFeat);        // -> m_featS4/S2/S1
  int runText(const std::vector<int32_t>& ids);  // -> m_textFeat [32*256]
  int runGrounding();                       // -> m_hs [6,200,256], m_refBox, m_memory
  int postProcess(const std::string& prompt, Sam3Result& r);  // last layer only
  int predictMask(Sam3Result& r);           // mask decoder, best query only

  // --- small math helpers ---
  void computeSinePos();                    // m_s1Pos [256,36,36] channel-major
  static void groupNorm8(float* x, const float* w, const float* b, int C,
                         int H, int W);
  void im2col3x3(const float* x, int C, int H, int W);
  void conv3x3Gemm(int stage, const float* x, float* y, int H, int W);
  // Y[M,N] = A[M,K] @ B[K,N], row-major. A is transposed into scratch first
  // so the hot loop streams contiguous memory; threads split N. nthreads=1
  // runs single-threaded (callers with per-token work parallelize above this).
  static void gemm(std::vector<float>& aT, const float* A, const float* B,
                   float* Y, int M, int N, int K, int nthreads);

  // weights (post_process_weights.npz / seg_head_weights.npz)
  NpzMap m_postW, m_segW;
  const float* m_promptProjW = nullptr;   // [256,256]
  const float* m_promptProjB = nullptr;   // [256]
  const float* m_hsProjW = nullptr;       // [256,256]
  const float* m_hsProjB = nullptr;       // [256]
  double m_dProj = 256;
  const float* m_bboxW[3] = {nullptr, nullptr, nullptr};
  const float* m_bboxB[3] = {nullptr, nullptr, nullptr};

  const float* m_convW[3] = {nullptr, nullptr, nullptr};    // [256,256,3,3]
  const float* m_convB[3] = {nullptr, nullptr, nullptr};
  const float* m_normW[3] = {nullptr, nullptr, nullptr};    // [256]
  const float* m_normB[3] = {nullptr, nullptr, nullptr};
  const float* m_instW = nullptr;          // [256,256]
  const float* m_instB = nullptr;          // [256]
  const float* m_meW[3] = {nullptr, nullptr, nullptr};      // [256,256] x3
  const float* m_meB[3] = {nullptr, nullptr, nullptr};
  const float* m_inProjW = nullptr;        // [768,256]
  const float* m_inProjB = nullptr;        // [768]
  const float* m_outProjW = nullptr;       // [256,256]
  const float* m_outProjB = nullptr;       // [256]

  // networks
  std::shared_ptr<BMNNHandle> m_handle;
  std::string m_modelDir;
  int m_devId;
  bool m_streaming = true;

  PartCtx m_vit0;                   // part0 (15MB, always resident)
  std::vector<PartCtx> m_vitParts;  // part1-4 (resident when !streaming)
  std::shared_ptr<BMNNContext> m_neckCtx;
  std::shared_ptr<BMNNNetwork> m_neckNet;
  std::shared_ptr<BMNNContext> m_textCtx;
  std::shared_ptr<BMNNNetwork> m_textNet;
  std::shared_ptr<BMNNContext> m_gndEncCtx;
  std::shared_ptr<BMNNNetwork> m_gndEncNet;
  std::shared_ptr<BMNNContext> m_gndDecCtx;
  std::shared_ptr<BMNNNetwork> m_gndDecNet;

  // input device tensors of the resident networks
  bm_tensor_t m_vit0In{};
  bool m_vit0InOk = false;
  bm_tensor_t m_neckIn{};
  bool m_neckInOk = false;
  bm_tensor_t m_textIn{};
  bool m_textInOk = false;
  std::vector<bm_tensor_t> m_gndEncIn;
  std::vector<bm_tensor_t> m_gndDecIn;
  // input index by role
  std::vector<int> m_gndEncIdx;  // 0=src 1=src_pos 2=prompt 3=prompt_mask
  std::vector<int> m_gndDecIdx;  // 0=memory 1=memory_pos 2=memory_mask
                                 // 3=spatial_shapes 4=valid_ratios 5=prompt
                                 // 6=prompt_mask

  // tokenizer
  std::unique_ptr<Sam3Tokenizer> m_tokenizer;

  // host buffers (reused across frames)
  std::vector<float> m_inputBuf;   // [3*504*504]
  std::vector<float> m_vitChain;   // [36*36*1024] ViT host roundtrip
  std::vector<float> m_vitFeat;    // [1024*36*36] transposed (neck input)
  std::vector<float> m_featS4;     // [256,144,144]
  std::vector<float> m_featS2;     // [256,72,72]
  std::vector<float> m_featS1;     // [256,36,36]
  std::vector<float> m_s1Pos;      // [256,36,36] sine embedding
  std::vector<float> m_textFeat;   // [32,256] (== [1,32,256] flat, B=1)
  std::vector<float> m_textMask;   // [32] 1.0 where token id == 0 (pad)
  std::vector<int32_t> m_tokenIds;  // [32]
  std::vector<float> m_memory;     // [1296,256] grounding encoder output
  std::vector<float> m_memoryPos;  // [1296,256]
  std::vector<float> m_hs;         // [6,200,256]
  std::vector<float> m_refBox;     // [6,200,4]
  std::vector<float> m_col;        // im2col [256*9, H*W]
  std::vector<float> m_xpad;       // padded conv input [256,(H+2),(W+2)]
  std::vector<float> m_convW2;     // prepacked conv weights [2][256, 256*9]
  std::vector<float> m_qW;         // transposed q proj [256c, 256d]
  std::vector<float> m_oW;         // transposed out proj [256c, 256d]
  std::vector<float> m_qproj;      // [1296,256] Q projections
  std::vector<float> m_attendOut;  // [1296,256] attention output
  std::vector<float> m_attended;   // [1296,256] cross-attended (+residual)
  std::vector<float> m_aT;         // gemm scratch: A transposed to [K,M]
  std::vector<float> m_gemmA;      // gemm scratch for member call sites
  std::vector<float> m_encVisual;  // [256,36,36]
  std::vector<float> m_pixel;      // [256,72,72] iter0 / [256,144,144] cap
  std::vector<float> m_pixel2;     // [256,144,144] iter1
  std::vector<float> m_up;         // upsample+add scratch [256,144,144]
  std::vector<float> m_instEmb;    // [256,144,144]
  int m_nThreads = 1;

  TimeStamp* m_ts = nullptr;
};

#endif  // SAM3_HPP
