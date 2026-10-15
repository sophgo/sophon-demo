//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "sam3.hpp"

#include <cmath>
#include <cstring>
#include <thread>

#include <algorithm>

// 504-resolution constants.
static constexpr int kRes = 504;
static constexpr int kGrid = 36;         // 504/14
static constexpr int kToks = 1296;       // 36*36 spatial tokens
static constexpr int kQueries = 200;
static constexpr int kLayers = 6;
static constexpr int kTextLen = 32;      // text context length
static constexpr int kC = 256;           // feature channels

Sam3::Sam3(std::shared_ptr<BMNNHandle> handle, const std::string& modelDir,
           const std::string& bpePath, const std::string& postNpzPath,
           const std::string& segNpzPath, int devId)
    : m_handle(handle), m_modelDir(modelDir), m_devId(devId) {
  std::string err;
  if (!loadNpz(postNpzPath, m_postW, err)) {
    std::cout << "load " << postNpzPath << " failed: " << err << std::endl;
    exit(1);
  }
  if (!loadNpz(segNpzPath, m_segW, err)) {
    std::cout << "load " << segNpzPath << " failed: " << err << std::endl;
    exit(1);
  }
  m_tokenizer = std::make_unique<Sam3Tokenizer>(bpePath, kTextLen);
}

Sam3::~Sam3() {
  bm_handle_t h = m_handle->handle();
  if (m_vit0InOk) bm_free_device(h, m_vit0In.device_mem);
  if (m_neckInOk) bm_free_device(h, m_neckIn.device_mem);
  if (m_textInOk) bm_free_device(h, m_textIn.device_mem);
  for (auto& m : m_gndEncIn) bm_free_device(h, m.device_mem);
  for (auto& m : m_gndDecIn) bm_free_device(h, m.device_mem);
  if (m_vit0.has_dev) bm_free_device(h, m_vit0.in_dev.device_mem);
  for (auto& p : m_vitParts)
    if (p.has_dev) bm_free_device(h, p.in_dev.device_mem);
}

// Convenience: one .npy entry as flat float array + shape sanity check.
static const float* wf(const NpzMap& m, const char* key, size_t expect,
                       const char* what) {
  auto it = m.find(key);
  if (it == m.end()) {
    std::cout << "missing weight " << key << " (" << what << ")" << std::endl;
    exit(1);
  }
  if (!it->second.is_float || it->second.count() != expect) {
    std::cout << "bad weight " << key << ": count=" << it->second.count()
              << " expect=" << expect << std::endl;
    exit(1);
  }
  return it->second.data.data();
}

int Sam3::Init(bool streaming) {
  m_streaming = streaming;
  m_nThreads = std::max(1u, std::thread::hardware_concurrency());

  // --- weight pointers ---
  m_promptProjW = wf(m_postW, "prompt_proj_weight", kC * kC, "post");
  m_promptProjB = wf(m_postW, "prompt_proj_bias", kC, "post");
  m_hsProjW = wf(m_postW, "hs_proj_weight", kC * kC, "post");
  m_hsProjB = wf(m_postW, "hs_proj_bias", kC, "post");
  {
    auto it = m_postW.find("d_proj");
    if (it == m_postW.end() || it->second.is_float) {
      std::cout << "missing int d_proj" << std::endl;
      exit(1);
    }
    m_dProj = (double)it->second.int_scalar;
  }
  for (int i = 0; i < 3; ++i) {
    const std::string p = "bbox_layer" + std::to_string(i);
    const size_t outc = (i == 2) ? 4 : kC;
    m_bboxW[i] = wf(m_postW, (p + "_weight").c_str(), outc * kC, "post");
    m_bboxB[i] = wf(m_postW, (p + "_bias").c_str(), outc, "post");
  }
  for (int i = 0; i < 3; ++i) {
    const std::string c = "pixel_decoder.conv_" + std::to_string(i);
    const std::string n = "pixel_decoder.norm_" + std::to_string(i);
    m_convW[i] = wf(m_segW, (c + ".weight").c_str(), kC * kC * 9, "seg");
    m_convB[i] = wf(m_segW, (c + ".bias").c_str(), kC, "seg");
    m_normW[i] = wf(m_segW, (n + ".weight").c_str(), kC, "seg");
    m_normB[i] = wf(m_segW, (n + ".bias").c_str(), kC, "seg");
  }
  m_instW = wf(m_segW, "instance_seg_head.weight", kC * kC, "seg");
  m_instB = wf(m_segW, "instance_seg_head.bias", kC, "seg");
  for (int i = 0; i < 3; ++i) {
    const std::string l = "mask_embed.layer_" + std::to_string(i);
    m_meW[i] = wf(m_segW, (l + ".weight").c_str(), kC * kC, "seg");
    m_meB[i] = wf(m_segW, (l + ".bias").c_str(), kC, "seg");
  }
  m_inProjW = wf(m_segW, "cross_attn.in_proj_weight", 3 * kC * kC, "seg");
  m_inProjB = wf(m_segW, "cross_attn.in_proj_bias", 3 * kC, "seg");
  m_outProjW = wf(m_segW, "cross_attn.out_proj.weight", kC * kC, "seg");
  m_outProjB = wf(m_segW, "cross_attn.out_proj.bias", kC, "seg");

  // semantic_seg_head weights exist in the npz but the semantic mask is
  // never used by the python pipeline's output — skipped here too.

  // --- prepack conv weights: [256,256,3,3] -> W2[256, 2304]
  // W2[k, c*9+kk] = w[k, c, kh, kw], col row order matches im2col3x3.
  m_convW2.resize(2 * kC * kC * 9);
  for (int s = 0; s < 2; ++s) {  // stage 0 (72x72) and stage 1 (144x144)
    float* dst = &m_convW2[s * kC * kC * 9];
    const float* src = m_convW[s];
    for (int k = 0; k < kC; ++k)
      for (int c = 0; c < kC; ++c)
        for (int kk = 0; kk < 9; ++kk)
          dst[k * (kC * 9) + c * 9 + kk] = src[(k * kC + c) * 9 + kk];
  }

  // --- prepack attention weights: in_proj/out_proj are [out,in] row-major;
  // the gemm path needs the transpose so B[c][d] = W[d][c].
  m_qW.resize(kC * kC);
  m_oW.resize(kC * kC);
  for (int d = 0; d < kC; ++d)
    for (int c = 0; c < kC; ++c) {
      m_qW[(size_t)c * kC + d] = m_inProjW[(size_t)d * kC + c];
      m_oW[(size_t)c * kC + d] = m_outProjW[(size_t)d * kC + c];
    }

  // --- host buffers ---
  m_inputBuf.resize(3 * kRes * kRes);
  m_vitChain.resize(kGrid * kGrid * 1024);
  m_vitFeat.resize(1024 * kGrid * kGrid);
  m_featS4.resize(kC * 144 * 144);
  m_featS2.resize(kC * 72 * 72);
  m_featS1.resize(kC * kGrid * kGrid);
  m_s1Pos.resize(kC * kGrid * kGrid);
  computeSinePos();
  m_textFeat.resize(kTextLen * kC);
  m_textMask.resize(kTextLen);
  m_tokenIds.resize(kTextLen);
  m_memory.resize(kToks * kC);
  m_memoryPos.resize(kToks * kC);
  m_hs.resize(kLayers * kQueries * kC);
  m_refBox.resize(kLayers * kQueries * 4);
  m_col.resize((size_t)kC * 9 * 144 * 144);
  m_xpad.resize((size_t)kC * (144 + 2) * (144 + 2));
  m_qproj.resize(kToks * kC);
  m_attendOut.resize(kToks * kC);
  m_attended.resize(kToks * kC);
  m_encVisual.resize(kC * kGrid * kGrid);
  m_pixel.resize(kC * 144 * 144);
  m_pixel2.resize(kC * 144 * 144);
  m_up.resize(kC * 144 * 144);
  m_instEmb.resize(kC * 144 * 144);

  // --- networks ---
  const std::string md = m_modelDir;
  m_vit0.ctx = std::make_shared<BMNNContext>(m_handle,
      (md + "/vit/sam3_vit_part0_f16_1b.bmodel").c_str());
  m_vit0.net = m_vit0.ctx->network(0);

  m_neckCtx = std::make_shared<BMNNContext>(m_handle,
      (md + "/neck/sam3_neck_f16_1b.bmodel").c_str());
  m_neckNet = m_neckCtx->network(0);

  m_textCtx = std::make_shared<BMNNContext>(m_handle,
      (md + "/grounding/sam3_text_encoder_f16_1b.bmodel").c_str());
  m_textNet = m_textCtx->network(0);

  m_gndEncCtx = std::make_shared<BMNNContext>(m_handle,
      (md + "/grounding/sam3_grounding_encoder_f16_1b.bmodel").c_str());
  m_gndEncNet = m_gndEncCtx->network(0);

  m_gndDecCtx = std::make_shared<BMNNContext>(m_handle,
      (md + "/grounding/sam3_grounding_decoder_f16_1b.bmodel").c_str());
  m_gndDecNet = m_gndDecCtx->network(0);

  if (!m_streaming) {
    m_vitParts.resize(4);
    for (int i = 0; i < 4; ++i) {
      auto& p = m_vitParts[i];
      p.ctx = std::make_shared<BMNNContext>(m_handle,
          (md + "/vit/sam3_vit_part" + std::to_string(i + 1) +
           "_f16_1b.bmodel").c_str());
      p.net = p.ctx->network(0);
      auto in = p.net->inputTensor(0);
      bmrt_tensor(&p.in_dev, p.ctx->bmrt(), in->get_dtype(), *in->get_shape());
      in->set_device_mem(&p.in_dev.device_mem);
      p.has_dev = true;
    }
  }

  // input tensors of resident nets
  {
    auto in = m_vit0.net->inputTensor(0);
    bmrt_tensor(&m_vit0In, m_vit0.ctx->bmrt(), in->get_dtype(), *in->get_shape());
    in->set_device_mem(&m_vit0In.device_mem);
    m_vit0InOk = true;
  }
  {
    auto in = m_neckNet->inputTensor(0);
    bmrt_tensor(&m_neckIn, m_neckCtx->bmrt(), in->get_dtype(), *in->get_shape());
    in->set_device_mem(&m_neckIn.device_mem);
    m_neckInOk = true;
  }
  {
    auto in = m_textNet->inputTensor(0);
    bmrt_tensor(&m_textIn, m_textCtx->bmrt(), in->get_dtype(), *in->get_shape());
    in->set_device_mem(&m_textIn.device_mem);
    m_textInOk = true;
  }
  // grounding encoder: match inputs by name (src, src_pos, prompt, prompt_mask)
  m_gndEncIn.resize(m_gndEncNet->m_netinfo->input_num);
  for (int i = 0; i < m_gndEncNet->m_netinfo->input_num; ++i) {
    auto in = m_gndEncNet->inputTensor(i);
    bmrt_tensor(&m_gndEncIn[i], m_gndEncCtx->bmrt(), in->get_dtype(), *in->get_shape());
    in->set_device_mem(&m_gndEncIn[i].device_mem);
  }
  m_gndEncIdx = {-1, -1, -1, -1};
  for (int i = 0; i < m_gndEncNet->m_netinfo->input_num; ++i) {
    const std::string n = m_gndEncNet->m_netinfo->input_names[i];
    if (n == "src") m_gndEncIdx[0] = i;
    else if (n == "src_pos") m_gndEncIdx[1] = i;
    else if (n == "prompt") m_gndEncIdx[2] = i;
    else if (n == "prompt_mask") m_gndEncIdx[3] = i;
  }
  // grounding decoder: memory, memory_pos, memory_mask, valid_ratios, prompt, prompt_mask
  m_gndDecIn.resize(m_gndDecNet->m_netinfo->input_num);
  for (int i = 0; i < m_gndDecNet->m_netinfo->input_num; ++i) {
    auto in = m_gndDecNet->inputTensor(i);
    bmrt_tensor(&m_gndDecIn[i], m_gndDecCtx->bmrt(), in->get_dtype(), *in->get_shape());
    in->set_device_mem(&m_gndDecIn[i].device_mem);
  }
  m_gndDecIdx = {-1, -1, -1, -1, -1, -1, -1};
  for (int i = 0; i < m_gndDecNet->m_netinfo->input_num; ++i) {
    const std::string n = m_gndDecNet->m_netinfo->input_names[i];
    if (n == "memory") m_gndDecIdx[0] = i;
    else if (n == "memory_pos") m_gndDecIdx[1] = i;
    else if (n == "memory_mask") m_gndDecIdx[2] = i;
    else if (n == "spatial_shapes") m_gndDecIdx[3] = i;  // static, filled below
    else if (n == "valid_ratios") m_gndDecIdx[4] = i;
    else if (n == "prompt") m_gndDecIdx[5] = i;
    else if (n == "prompt_mask") m_gndDecIdx[6] = i;
  }
  // valid_ratios: reuse slot — see runGrounding; keep both mapped.
  // spatial_shapes (slot 3) is optional: some bmodels fold it as a constant.
  for (int k = 0; k < 7; ++k) {
    if (k == 3) continue;
    if (m_gndDecIdx[k] < 0) {
      std::cout << "[SAM3] grounding decoder input " << k
                << " not found in bmodel" << std::endl;
      return -1;
    }
  }
  if (m_gndDecIdx[3] >= 0) {
    // spatial_shapes is an int32 [1,2] static input: [36,36]
    int32_t shapes[2] = {kGrid, kGrid};
    bm_memcpy_s2d(m_gndDecCtx->handle(), m_gndDecIn[m_gndDecIdx[3]].device_mem,
                  (void*)shapes);
  }
  if (m_gndDecIdx[4] >= 0) {
    const float vr[2] = {1.f, 1.f};
    bm_memcpy_s2d(m_gndDecCtx->handle(), m_gndDecIn[m_gndDecIdx[4]].device_mem,
                  (void*)vr);
  }

  std::cout << "[SAM3] init done: streaming=" << (m_streaming ? "on" : "off")
            << " threads=" << m_nThreads << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// Sine position embedding for the 36x36 s1 feature (PositionEmbeddingSine).
// Output layout matches the bmodel src_pos input [1,256,36,36]: channel-major,
// pos[c][y*W+x]. Channels 0..127 come from y, 128..255 from x; within each
// half dim_t[2j] == dim_t[2j+1] so sin/cos pairs share one frequency.
// ---------------------------------------------------------------------------
void Sam3::computeSinePos() {
  const int H = kGrid, W = kGrid;
  const int half = 128;  // num_pos_feats/2
  const double scale = 2.0 * M_PI;
  // dim_t[i] = 10000^(2*(i//2)/128), i in [0,128)
  double dimt[128];
  for (int i = 0; i < half; ++i)
    dimt[i] = pow(10000.0, 2.0 * (i / 2) / half);

  for (int y = 0; y < H; ++y) {
    const double yv = (double)(y + 1) / (H + 1e-6) * scale;
    for (int x = 0; x < W; ++x) {
      const double xv = (double)(x + 1) / (W + 1e-6) * scale;
      const int hw = y * W + x;
      for (int i = 0; i < 64; ++i) {
        const double ty = yv / dimt[2 * i];
        const double tx = xv / dimt[2 * i];
        m_s1Pos[(size_t)(2 * i) * kToks + hw] = (float)sin(ty);
        m_s1Pos[(size_t)(2 * i + 1) * kToks + hw] = (float)cos(ty);
        m_s1Pos[(size_t)(half + 2 * i) * kToks + hw] = (float)sin(tx);
        m_s1Pos[(size_t)(half + 2 * i + 1) * kToks + hw] = (float)cos(tx);
      }
    }
  }
}

void Sam3::groupNorm8(float* x, const float* w, const float* b, int C, int H,
                      int W) {
  // 8 groups over C channels, N=1.
  const int cg = C / 8;
  const int hw = H * W;
  for (int g = 0; g < 8; ++g) {
    const float* gx = x + (size_t)g * cg * hw;
    double sum = 0, sq = 0;
    const size_t n = (size_t)cg * hw;
    for (size_t i = 0; i < n; ++i) {
      sum += gx[i];
      sq += (double)gx[i] * gx[i];
    }
    const double mean = sum / n;
    const double var = sq / n - mean * mean;
    const float inv = (float)(1.0 / sqrt(var + 1e-5));
    for (int c = 0; c < cg; ++c) {
      float* cx = x + (size_t)(g * cg + c) * hw;
      const float a = w[g * cg + c], bb = b[g * cg + c];
      for (size_t i = 0; i < hw; ++i) cx[i] = (cx[i] - (float)mean) * inv * a + bb;
    }
  }
}

// ---------------------------------------------------------------------------
// im2col + GEMM conv3x3 (stride 1, pad 1). stage selects the prepacked weight.
// ---------------------------------------------------------------------------
void Sam3::im2col3x3(const float* x, int C, int H, int W) {
  const int PW = W + 2;
  // zero-pad into m_xpad [C, H+2, W+2]
  for (int c = 0; c < C; ++c) {
    float* dst = &m_xpad[(size_t)c * (H + 2) * PW];
    memset(dst, 0, sizeof(float) * (H + 2) * PW);
    const float* src = x + (size_t)c * H * W;
    for (int y = 0; y < H; ++y)
      memcpy(dst + (y + 1) * PW + 1, src + (size_t)y * W, sizeof(float) * W);
  }
  // col[c*9+kk, y*W+x] = xpad[c, (y+kh)*PW + (x+kw)]
  const int N = H * W;
  for (int c = 0; c < C; ++c) {
    const float* xc = &m_xpad[(size_t)c * (H + 2) * PW];
    for (int kk = 0; kk < 9; ++kk) {
      const int kh = kk / 3, kw = kk % 3;
      float* col = &m_col[(size_t)(c * 9 + kk) * N];
      for (int y = 0; y < H; ++y) {
        const float* row = xc + (size_t)(y + kh) * PW + kw;
        memcpy(col + (size_t)y * W, row, sizeof(float) * W);
      }
    }
  }
}

void Sam3::gemm(std::vector<float>& aT, const float* A, const float* B,
                float* Y, int M, int N, int K, int nthreads) {
  // Y[M,N] = A[M,K] @ B[K,N]. A is packed transposed into aT [K,M] first so
  // the hot loop streams A, B and Y rows contiguously (a k-outer loop over
  // column-major B touches one float per cache line and is ~16x slower on the
  // Cortex-A53 cluster). Threads split N; m is blocked so each thread's Y tile
  // stays L2-resident while B is streamed once per m-block.
  aT.resize((size_t)K * M);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      aT[(size_t)k * M + m] = A[(size_t)m * K + k];

  auto worker = [&](int t) {
    const int nb = (N + nthreads - 1) / nthreads;
    const int n0 = t * nb;
    const int n1 = std::min(N, n0 + nb);
    constexpr int MB = 16;
    for (int m0 = 0; m0 < M; m0 += MB) {
      const int m1 = std::min(M, m0 + MB);
      for (int k = 0; k < K; ++k) {
        const float* ak = &aT[(size_t)k * M];
        const float* bk = B + (size_t)k * N + n0;
        for (int m = m0; m < m1; ++m) {
          const float a = ak[m];
          if (a == 0.f) continue;
          float* yr = Y + (size_t)m * N + n0;
          for (int i = 0; i < n1 - n0; ++i) yr[i] += a * bk[i];
        }
      }
    }
  };
  std::vector<std::thread> th;
  for (int t = 1; t < nthreads; ++t) th.emplace_back(worker, t);
  worker(0);
  for (auto& x : th) x.join();
}

void Sam3::conv3x3Gemm(int stage, const float* x, float* y, int H, int W) {
  im2col3x3(x, kC, H, W);
  const float* W2 = &m_convW2[(size_t)stage * kC * kC * 9];
  gemm(m_gemmA, W2, m_col.data(), y, kC, H * W, kC * 9, m_nThreads);
  const float* b = m_convB[stage];
  for (int k = 0; k < kC; ++k) {
    const float bias = b[k];
    float* yk = y + (size_t)k * H * W;
    for (int i = 0; i < H * W; ++i) yk[i] += bias;
  }
}

// ---------------------------------------------------------------------------
// Stage 1+2: ViT chain and neck.
// ---------------------------------------------------------------------------
int Sam3::runVit(const float* input) {
  bm_handle_t h = m_handle->handle();

  // part0: [1,3,504,504] -> [1,36,36,1024]
  bm_memcpy_s2d(h, m_vit0In.device_mem, (void*)input);
  if (m_vit0.net->forward() != 0) return -1;
  auto out0 = m_vit0.net->outputTensor(0);
  bm_memcpy_d2s_partial(h, m_vitChain.data(), *out0->get_device_mem(),
                        m_vitChain.size() * 4);

  // part1..4: [1,36,36,1024] -> [1,36,36,1024] each
  for (int i = 1; i <= 4; ++i) {
    if (!m_streaming) {
      auto& p = m_vitParts[i - 1];
      bm_memcpy_s2d(h, p.in_dev.device_mem, m_vitChain.data());
      if (p.net->forward() != 0) return -1;
      auto out = p.net->outputTensor(0);
      bm_memcpy_d2s_partial(h, m_vitChain.data(), *out->get_device_mem(),
                            m_vitChain.size() * 4);
    } else {
      // Load the bmodel, run, destroy: keeps peak device memory ~1.4G so the
      // default SE7 npu heap (~2.4G) is enough.
      BMNNContext ctx(m_handle, (m_modelDir + "/vit/sam3_vit_part" +
                                 std::to_string(i) + "_f16_1b.bmodel").c_str());
      auto net = ctx.network(0);
      auto in = net->inputTensor(0);
      bm_tensor_t dev;
      bmrt_tensor(&dev, ctx.bmrt(), in->get_dtype(), *in->get_shape());
      in->set_device_mem(&dev.device_mem);
      bm_memcpy_s2d(h, dev.device_mem, m_vitChain.data());
      int ret = net->forward();
      if (ret == 0) {
        auto out = net->outputTensor(0);
        bm_memcpy_d2s_partial(h, m_vitChain.data(), *out->get_device_mem(),
                              m_vitChain.size() * 4);
      }
      bm_free_device(h, dev.device_mem);
      if (ret != 0) return -1;
    }
  }

  // CPU transpose [1,36,36,1024] -> [1,1024,36,36]
  const int HW = kGrid * kGrid;
  for (int c = 0; c < 1024; ++c)
    for (int hw = 0; hw < HW; ++hw)
      m_vitFeat[c * HW + hw] = m_vitChain[hw * 1024 + c];
  return 0;
}

int Sam3::runNeck(const float* vitFeat) {
  bm_handle_t h = m_handle->handle();
  bm_memcpy_s2d(h, m_neckIn.device_mem, (void*)vitFeat);
  if (m_neckNet->forward() != 0) return -1;

  // outputs: feat_s4 [1,256,144,144], feat_s2 [1,256,72,72], feat_s1
  // [1,256,36,36], feat_s05 [1,256,18,18] (s05 unused at scalp=1)
  const int n = m_neckNet->outputTensorNum();
  for (int i = 0; i < n; ++i) {
    const std::string name = m_neckNet->m_netinfo->output_names[i];
    float* dst = nullptr;
    size_t cnt = 0;
    if (name.find("feat_s4") != std::string::npos) {
      dst = m_featS4.data(); cnt = m_featS4.size();
    } else if (name.find("feat_s2") != std::string::npos) {
      dst = m_featS2.data(); cnt = m_featS2.size();
    } else if (name.find("feat_s1") != std::string::npos) {
      dst = m_featS1.data(); cnt = m_featS1.size();
    } else continue;
    auto t = m_neckNet->outputTensor(i);
    bm_memcpy_d2s_partial(h, dst, *t->get_device_mem(), cnt * 4);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Text encoder: token ids -> [32,1,256] features (flat == [1,32,256]).
// ---------------------------------------------------------------------------
int Sam3::runText(const std::vector<int32_t>& ids) {
  bm_handle_t h = m_handle->handle();
  bm_memcpy_s2d(h, m_textIn.device_mem, (void*)ids.data());
  if (m_textNet->forward() != 0) return -1;
  auto t = m_textNet->outputTensor(0);
  bm_memcpy_d2s_partial(h, m_textFeat.data(), *t->get_device_mem(),
                        m_textFeat.size() * 4);
  return 0;
}

// ---------------------------------------------------------------------------
// Grounding encoder + decoder (TPU). Uses m_featS1/m_s1Pos/m_textFeat.
// ---------------------------------------------------------------------------
int Sam3::runGrounding() {
  bm_handle_t h = m_handle->handle();
  const size_t promptBytes = (size_t)kTextLen * kC * 4;
  const size_t maskBytes = (size_t)kTextLen * 4;

  // --- encoder: src [1,256,36,36], src_pos, prompt [1,32,256], prompt_mask
  bm_memcpy_s2d(h, m_gndEncIn[m_gndEncIdx[0]].device_mem, m_featS1.data());
  bm_memcpy_s2d(h, m_gndEncIn[m_gndEncIdx[1]].device_mem, m_s1Pos.data());
  // text features [32,1,256] and [1,32,256] share the same flat layout (B=1)
  bm_memcpy_s2d(h, m_gndEncIn[m_gndEncIdx[2]].device_mem, m_textFeat.data());
  bm_memcpy_s2d(h, m_gndEncIn[m_gndEncIdx[3]].device_mem, m_textMask.data());
  if (m_gndEncNet->forward() != 0) return -1;
  {
    auto t0 = m_gndEncNet->outputTensor(0);  // memory [1296,1,256]
    auto t1 = m_gndEncNet->outputTensor(1);  // pos_embed [1296,1,256]
    bm_memcpy_d2s_partial(h, m_memory.data(), *t0->get_device_mem(),
                          m_memory.size() * 4);
    bm_memcpy_d2s_partial(h, m_memoryPos.data(), *t1->get_device_mem(),
                          m_memoryPos.size() * 4);
  }

  // --- decoder ---
  // memory_mask [1296,1] zeros
  std::vector<float> zeros(kToks, 0.f);
  bm_memcpy_s2d(h, m_gndDecIn[m_gndDecIdx[2]].device_mem, zeros.data());
  bm_memcpy_s2d(h, m_gndDecIn[m_gndDecIdx[0]].device_mem, m_memory.data());
  bm_memcpy_s2d(h, m_gndDecIn[m_gndDecIdx[1]].device_mem, m_memoryPos.data());
  bm_memcpy_s2d(h, m_gndDecIn[m_gndDecIdx[5]].device_mem, m_textFeat.data());
  bm_memcpy_s2d(h, m_gndDecIn[m_gndDecIdx[6]].device_mem, m_textMask.data());
  if (m_gndDecNet->forward() != 0) return -1;
  {
    auto t0 = m_gndDecNet->outputTensor(0);  // hs [6,200,1,256]
    auto t1 = m_gndDecNet->outputTensor(1);  // reference_boxes [6,200,1,4]
    bm_memcpy_d2s_partial(h, m_hs.data(), *t0->get_device_mem(),
                          m_hs.size() * 4);
    bm_memcpy_d2s_partial(h, m_refBox.data(), *t1->get_device_mem(),
                          m_refBox.size() * 4);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// CPU post-processing (last decoder layer only): scoring + box refinement.
// ---------------------------------------------------------------------------
int Sam3::postProcess(const std::string& prompt, Sam3Result& r) {
  // text mask as float 1=pad
  float tmask[kTextLen];
  for (int t = 0; t < kTextLen; ++t) tmask[t] = m_textMask[t];

  // --- mean-pool text over valid tokens ---
  double pooled[kC];
  memset(pooled, 0, sizeof(pooled));
  double nValid = 0;
  for (int t = 0; t < kTextLen; ++t) {
    if (tmask[t] == 0.f) {
      nValid += 1;
      const float* f = &m_textFeat[(size_t)t * kC];
      for (int c = 0; c < kC; ++c) pooled[c] += f[c];
    }
  }
  if (nValid < 1) nValid = 1;
  for (int c = 0; c < kC; ++c) pooled[c] /= nValid;

  // --- project pooled text: [256] @ prompt_proj_weight^T + b ---
  float projP[kC];
  for (int d = 0; d < kC; ++d) {
    double s = m_promptProjB[d];
    const float* wrow = m_promptProjW + (size_t)d * kC;
    for (int c = 0; c < kC; ++c) s += pooled[c] * wrow[c];
    projP[d] = (float)s;
  }

  // --- last layer hs: [200,256] -> scores, boxes ---
  const float* hsL = &m_hs[(size_t)(kLayers - 1) * kQueries * kC];
  const float* refL = &m_refBox[(size_t)(kLayers - 1) * kQueries * 4];
  const float scale = (float)(1.0 / sqrt(m_dProj));

  int best = 0;
  float bestScore = -1.f;
  std::vector<float> scores(kQueries);
  std::vector<float> boxes(kQueries * 4);
  for (int q = 0; q < kQueries; ++q) {
    // proj_hs = hs @ hs_proj_weight^T + b, then dot with projP
    const float* hq = hsL + (size_t)q * kC;
    double dot = 0;
    for (int d = 0; d < kC; ++d) {
      double s = m_hsProjB[d];
      const float* wrow = m_hsProjW + (size_t)d * kC;
      for (int c = 0; c < kC; ++c) s += (double)hq[c] * wrow[c];
      dot += s * projP[d];
    }
    const float score = 1.f / (1.f + expf(-(float)(dot * scale)));
    scores[q] = score;

    // bbox offsets: 3-layer MLP on hs
    float x[kC];
    memcpy(x, hq, kC * 4);
    for (int l = 0; l < 3; ++l) {
      const int oc = (l == 2) ? 4 : kC;
      float y[4 <= kC ? kC : 4];
      for (int o = 0; o < oc; ++o) {
        double s = m_bboxB[l][o];
        const float* wrow = m_bboxW[l] + (size_t)o * kC;
        for (int c = 0; c < kC; ++c) s += (double)x[c] * wrow[c];
        y[o] = (l < 2) ? std::max(0.f, (float)s) : (float)s;
      }
      memcpy(x, y, oc * 4);
    }
    // box = sigmoid(inverse_sigmoid(ref) + offset)
    for (int k = 0; k < 4; ++k) {
      float rx = std::min(1.f, std::max(0.f, refL[q * 4 + k]));
      const float x1 = std::max(rx, 1e-3f);
      const float x2 = std::max(1.f - rx, 1e-3f);
      const float refInv = logf(x1 / x2);
      boxes[q * 4 + k] = 1.f / (1.f + expf(-(refInv + x[k])));
    }

    if (score > bestScore) {
      bestScore = score;
      best = q;
    }
  }

  r.prompt = prompt;
  r.best_idx = best;
  r.score = bestScore;
  memcpy(r.box_norm, &boxes[(size_t)best * 4], 4 * 4);
  (void)scores;
  return 0;
}

// ---------------------------------------------------------------------------
// Mask decoder (CPU): cross-attend -> pixel decoder -> instance head ->
// mask embed MLP -> einsum, best query only.
// ---------------------------------------------------------------------------
int Sam3::predictMask(Sam3Result& r) {
  const int Q = r.best_idx;
  if (Q < 0) return -1;

  // --- 1. cross-attend: encoder memory [1296,256] attends to text [32,256] ---
  // Note: python applies a layer norm to build `tgt` but never uses it (dead
  // code in sam3_infer.py::_cross_attend); skipped here.
  // in_proj rows 0-255 Q (from memory), 256-511 K / 512-767 V (from text).
  const int heads = 8, hdim = 32;
  {
    // Q = memory @ q_w^T + q_b; q_w^T prepacked in m_qW ([c][d] row-major)
    gemm(m_gemmA, m_memory.data(), m_qW.data(), m_qproj.data(), kToks, kC, kC,
         m_nThreads);
    for (int t = 0; t < kToks; ++t) {
      float* q = &m_qproj[(size_t)t * kC];
      for (int d = 0; d < kC; ++d) q[d] += m_inProjB[d];
    }
    // K/V of the 32 text tokens stay small: direct per-token matvec.
    float Kp[kTextLen * kC], Vp[kTextLen * kC];
    for (int t = 0; t < kTextLen; ++t) {
      const float* src = &m_textFeat[(size_t)t * kC];
      for (int half = 0; half < 2; ++half) {  // K then V
        float* dst = half == 0 ? &Kp[(size_t)t * kC] : &Vp[(size_t)t * kC];
        const float* bias = m_inProjB + (256 + half * 256);
        const float* wbase = m_inProjW + (size_t)(256 + half * 256) * kC;
        for (int d = 0; d < kC; ++d) {
          double s = bias[d];
          const float* wrow = wbase + (size_t)d * kC;
          for (int c = 0; c < kC; ++c) s += (double)src[c] * wrow[c];
          dst[d] = (float)s;
        }
      }
    }

    // attention per token per head; padded text positions get -1e9 before
    // the max-subtract softmax (python: attn - 1e9 * mask).
    const float ascale = 1.f / sqrtf((float)hdim);
    auto attnToken = [&](int t0, int t1) {
      float attn[kTextLen];
      for (int t = t0; t < t1; ++t) {
        float out[kC];
        memset(out, 0, sizeof(out));
        const float* qt = &m_qproj[(size_t)t * kC];
        for (int h = 0; h < heads; ++h) {
          float mx = -1e30f;
          for (int s = 0; s < kTextLen; ++s) {
            if (m_textMask[s] != 0.f) {
              attn[s] = -1e30f;
              continue;
            }
            const float* kt = &Kp[(size_t)s * kC + h * hdim];
            float acc = 0;
            for (int d = 0; d < hdim; ++d) acc += qt[h * hdim + d] * kt[d];
            attn[s] = acc * ascale;
            mx = std::max(mx, attn[s]);
          }
          float denom = 0;
          for (int s = 0; s < kTextLen; ++s) {
            attn[s] = expf(attn[s] - mx);
            denom += attn[s];
          }
          for (int s = 0; s < kTextLen; ++s) {
            const float w = attn[s] / denom;
            if (w == 0.f) continue;
            const float* vt = &Vp[(size_t)s * kC + h * hdim];
            for (int d = 0; d < hdim; ++d) out[h * hdim + d] += w * vt[d];
          }
        }
        memcpy(&m_attendOut[(size_t)t * kC], out, sizeof(out));
      }
    };
    if (m_nThreads <= 1) {
      attnToken(0, kToks);
    } else {
      // 1296 tokens is embarrassingly parallel across threads.
      std::vector<std::thread> th;
      const int nb = (kToks + m_nThreads - 1) / m_nThreads;
      for (int tId = 1; tId < m_nThreads; ++tId) {
        const int b0 = tId * nb;
        if (b0 >= kToks) break;
        const int b1 = std::min(kToks, b0 + nb);
        th.emplace_back(attnToken, b0, b1);
      }
      attnToken(0, nb);
      for (auto& x : th) x.join();
    }

    // out_proj + residual: attended = attendOut @ o_w^T + b + memory
    gemm(m_gemmA, m_attendOut.data(), m_oW.data(), m_attended.data(), kToks,
         kC, kC, m_nThreads);
    for (int t = 0; t < kToks; ++t) {
      float* dst = &m_attended[(size_t)t * kC];
      const float* mem = &m_memory[(size_t)t * kC];
      for (int d = 0; d < kC; ++d) dst[d] += m_outProjB[d] + mem[d];
    }
  }

  // --- 2. encoder_visual: attended[:1296] token-major -> [256,36,36] ---
  for (int c = 0; c < kC; ++c)
    for (int hw = 0; hw < kToks; ++hw)
      m_encVisual[(size_t)c * kToks + hw] = m_attended[(size_t)hw * kC + c];

  // --- 3. pixel decoder ---
  // iter 0: upsample enc_visual 36->72 nearest, add s2, conv_0, GN, ReLU
  {
    const int SH = 36, SW = 72;
    for (int c = 0; c < kC; ++c) {
      const float* lo = &m_encVisual[(size_t)c * 36 * 36];
      const float* s2 = &m_featS2[(size_t)c * 72 * 72];
      float* up = &m_up[(size_t)c * 72 * 72];
      for (int y = 0; y < SW; ++y) {
        const int sy = y * SH / SW;  // floor(y*36/72)
        for (int x = 0; x < SW; ++x) {
          const int sx = x * SH / SW;
          up[y * SW + x] = s2[y * SW + x] + lo[sy * 36 + sx];
        }
      }
    }
    conv3x3Gemm(0, m_up.data(), m_pixel.data(), 72, 72);
    groupNorm8(m_pixel.data(), m_normW[0], m_normB[0], kC, 72, 72);
    for (auto& v : m_pixel) v = std::max(0.f, v);
  }
  // iter 1: upsample 72->144, add s4, conv_1, GN, ReLU
  {
    const int SH = 72, SW = 144;
    for (int c = 0; c < kC; ++c) {
      const float* lo = &m_pixel[(size_t)c * 72 * 72];
      const float* s4 = &m_featS4[(size_t)c * 144 * 144];
      float* up = &m_up[(size_t)c * 144 * 144];
      for (int y = 0; y < SW; ++y) {
        const int sy = y * SH / SW;
        for (int x = 0; x < SW; ++x) {
          const int sx = x * SH / SW;
          up[y * SW + x] = s4[y * SW + x] + lo[sy * 72 + sx];
        }
      }
    }
    conv3x3Gemm(1, m_up.data(), m_pixel2.data(), 144, 144);
    groupNorm8(m_pixel2.data(), m_normW[1], m_normB[1], kC, 144, 144);
    for (auto& v : m_pixel2) v = std::max(0.f, v);
  }

  // --- 4. instance 1x1 conv: instEmb = instW @ pixel2 + b ---
  // im2col of a 1x1 conv is the identity: gemm directly on pixel2 [K=256, N=HW]
  gemm(m_gemmA, m_instW, m_pixel2.data(), m_instEmb.data(), kC, 144 * 144, kC,
       m_nThreads);
  for (int k = 0; k < kC; ++k) {
    const float bias = m_instB[k];
    float* yk = &m_instEmb[(size_t)k * 144 * 144];
    for (int i = 0; i < 144 * 144; ++i) yk[i] += bias;
  }

  // --- 5. mask embed MLP on the best query: [256] -> [256] ---
  float mvec[kC];
  memcpy(mvec, &m_hs[(size_t)((kLayers - 1) * kQueries + Q) * kC], kC * 4);
  for (int l = 0; l < 3; ++l) {
    float y[kC];
    for (int o = 0; o < kC; ++o) {
      double s = m_meB[l][o];
      const float* wrow = m_meW[l] + (size_t)o * kC;
      for (int c = 0; c < kC; ++c) s += (double)mvec[c] * wrow[c];
      y[o] = (l < 2) ? std::max(0.f, (float)s) : (float)s;
    }
    memcpy(mvec, y, kC * 4);
  }

  // --- 6. pred_mask[144,144] = mvec [256] @ instEmb [256, 20736] ---
  r.mask.create(144, 144, CV_8UC1);
  for (int y = 0; y < 144; ++y) {
    uchar* row = r.mask.ptr<uchar>(y);
    for (int x = 0; x < 144; ++x) {
      const size_t idx = (size_t)y * 144 + x;
      float acc = 0;
      for (int c = 0; c < kC; ++c) acc += mvec[c] * m_instEmb[c * 20736 + idx];
      row[x] = acc > 0 ? 255 : 0;
    }
  }
  r.mask_pixels = cv::countNonZero(r.mask);
  return 0;
}

// ---------------------------------------------------------------------------
// Full pipeline.
// ---------------------------------------------------------------------------
int Sam3::predict(const cv::Mat& bgr, const std::vector<std::string>& prompts,
                  float scoreThresh, std::vector<Sam3Result>& results) {
  LOG_TS(m_ts, "sam3 preprocess");
  // BGR->RGB, resize 504x504 INTER_LINEAR, /127.5-1, HWC->CHW
  cv::Mat rgb, resized;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(kRes, kRes), 0, 0, cv::INTER_LINEAR);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < kRes; ++y)
      for (int x = 0; x < kRes; ++x) {
        const cv::Vec3b px = resized.at<cv::Vec3b>(y, x);
        m_inputBuf[((size_t)c * kRes + y) * kRes + x] =
            (float)px[c] / 127.5f - 1.f;
      }
  LOG_TS(m_ts, "sam3 preprocess");

  LOG_TS(m_ts, "sam3 vit");
  if (runVit(m_inputBuf.data()) != 0) return -1;
  LOG_TS(m_ts, "sam3 vit");

  LOG_TS(m_ts, "sam3 neck");
  if (runNeck(m_vitFeat.data()) != 0) return -1;
  LOG_TS(m_ts, "sam3 neck");

  const int imgH = bgr.rows, imgW = bgr.cols;
  results.clear();
  for (const std::string& prompt : prompts) {
    m_tokenIds = m_tokenizer->encode(prompt);
    for (int t = 0; t < kTextLen; ++t)
      m_textMask[t] = (m_tokenIds[t] == 0) ? 1.f : 0.f;

    LOG_TS(m_ts, "sam3 text");
    if (runText(m_tokenIds) != 0) return -1;
    LOG_TS(m_ts, "sam3 text");

    LOG_TS(m_ts, "sam3 grounding");
    if (runGrounding() != 0) return -1;
    LOG_TS(m_ts, "sam3 grounding");

    LOG_TS(m_ts, "sam3 post");
    Sam3Result r;
    if (postProcess(prompt, r) != 0) return -1;
    // normalized cxcywh -> pixel xyxy
    const float cx = r.box_norm[0], cy = r.box_norm[1];
    const float w = r.box_norm[2], h = r.box_norm[3];
    r.box_xyxy[0] = (int)((cx - w / 2) * imgW);
    r.box_xyxy[1] = (int)((cy - h / 2) * imgH);
    r.box_xyxy[2] = (int)((cx + w / 2) * imgW);
    r.box_xyxy[3] = (int)((cy + h / 2) * imgH);
    LOG_TS(m_ts, "sam3 post");

    if (r.score >= scoreThresh) {
      LOG_TS(m_ts, "sam3 mask");
      if (predictMask(r) != 0) return -1;
      // 144x144 -> full image, nearest
      if (!r.mask.empty()) {
        cv::Mat full;
        cv::resize(r.mask, full, cv::Size(imgW, imgH), 0, 0, cv::INTER_NEAREST);
        r.mask = full;
        r.mask_pixels = cv::countNonZero(r.mask);
      }
      LOG_TS(m_ts, "sam3 mask");
    }
    results.push_back(std::move(r));
  }
  return 0;
}

void Sam3::drawResults(cv::Mat& image, const std::vector<Sam3Result>& results) {
  const cv::Scalar color(0, 255, 0);
  for (const auto& r : results) {
    cv::Mat colorLayer = cv::Mat::zeros(image.size(), image.type());
    if (!r.mask.empty()) colorLayer.setTo(color, r.mask);
    cv::addWeighted(colorLayer, 0.4, image, 1.0, 0, image);

    cv::rectangle(image, cv::Point(r.box_xyxy[0], r.box_xyxy[1]),
                  cv::Point(r.box_xyxy[2], r.box_xyxy[3]), color, 2);
    char label[256];
    snprintf(label, sizeof(label), "%s: %.2f", r.prompt.c_str(), r.score);
    cv::putText(image, label, cv::Point(r.box_xyxy[0], r.box_xyxy[1] - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }
}
