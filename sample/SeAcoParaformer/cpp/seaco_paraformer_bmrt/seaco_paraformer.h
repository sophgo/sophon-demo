//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef SEACO_PARAFORMER_H
#define SEACO_PARAFORMER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <algorithm>

#include "json.hpp"
#include "audio_process.h"
#include "utils.hpp"

extern "C" {
#include "bmruntime_interface.h"
}

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// CIF (Continuous Integrate-and-Fire) -- CPU implementation
// ---------------------------------------------------------------------------

// With hidden states: returns (N, D) pre_acoustic_embeds
void cif_cpu(const float* hidden, const float* alphas,
             int B, int T, int D, float threshold,
             std::vector<float>& acoustic_embeds, int& out_n);

// Without hidden states: peak detection only, returns fires (T,)
void cif_wo_hidden_cpu(const float* alphas, int T, float threshold,
                       std::vector<float>& fires);

// ---------------------------------------------------------------------------
// Timestamp prediction (ported from FunASR timestamp_tools.py)
// ---------------------------------------------------------------------------

struct TimestampInfo {
    int start_ms;
    int end_ms;
    std::string text;
};

void ts_prediction_lfr6(const float* us_alphas, const float* us_peaks,
                        int num_frames,
                        const std::vector<std::string>& char_list,
                        float force_time_shift, int upsample_rate,
                        std::vector<TimestampInfo>& timestamps,
                        std::vector<std::string>& new_char_list);

// ---------------------------------------------------------------------------
// SeacoParaformer inference class (raw bmrt C API, YOLOv8_cls pattern)
// ---------------------------------------------------------------------------

class SeacoParaformer {
public:
    static const int SAMPLE_RATE = 16000;
    static const int N_MELS = 80;
    static const int FRAME_LENGTH_MS = 25;
    static const int FRAME_SHIFT_MS = 10;
    static const int LFR_M = 7;
    static const int LFR_N = 6;
    static const int FEAT_DIM = N_MELS * LFR_M;   // 560
    static const int SOS_ID = 1;
    static const int EOS_ID = 2;
    static const int BLANK_ID = 0;

    SeacoParaformer(const std::string& model_dir, int dev_id = 0);
    ~SeacoParaformer();

    // Full inference from raw float32 mono audio samples
    struct InferResult {
        std::string text;
        std::vector<std::string> tokens;
        std::vector<int> token_ids;
        std::vector<TimestampInfo> sentence_info;
    };

    InferResult infer(const std::vector<float>& audio, int audio_sample_rate = 16000);

    // Get timing accumulators (in seconds)
    double t_pre() const { return t_pre_; }
    double t_enc() const { return t_enc_; }
    double t_cif() const { return t_cif_; }
    double t_dec() const { return t_dec_; }
    double t_pred() const { return t_pred_; }
    double t_tok() const { return t_tok_; }
    void reset_timing();
    TimeStamp* ts() { return &ts_; }

private:
    // Device
    bm_handle_t handle_;
    void* bmrt_;
    bm_misc_info misc_info_;

    // Three bmodel networks
    struct NetInfo {
        const bm_net_info_t* info;
        int batch_size;
        bool is_dynamic;
        std::vector<bm_shape_t> input_shapes;
        std::vector<bm_shape_t> output_shapes;
        std::vector<bm_data_type_t> input_dtypes;
        std::vector<bm_data_type_t> output_dtypes;
        std::vector<float> input_scales;
        std::vector<float> output_scales;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
    };
    NetInfo enc_net_, dec_net_, pred_net_;

    // Tokenizer
    std::vector<std::string> tokens_;

    // CMVN
    CMVN cmvn_;

    // Timing accumulators
    double t_pre_, t_enc_, t_cif_, t_dec_, t_pred_, t_tok_;
    TimeStamp ts_;

    // Helpers
    void load_network(const std::string& bmodel_path, NetInfo& net);
    void encoder_forward(const float* speech, int speech_len,
                         std::vector<float>& enc_out, int& enc_out_T, int& enc_out_D,
                         std::vector<float>& hidden, int& hidden_T,
                         std::vector<float>& alphas, int& alphas_T,
                         float& token_num);
    void decoder_forward(const float* enc_out, int enc_T, int enc_D,
                         const float* pre_embeds, int pre_N, int pre_D,
                         int pre_token_len,
                         std::vector<float>& logits, int& logits_N, int& vocab_size,
                         std::vector<float>& dec_hidden);
    void predictor_forward(const float* enc_out, int enc_T, int enc_D,
                           std::vector<float>& us_alphas, int& us_T,
                           float& pred_token_num);

    // Data transfer (pattern from YOLOv8_cls)
    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    void free_cpu_data(bm_tensor_t* tensor, float* data);

    // Shape helpers
    static int shape_count(const bm_shape_t& shape);
    static int shape_dim_at(const bm_shape_t& shape, int dim, int default_val);
};

#endif // SEACO_PARAFORMER_H
