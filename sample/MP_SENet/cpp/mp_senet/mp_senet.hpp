//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef MP_SENET_HPP
#define MP_SENET_HPP

#include <string>
#include <vector>
#include <memory>

#include "bmruntime_interface.h"
#include "bmcv_api_ext.h"
#include "wav_utils.h"
#include "utils.hpp"

struct MpSenetConfig {
    std::string bmodel_path;
    std::string wav_path;
    std::string output_path;
    int   dev_id       = 0;
    float compress_factor = 0.27f;   // bf16 default; 0.30 for fp32
    int   n_fft        = 400;
    int   hop_size     = 150;        // bf16: 150, fp32: 100
    int   win_size     = 400;
    int   sample_rate  = 16000;
    bool  is_bf16      = true;
};

class MpSenet {
public:
    MpSenet(const MpSenetConfig& cfg) : cfg_(cfg) {}
    ~MpSenet();

    int Init();
    // Process a single WAV file, write denoised output. Returns 0 on success.
    int process(const std::string& wav_file, const std::string& output_file);

    void enableProfile(TimeStamp* ts) { ts_ = ts; }

private:
    MpSenetConfig cfg_;
    bm_handle_t   bm_handle_ = nullptr;
    void*         p_bmrt_    = nullptr;
    const char*   net_name_  = nullptr;
    const bm_net_info_t* net_info_ = nullptr;

    int max_length_ = 640;       // input dim[2]
    int max_wav_length_ = 0;     // (max_length-1)*hop_size + (hop_size-1)

    float norm_factor_   = 0.0f;
    int   noisy_end_len_ = 0;

    TimeStamp* ts_ = nullptr;

    // One chunk's STFT-derived amp/pha (padded to max_length_).
    struct Chunk {
        std::vector<float> amp;  // [1, 201, 640]
        std::vector<float> pha;  // [1, 201, 640]
        int wav_stft_L;          // original wav length of this chunk
    };

    // Compute mag/pha STFT for a wav chunk [1, L]. Returns amp/pha [1, 201, T].
    int mag_pha_stft(const float* wav, int wav_L,
                     std::vector<float>& amp, std::vector<float>& pha, int& T);
    // Reconstruct wav from amp_g/pha_g [1, 201, real_len]. Returns wav [wav_L].
    int mag_pha_istft(const float* amp_g, const float* pha_g, int real_len,
                      int wav_L, std::vector<float>& wav_out);
};

#endif // MP_SENET_HPP
