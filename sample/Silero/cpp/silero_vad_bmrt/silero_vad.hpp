//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SILERO_VAD_HPP
#define SILERO_VAD_HPP

#include <iostream>
#include <vector>
#include <string>
#include "bmruntime_interface.h"
#include "bmlib_runtime.h"
#include "utils.hpp"

struct SpeechSegment {
    int start;
    int end;
};

class SileroVAD {
public:
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int NUM_SAMPLES = 512;
    static constexpr int CONTEXT_SIZE = 64;
    static constexpr int STATE_DIM = 128;

    SileroVAD(const std::string& bmodel_file, int dev_id = 0);
    ~SileroVAD();

    int process_audio(const float* audio, int audio_len,
                      std::vector<SpeechSegment>& speeches,
                      std::vector<float>& speech_probs,
                      float threshold = 0.5f,
                      int min_speech_duration_ms = 250,
                      int min_silence_duration_ms = 100,
                      int speech_pad_ms = 30);

    TimeStamp* m_ts;

private:
    int forward(const float* x, const float* h, const float* c,
                float* prob, float* h_new, float* c_new);

    bm_handle_t handle;
    void* bmrt;
    const bm_net_info_t* netinfo;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    TimeStamp tmp_ts;
};

float* read_wav(const std::string& path, int& num_samples, int& sample_rate);
bool save_wav(const std::string& path, const float* audio, int num_samples, int sample_rate);

#endif