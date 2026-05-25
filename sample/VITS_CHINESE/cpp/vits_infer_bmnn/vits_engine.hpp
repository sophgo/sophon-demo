//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef VITS_ENGINE_HPP
#define VITS_ENGINE_HPP

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "bmruntime_interface.h"
#include "bmlib_runtime.h"
#include "utils.hpp"

class VitsEngine {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    int m_max_length;
    int m_char_embed_dim;
    int m_char_embed_seq_len;
    int m_audio_len;

    TimeStamp* m_ts = NULL;
    TimeStamp tmp_ts;

    float estimate_silence_threshold(const float* audio, int len);
    int remove_silence_from_end(float* audio, int len);

    float* get_cpu_data(bm_tensor_t* tensor);

public:
    int batch_size = -1;
    TimeStamp* m_ts_ptr = NULL;

    VitsEngine(std::string bmodel_file, int dev_id = 0);
    ~VitsEngine();

    int max_length() const { return m_max_length; }
    int audio_len() const { return m_audio_len; }
    int char_embed_dim() const { return m_char_embed_dim; }

    int RunInference(const int32_t* x, int x_len,
                     const float* char_embeds, int char_embeds_len,
                     std::vector<float>& output_audio);

    int RunInferenceRaw(const int32_t* x, int x_len,
                        const float* char_embeds, int char_embeds_len,
                        float*& y_segment, int& y_segment_len, float*& y_max);

    int Postprocess(float* y_segment, int y_segment_len,
                       const float* y_max, std::vector<float>& output_audio);

    static void WriteWav(const std::string& filename,
                         const std::vector<float>& audio,
                         int sample_rate);
};

#endif // VITS_ENGINE_HPP