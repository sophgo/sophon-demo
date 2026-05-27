//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "silero_vad.hpp"
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

constexpr int SileroVAD::SAMPLE_RATE;
constexpr int SileroVAD::NUM_SAMPLES;
constexpr int SileroVAD::CONTEXT_SIZE;
constexpr int SileroVAD::STATE_DIM;

SileroVAD::SileroVAD(const std::string& bmodel_file, int dev_id)
    : handle(nullptr), bmrt(nullptr), netinfo(nullptr), m_ts(nullptr)
{
    auto ret = bm_dev_request(&handle, dev_id);
    assert(BM_SUCCESS == ret);

    ret = bm_get_misc_info(handle, &misc_info);
    assert(BM_SUCCESS == ret);

    bmrt = bmrt_create(handle);
    if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
        std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
        exit(1);
    }

    const char** names;
    int num = bmrt_get_network_number(bmrt);
    if (num > 1) {
        std::cout << "This bmodel have " << num << " networks, and this program will only take network 0." << std::endl;
    }
    bmrt_get_network_names(bmrt, &names);
    for (int i = 0; i < num; ++i) {
        network_names.push_back(names[i]);
    }
    free(names);

    netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
    if (netinfo->stage_num > 1) {
        std::cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << std::endl;
    }

    std::cout << "Model loaded: " << bmodel_file << std::endl;
    std::cout << "  inputs:  " << netinfo->input_num << std::endl;
    for (int i = 0; i < netinfo->input_num; ++i) {
        std::cout << "    " << netinfo->input_names[i] << ": [";
        for (int j = 0; j < netinfo->stages[0].input_shapes[i].num_dims; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << netinfo->stages[0].input_shapes[i].dims[j];
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "  outputs: " << netinfo->output_num << std::endl;
    for (int i = 0; i < netinfo->output_num; ++i) {
        std::cout << "    " << netinfo->output_names[i] << ": [";
        for (int j = 0; j < netinfo->stages[0].output_shapes[i].num_dims; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << netinfo->stages[0].output_shapes[i].dims[j];
        }
        std::cout << "]" << std::endl;
    }

    m_ts = &tmp_ts;
}

SileroVAD::~SileroVAD()
{
    if (bmrt != NULL) {
        bmrt_destroy(bmrt);
        bmrt = NULL;
    }
    bm_dev_free(handle);
}

int SileroVAD::forward(const float* x, const float* h, const float* c,
                       float* prob, float* h_new, float* c_new)
{
    bm_tensor_t input_tensors[3];
    bm_tensor_t output_tensors[3];

    bool ok = true;

    // input 0: x [1, 576]
    ok &= bmrt_tensor(&input_tensors[0], bmrt, netinfo->input_dtypes[0],
                      netinfo->stages[0].input_shapes[0]);
    bm_memcpy_s2d(handle, input_tensors[0].device_mem, (void*)x);

    // input 1: h [1, 128]
    ok &= bmrt_tensor(&input_tensors[1], bmrt, netinfo->input_dtypes[1],
                      netinfo->stages[0].input_shapes[1]);
    bm_memcpy_s2d(handle, input_tensors[1].device_mem, (void*)h);

    // input 2: c [1, 128]
    ok &= bmrt_tensor(&input_tensors[2], bmrt, netinfo->input_dtypes[2],
                      netinfo->stages[0].input_shapes[2]);
    bm_memcpy_s2d(handle, input_tensors[2].device_mem, (void*)c);

    if (!ok) {
        std::cout << "bmrt_tensor failed!" << std::endl;
        return -1;
    }

    m_ts->save("inference", 1);
    ok = bmrt_launch_tensor(bmrt, netinfo->name, input_tensors, netinfo->input_num,
                            output_tensors, netinfo->output_num);
    if (!ok) {
        std::cout << "bmrt_launch_tensor failed!" << std::endl;
        return -1;
    }

    auto ret = bm_thread_sync(handle);
    if (ret != BM_SUCCESS) {
        std::cout << "bm_thread_sync failed!" << std::endl;
        return -1;
    }
    m_ts->save("inference", 1);

    // copy outputs
    bm_memcpy_d2s_partial(handle, prob, output_tensors[0].device_mem, 1 * sizeof(float));
    bm_memcpy_d2s_partial(handle, h_new, output_tensors[1].device_mem, STATE_DIM * sizeof(float));
    bm_memcpy_d2s_partial(handle, c_new, output_tensors[2].device_mem, STATE_DIM * sizeof(float));

    // free device memory
    for (int i = 0; i < netinfo->input_num; ++i) {
        bm_free_device(handle, input_tensors[i].device_mem);
    }
    for (int i = 0; i < netinfo->output_num; ++i) {
        bm_free_device(handle, output_tensors[i].device_mem);
    }

    return 0;
}

int SileroVAD::process_audio(const float* audio, int audio_len,
                             std::vector<SpeechSegment>& speeches,
                             std::vector<float>& speech_probs,
                             float threshold,
                             int min_speech_duration_ms,
                             int min_silence_duration_ms,
                             int speech_pad_ms)
{
    speeches.clear();
    speech_probs.clear();

    float context[CONTEXT_SIZE] = {0};
    float h[STATE_DIM] = {0};
    float c[STATE_DIM] = {0};

    int num_frames = (audio_len + NUM_SAMPLES - 1) / NUM_SAMPLES;
    speech_probs.reserve(num_frames);

    for (int pos = 0; pos < audio_len; pos += NUM_SAMPLES) {
        m_ts->save("preprocess", 1);
        // build input frame: [context(64) + chunk(512)]
        float x[NUM_SAMPLES + CONTEXT_SIZE];
        std::memcpy(x, context, CONTEXT_SIZE * sizeof(float));

        int chunk_len = std::min(NUM_SAMPLES, audio_len - pos);
        std::memcpy(x + CONTEXT_SIZE, audio + pos, chunk_len * sizeof(float));
        if (chunk_len < NUM_SAMPLES) {
            std::memset(x + CONTEXT_SIZE + chunk_len, 0,
                        (NUM_SAMPLES - chunk_len) * sizeof(float));
        }
        m_ts->save("preprocess", 1);

        float prob, h_new[STATE_DIM], c_new[STATE_DIM];
        int ret = forward(x, h, c, &prob, h_new, c_new);
        if (ret != 0) return ret;

        speech_probs.push_back(prob);

        // update state and context
        std::memcpy(h, h_new, STATE_DIM * sizeof(float));
        std::memcpy(c, c_new, STATE_DIM * sizeof(float));
        std::memcpy(context, x + NUM_SAMPLES, CONTEXT_SIZE * sizeof(float));
    }

    // Post-process: probabilities → speech segments
    m_ts->save("postprocess", 1);
    int sr = SAMPLE_RATE;
    int min_speech_samples = sr * min_speech_duration_ms / 1000;
    int speech_pad_samples = sr * speech_pad_ms / 1000;
    int min_silence_samples = sr * min_silence_duration_ms / 1000;
    float neg_threshold = std::max(threshold - 0.15f, 0.01f);

    bool triggered = false;
    SpeechSegment current_speech = {0, 0};
    int temp_end = 0;

    for (size_t i = 0; i < speech_probs.size(); ++i) {
        int cur_sample = NUM_SAMPLES * i;

        if (speech_probs[i] >= threshold && temp_end) {
            temp_end = 0;
        }

        if (speech_probs[i] >= threshold && !triggered) {
            triggered = true;
            current_speech.start = cur_sample;
            continue;
        }

        if (speech_probs[i] < neg_threshold && triggered) {
            if (!temp_end) {
                temp_end = cur_sample;
            }
            if (cur_sample - temp_end < min_silence_samples) {
                continue;
            }
            current_speech.end = temp_end;
            if (current_speech.end - current_speech.start > min_speech_samples) {
                speeches.push_back(current_speech);
            }
            current_speech = {0, 0};
            temp_end = 0;
            triggered = false;
        }
    }

    if (triggered) {
        current_speech.end = speech_probs.size() * NUM_SAMPLES;
        if (current_speech.end - current_speech.start > min_speech_samples) {
            speeches.push_back(current_speech);
        }
    }

    // Apply padding
    for (size_t i = 0; i < speeches.size(); ++i) {
        speeches[i].start = std::max(0, speeches[i].start - speech_pad_samples);
        if (i < speeches.size() - 1) {
            int gap = speeches[i + 1].start - speeches[i].end;
            if (gap < 2 * speech_pad_samples) {
                speeches[i].end += gap / 2;
                speeches[i + 1].start -= gap / 2;
            } else {
                speeches[i].end = std::min(audio_len, speeches[i].end + speech_pad_samples);
                speeches[i + 1].start = std::max(0, speeches[i + 1].start - speech_pad_samples);
            }
        } else {
            speeches[i].end = std::min(audio_len, speeches[i].end + speech_pad_samples);
        }
    }
    m_ts->save("postprocess", 1);

    return 0;
}