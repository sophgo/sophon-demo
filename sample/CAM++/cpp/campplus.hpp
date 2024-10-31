//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef CAMPPLUS_H
#define CAMPPLUS_H

#include <cassert>
#include <memory>

#include "bmruntime_interface.h"
#include "utils.hpp"
#include "utils/wav_reader.h"
#include "feature/feature_fbank.h"

float cosine_similarity(float* emb1, float* emb2, unsigned long size, float eps = 1e-6) {
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        dot_product += emb1[i] * emb2[i];
        norm1 += emb1[i] * emb1[i];
        norm2 += emb2[i] * emb2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    return dot_product / (norm1 * norm2 + eps);
}

int compute_embedding(const std::string& wav_file, std::shared_ptr<speakerlab::FbankComputer> fbank_computer, void* p_bmrt, float* output, TimeStamp* m_ts) {
    // get net_names and net_info
    const char **net_names = NULL;
    bmrt_get_network_names(p_bmrt, &net_names);

    const bm_net_info_t *net_info = bmrt_get_network_info(p_bmrt, net_names[0]);
    assert(NULL != net_info);

    // get bm_handle and init status
    bm_handle_t bm_handle = (bm_handle_t)bmrt_get_bm_handle (p_bmrt);
    bm_status_t status;
    bool ret;

    // get max_batch
    int max_batch = net_info->stages[0].input_shapes[0].dims[0];

    m_ts->save("Campplus decode_time", max_batch);

    // read wav_file
    speakerlab::WavReader wav_reader(wav_file);

    // check sample rate
    int target_sample_rate = 16000;
    if (wav_reader.sample_rate() != target_sample_rate)
        std::cerr << "[WARNING]: The sample rate of " << wav_file << " is not " << target_sample_rate << ", resampling is needed.\n";

    m_ts->save("Campplus decode_time", max_batch);

    m_ts->save("Campplus preprocess_time", max_batch);

    // extract feat from wave_data
    speakerlab::Feature feature = fbank_computer -> compute_feature(wav_reader);
    speakerlab::subtract_feature_mean(feature);

    int rows = feature.size();
    int cols = feature[0].size();

    // compare feat size and model input size
    unsigned int feat_bytes = rows * cols * sizeof(float);
    unsigned int input_bytes = net_info->max_input_bytes[0];
    if (feat_bytes > input_bytes){
        std::cerr << "The current input exceeds the model's maximum allowed input size."
                  << "Please retry with larger input size parameters for the model." << std::endl;
        exit(1);
    }

    // init input tensors
    float feat[rows * cols];
    bm_tensor_t input_tensors[1];
    bmrt_tensor(&input_tensors[0], p_bmrt, BM_FLOAT32, {3, {max_batch, rows, cols}});
    assert(BM_SUCCESS == status);

    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col) {
            feat[cols*row+col] = feature[row][col];
    }

    // s2d
    status = bm_memcpy_s2d_partial(bm_handle, input_tensors[0].device_mem, feat, bmrt_tensor_bytesize(&input_tensors[0]));
    assert(BM_SUCCESS == status);

    // init output tensors
    bm_tensor_t output_tensors[1];
    status = bm_malloc_device_byte(bm_handle, &output_tensors[0].device_mem, net_info->max_output_bytes[0]);
    assert(BM_SUCCESS == status);

    m_ts->save("Campplus preprocess_time", max_batch);

    m_ts->save("Campplus inference", max_batch);

    // forward
    ret = bmrt_launch_tensor_ex(p_bmrt, net_names[0], input_tensors, 1, output_tensors, 1, true, false);
    assert(true == ret);

    // sync, wait for finishing inference
    status = bm_thread_sync(bm_handle);
    assert(status == BM_SUCCESS);

    m_ts->save("Campplus inference", max_batch);

    m_ts->save("Campplus postprocess_time", max_batch);

    status = bm_memcpy_d2s_partial(bm_handle, output, output_tensors[0].device_mem, bmrt_tensor_bytesize(&output_tensors[0]));
    assert(BM_SUCCESS == status);

    m_ts->save("Campplus postprocess_time", max_batch);

    // at last, free device memory
    for (int i = 0; i < net_info->input_num; ++i)
        bm_free_device(bm_handle, input_tensors[i].device_mem);
    for (int i = 0; i < net_info->output_num; ++i)
        bm_free_device(bm_handle, output_tensors[i].device_mem);
    free(net_names);

    return 0;
}

#endif //!CAMPPLUS_H
