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

#include "feat/feature-fbank.h"
#include "feat/wave-reader.h"
#include "bmruntime_interface.h"
#include "utils.hpp"

class FBank {
public:
    FBank(int n_mels, float sample_rate)
        : n_mels(n_mels), sample_rate(sample_rate) {}

    kaldi::Matrix<float> operator()(const kaldi::WaveData& wave_data, float dither = 0.0f) {
        kaldi::FbankOptions fbank_opts;
        fbank_opts.mel_opts.num_bins = n_mels;
        fbank_opts.frame_opts.samp_freq = sample_rate;
        fbank_opts.frame_opts.dither = dither;

        kaldi::Fbank fbank(fbank_opts);
        kaldi::Matrix<float> kaldi_feat;
        fbank.ComputeFeatures(wave_data.Data().Row(0), sample_rate, 1.0f, &kaldi_feat);

        return kaldi_feat;
    }

    float getSampleRate() const { return sample_rate; }

private:
    int n_mels;
    float sample_rate;
    bool mean_nor;
};

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

int compute_embedding(const std::string& wav_file, FBank& feature_extractor, void* p_bmrt, float* output, TimeStamp* m_ts) {
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

    // read wav file to wave_data
    kaldi::WaveData wave_data;
    kaldi::Input ki(wav_file);
    wave_data.Read(ki.Stream());

    // check sample rate
    int target_sample_rate = feature_extractor.getSampleRate();
    if (wave_data.SampFreq() != target_sample_rate)
        std::cerr << "[WARNING]: The sample rate of " << wav_file << " is not " << target_sample_rate << ", resampling is needed.\n";

    m_ts->save("Campplus decode_time", max_batch);

    m_ts->save("Campplus preprocess_time", max_batch);

    // extract feat from wave_data
    kaldi::Matrix<float> kaldi_feat = feature_extractor(wave_data);
    int rows = kaldi_feat.NumRows();
    int cols = kaldi_feat.NumCols();

    // compare feat size and model input size
    unsigned int feat_bytes = rows * cols * sizeof(float);
    unsigned int input_bytes = net_info->max_input_bytes[0];
    if (feat_bytes > input_bytes){
        std::cerr << "The current input exceeds the model's maximum allowed input size. Please retry with larger input size parameters for the model." << std::endl;
        exit(1);
    }

    // determine the mode in soc or pcie
    struct bm_misc_info misc_info;
    status = bm_get_misc_info(bm_handle, &misc_info);
    assert(BM_SUCCESS == status);
    bool is_soc = misc_info.pcie_soc_mode == 1;

    // init input tensors
    float *feat = nullptr;
    bm_tensor_t input_tensors[1];
    bmrt_tensor(&input_tensors[0], p_bmrt, BM_FLOAT32, {3, {max_batch, rows, cols}});
    assert(BM_SUCCESS == status);
    if (is_soc) {
        // memory map
        unsigned long long addr;
        status = bm_mem_mmap_device_mem(bm_handle, &input_tensors[0].device_mem, &addr);
        assert(BM_SUCCESS == status);
        feat = (float*)addr;
    } else {
        // alloc memory for s2d
        feat = new float[rows * cols];
    }

    // calculate mean
    float mean;
    for (int col = 0; col < cols; ++col) {
        mean = 0;
        for (int row = 0; row < rows; ++row)
            mean += kaldi_feat(row, col);
        mean /= rows;
        for (int row = 0; row < rows; ++row)
            feat[cols * row + col] = kaldi_feat(row, col) - mean;
    }

    if (is_soc) {
        // flush cpu cache
        status = bm_mem_flush_device_mem(bm_handle, &input_tensors[0].device_mem);
        assert(BM_SUCCESS == status);
    } else {
        // s2d
        status = bm_memcpy_s2d_partial(bm_handle, input_tensors[0].device_mem, feat, bmrt_tensor_bytesize(&input_tensors[0]));
        assert(BM_SUCCESS == status);
        delete[] feat;
        feat = nullptr;
    }

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
