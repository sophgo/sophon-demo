//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <sndfile.h>
#include <armadillo>
#include <string>
#include <cassert>
#include "processor.h"
#include "bmruntime_cpp.h"
#include "utils.hpp"
#include "util.h"

class WeNet {
    int sample_frequency;
    int num_mel_bins;
    int frame_shift;
    int frame_length;
    int decoding_chunk_size;
    int subsampling_rate;
    int context;
    std::vector<std::string> dict;

    int batch_size;
    int out_size;
    int beam_size;
    int out_length;
    int max_len;

    const char* model;
    std::shared_ptr<bmruntime::Context> encoder_ctx;
    std::shared_ptr<bmruntime::Context> decoder_ctx;
    std::shared_ptr<bmruntime::Network> encoder_net;
    std::shared_ptr<bmruntime::Network> decoder_net;
    std::vector<bmruntime::Tensor *> encoder_inputs;
    std::vector<bmruntime::Tensor *> encoder_outputs;
    std::vector<bmruntime::Tensor *> decoder_inputs;
    std::vector<bmruntime::Tensor *> decoder_outputs;
    std::string result;
    std::string mode;

    arma::fmat feats;
    TimeStamp *m_ts;

    int pre_process(const char* file_path);
    int inference();

    public:
    
    WeNet(std::shared_ptr<bmruntime::Context> encoder_ctx, std::shared_ptr<bmruntime::Context> decoder_ctx): encoder_ctx(encoder_ctx), decoder_ctx(decoder_ctx) {};
    int Init(const std::vector<std::string>& dict, int sample_frequency, int num_mel_bins, int frame_shift, int frame_length, int decoding_chunk_size, int subsampling_rate, int context, const std::string& mode);
    std::string Recognize(const char* file_path);
    void enableProfile(TimeStamp *ts);
};