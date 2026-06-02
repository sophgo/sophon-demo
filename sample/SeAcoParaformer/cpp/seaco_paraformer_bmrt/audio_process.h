//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef AUDIO_PROCESS_H
#define AUDIO_PROCESS_H

#include <armadillo>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// FBANK feature extraction (ported from WeNet cpp/include/processor.h)
// ---------------------------------------------------------------------------

arma::fmat fbank(arma::fmat input,
    int num_mel_bins,
    int frame_length,
    int frame_shift,
    int sample_frequency,
    double dither = 0.0,
    double energy_floor = 1.0,
    bool use_power = true,
    bool use_log_fbank = true,
    bool use_signal_log_energy = false);

// ---------------------------------------------------------------------------
// LFR (Low Frame Rate) stacking  (ported from Python _apply_lfr)
// ---------------------------------------------------------------------------

arma::fmat apply_lfr(const arma::fmat& inputs, int lfr_m, int lfr_n);

// ---------------------------------------------------------------------------
// CMVN (Cepstral Mean and Variance Normalization)
// ---------------------------------------------------------------------------

struct CMVN {
    arma::fvec means;
    arma::fvec vars;
};

CMVN load_cmvn(const std::string& path);

void apply_cmvn(arma::fmat& feats, const CMVN& cmvn);

#endif // AUDIO_PROCESS_H
