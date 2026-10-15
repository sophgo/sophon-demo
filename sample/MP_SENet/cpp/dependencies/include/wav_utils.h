//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef WAV_UTILS_H
#define WAV_UTILS_H

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <iostream>
#include <limits>

// RIFF/WAVE header (PCM)
struct WavHeader {
    char     riff[4];          // "RIFF"
    uint32_t file_size;
    char     wave[4];          // "WAVE"
    char     fmt[4];           // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

// Minimal PCM WAV reader: mono output, float samples in [-1, 1).
class WavReader {
public:
    WavReader() = default;
    explicit WavReader(const std::string& filename) { load(filename); }

    bool load(const std::string& filename);
    bool is_valid() const { return valid_; }
    uint32_t sample_rate() const { return sample_rate_; }
    uint16_t num_channels() const { return num_channels_; }
    uint16_t bits_per_sample() const { return bits_per_sample_; }
    size_t num_samples() const { return data_.size(); }  // mono samples
    const std::vector<float>& data() const { return data_; }

private:
    bool valid_ = false;
    uint32_t sample_rate_ = 0;
    uint16_t num_channels_ = 0;
    uint16_t bits_per_sample_ = 0;
    std::vector<float> data_;   // mono float in [-1, 1)
};

// Write mono PCM_16 WAV. Clips to [-1, 1].
bool write_wav(const std::string& filename, const float* data, size_t num_samples,
               uint32_t sample_rate);

// Linear-interpolation resample. Matches Python: n_out = int(len * dst/src).
std::vector<float> resample_linear(const std::vector<float>& data,
                                   uint32_t src_sr, uint32_t dst_sr);

#endif // WAV_UTILS_H
