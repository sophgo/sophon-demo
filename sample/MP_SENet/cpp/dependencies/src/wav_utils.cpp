//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "wav_utils.h"
#include <cmath>

bool WavReader::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Could not open WAV: " << filename << std::endl;
        return false;
    }

    WavHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(WavHeader));
    if (strncmp(hdr.riff, "RIFF", 4) != 0 || strncmp(hdr.wave, "WAVE", 4) != 0) {
        std::cerr << "Invalid WAV header: " << filename << std::endl;
        return false;
    }

    sample_rate_     = hdr.sample_rate;
    num_channels_    = hdr.num_channels;
    bits_per_sample_ = hdr.bits_per_sample;

    // Skip to "data" chunk
    char chunk_id[4];
    uint32_t chunk_size;
    do {
        in.read(chunk_id, 4);
        in.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (strncmp(chunk_id, "data", 4) != 0)
            in.seekg(chunk_size, std::ios::cur);
    } while (strncmp(chunk_id, "data", 4) != 0);

    // chunk_size = bytes of raw PCM audio data
    size_t bytes_per_sample = bits_per_sample_ / 8;
    size_t frame_size       = bytes_per_sample * num_channels_;
    size_t num_frames       = chunk_size / frame_size;

    std::vector<char> raw(chunk_size);
    in.read(raw.data(), chunk_size);
    in.close();

    if (bits_per_sample_ == 16) {
        const int16_t* p = reinterpret_cast<const int16_t*>(raw.data());
        const float scale = 1.0f / 32768.0f;
        for (size_t i = 0; i < num_frames; ++i) {
            // Take channel 0 (if multi-channel, take first channel)
            data_.push_back(static_cast<float>(p[i * num_channels_]) * scale);
        }
    } else if (bits_per_sample_ == 32) {
        const int32_t* p = reinterpret_cast<const int32_t*>(raw.data());
        const float scale = 1.0f / 2147483648.0f;
        for (size_t i = 0; i < num_frames; ++i) {
            data_.push_back(static_cast<float>(p[i * num_channels_]) * scale);
        }
    } else {
        std::cerr << "Unsupported bits_per_sample: " << bits_per_sample_ << std::endl;
        return false;
    }

    valid_ = true;
    return true;
}

bool write_wav(const std::string& filename, const float* data, size_t num_samples,
               uint32_t sample_rate) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Could not open for writing: " << filename << std::endl;
        return false;
    }

    uint16_t num_channels    = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate       = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align     = num_channels * bits_per_sample / 8;
    uint32_t data_size       = num_samples * block_align;
    uint32_t file_size       = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&file_size), 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    uint32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    uint16_t audio_format = 1;  // PCM
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&num_channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);

    std::vector<int16_t> pcm(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float v = data[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = static_cast<int16_t>(v * 32767.0f);
    }
    out.write(reinterpret_cast<const char*>(pcm.data()), data_size);
    out.close();
    return true;
}

std::vector<float> resample_linear(const std::vector<float>& data,
                                   uint32_t src_sr, uint32_t dst_sr) {
    if (src_sr == dst_sr) return data;

    size_t n_in  = data.size();
    double ratio = static_cast<double>(dst_sr) / static_cast<double>(src_sr);
    size_t n_out = static_cast<size_t>(n_in * ratio);

    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        double idx = static_cast<double>(i) / ratio;
        size_t low = static_cast<size_t>(std::floor(idx));
        size_t high = (low + 1 < n_in) ? low + 1 : n_in - 1;
        double frac = idx - static_cast<double>(low);
        out[i] = static_cast<float>(data[low] * (1.0 - frac) + data[high] * frac);
    }
    return out;
}
