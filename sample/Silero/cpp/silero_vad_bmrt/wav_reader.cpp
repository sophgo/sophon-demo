//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <string>

struct WavHeader {
    char riff[4];           // "RIFF"
    uint32_t file_size;
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;  // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

static bool read_wav_header(FILE* fp, WavHeader& header, uint32_t& data_size) {
    if (fread(&header, sizeof(header), 1, fp) != 1) return false;

    if (std::memcmp(header.riff, "RIFF", 4) != 0 ||
        std::memcmp(header.wave, "WAVE", 4) != 0 ||
        std::memcmp(header.fmt, "fmt ", 4) != 0) {
        return false;
    }

    // Skip extra fmt bytes if any
    if (header.fmt_size > 16) {
        fseek(fp, header.fmt_size - 16, SEEK_CUR);
    }

    // Find "data" chunk
    char chunk_id[4];
    while (fread(chunk_id, 1, 4, fp) == 4) {
        fread(&data_size, 4, 1, fp);
        if (std::memcmp(chunk_id, "data", 4) == 0) {
            return true;
        }
        fseek(fp, data_size, SEEK_CUR);
    }
    return false;
}

float* read_wav(const std::string& path, int& num_samples, int& sample_rate) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open WAV file: %s\n", path.c_str());
        return nullptr;
    }

    WavHeader header;
    uint32_t data_size;
    if (!read_wav_header(fp, header, data_size)) {
        fprintf(stderr, "Invalid WAV file: %s\n", path.c_str());
        fclose(fp);
        return nullptr;
    }

    sample_rate = header.sample_rate;
    int bytes_per_sample = header.bits_per_sample / 8;
    num_samples = data_size / bytes_per_sample;

    float* audio = new float[num_samples];
    assert(audio != nullptr);

    if (bytes_per_sample == 2) {
        int16_t* buf = new int16_t[num_samples];
        size_t n = fread(buf, bytes_per_sample, num_samples, fp);
        fclose(fp);
        for (int i = 0; i < (int)n; ++i) {
            audio[i] = buf[i] / 32768.0f;
        }
        num_samples = n;
        delete[] buf;
    } else if (bytes_per_sample == 4) {
        int32_t* buf = new int32_t[num_samples];
        size_t n = fread(buf, bytes_per_sample, num_samples, fp);
        fclose(fp);
        for (int i = 0; i < (int)n; ++i) {
            audio[i] = buf[i] / 2147483648.0f;
        }
        num_samples = n;
        delete[] buf;
    } else {
        fclose(fp);
        fprintf(stderr, "Unsupported WAV bit depth: %d\n", header.bits_per_sample);
        delete[] audio;
        return nullptr;
    }

    return audio;
}

bool save_wav(const std::string& path, const float* audio, int num_samples, int sample_rate) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;

    uint32_t data_size = num_samples * 2;
    uint32_t file_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, fp);
    fwrite(&file_size, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);

    // fmt chunk
    fwrite("fmt ", 1, 4, fp);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, fp);
    uint16_t audio_format = 1;
    fwrite(&audio_format, 2, 1, fp);
    uint16_t num_channels = 1;
    fwrite(&num_channels, 2, 1, fp);
    fwrite(&sample_rate, 4, 1, fp);
    uint32_t byte_rate = sample_rate * num_channels * 2;
    fwrite(&byte_rate, 4, 1, fp);
    uint16_t block_align = num_channels * 2;
    fwrite(&block_align, 2, 1, fp);
    uint16_t bits_per_sample = 16;
    fwrite(&bits_per_sample, 2, 1, fp);

    // data chunk
    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);

    // write samples as int16
    for (int i = 0; i < num_samples; ++i) {
        float clipped = std::max(-1.0f, std::min(1.0f, audio[i]));
        int16_t sample = (int16_t)(clipped * 32767.0f);
        fwrite(&sample, 2, 1, fp);
    }

    fclose(fp);
    return true;
}