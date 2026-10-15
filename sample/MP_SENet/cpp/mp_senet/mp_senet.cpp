//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "mp_senet.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

MpSenet::~MpSenet() {
    if (p_bmrt_) {
        bmrt_destroy(p_bmrt_);
    }
    if (bm_handle_) {
        bm_dev_free(bm_handle_);
    }
}

int MpSenet::Init() {
    bm_status_t status = bm_dev_request(&bm_handle_, cfg_.dev_id);
    if (status != BM_SUCCESS) {
        std::cerr << "Failed to request device " << cfg_.dev_id << std::endl;
        return -1;
    }

    p_bmrt_ = bmrt_create(bm_handle_);
    if (!p_bmrt_) {
        std::cerr << "Failed to create bmrt" << std::endl;
        return -1;
    }

    bool ret = bmrt_load_bmodel(p_bmrt_, cfg_.bmodel_path.c_str());
    if (!ret) {
        std::cerr << "Failed to load bmodel: " << cfg_.bmodel_path << std::endl;
        return -1;
    }
    std::cout << "Loaded bmodel: " << cfg_.bmodel_path << std::endl;

    const char** net_names = nullptr;
    bmrt_get_network_names(p_bmrt_, &net_names);
    net_name_ = net_names[0];  // keep pointer; do not free
    free(net_names);

    net_info_ = bmrt_get_network_info(p_bmrt_, net_name_);
    assert(net_info_ != nullptr);

    // max_length = input shape dim[2] (e.g. 640)
    max_length_ = net_info_->stages[0].input_shapes[0].dims[2];
    max_wav_length_ = (max_length_ - 1) * cfg_.hop_size + (cfg_.hop_size - 1);

    std::cout << "Network: " << net_name_
              << "  max_length=" << max_length_
              << "  max_wav_length=" << max_wav_length_
              << "  compress_factor=" << cfg_.compress_factor
              << "  hop_size=" << cfg_.hop_size
              << std::endl;

    return 0;
}

// Compute magnitude/phase STFT for a wav chunk.
// wav:     [1, wav_L]  (row-major, batch=1)
// output:  amp [1, freq, T], pha [1, freq, T]  where freq = n_fft/2+1 = 201
int MpSenet::mag_pha_stft(const float* wav, int wav_L,
                          std::vector<float>& amp, std::vector<float>& pha, int& T) {
    int freq = cfg_.n_fft / 2 + 1;   // 201
    // T = 1 + L / hop  (center=True, reflect-padded STFT)
    T = 1 + wav_L / cfg_.hop_size;

    int out_size = freq * T;  // batch=1
    std::vector<float> stft_R(out_size), stft_I(out_size);

    // bmcv_stft(handle, XR, XI, YR, YI, batch, L, realInput, pad_mode,
    //           n_fft, win_mode, hop_len, normalize)
    // Python binding order differs from C:
    //   py: stft(XR, XI, realInput, normalize, n_fft, hop_len, pad_mode, win_mode)
    //   →  C: realInput=true, pad_mode=1(constant), n_fft, win_mode=0(hanning),
    //         hop_len, normalize=false
    bm_status_t st = bmcv_stft(bm_handle_,
                               const_cast<float*>(wav), const_cast<float*>(wav),
                               stft_R.data(), stft_I.data(),
                               1, wav_L, true, 1,
                               cfg_.n_fft, 0, cfg_.hop_size, false);
    if (st != BM_SUCCESS) {
        std::cerr << "bmcv_stft failed: " << st << std::endl;
        return -1;
    }

    amp.resize(out_size);
    pha.resize(out_size);
    float cf = cfg_.compress_factor;

    for (int i = 0; i < out_size; ++i) {
        float re = stft_R[i];
        float im = stft_I[i];
        float mag = std::sqrt(re * re + im * im + 1e-9f);
        float theta = std::atan2(im + 1e-10f, re + 1e-5f);
        amp[i] = std::pow(mag, cf);
        pha[i] = theta;
    }
    return 0;
}

// Reconstruct wav from amp_g/pha_g [1, freq, real_len].
// Output: wav_out [wav_L]
int MpSenet::mag_pha_istft(const float* amp_g, const float* pha_g, int real_len,
                           int wav_L, std::vector<float>& wav_out) {
    int freq = cfg_.n_fft / 2 + 1;   // 201
    int spec_size = freq * real_len;
    std::vector<float> real_part(spec_size), imag_part(spec_size);
    float inv_cf = 1.0f / cfg_.compress_factor;

    for (int i = 0; i < spec_size; ++i) {
        float mag = std::pow(amp_g[i], inv_cf);
        real_part[i] = mag * std::cos(pha_g[i]);
        imag_part[i] = mag * std::sin(pha_g[i]);
    }

    // bmcv_istft(handle, XR, XI, YR, YI, batch, L, realInput, pad_mode,
    //            n_fft, win_mode, hop_len, normalize)
    // Same parameter mapping as stft: pad_mode=1, win_mode=0
    wav_out.assign(wav_L, 0.0f);
    std::vector<float> yr(wav_L), yi(wav_L);

    bm_status_t st = bmcv_istft(bm_handle_,
                                real_part.data(), imag_part.data(),
                                yr.data(), yi.data(),
                                1, wav_L, true, 1,
                                cfg_.n_fft, 0, cfg_.hop_size, false);
    if (st != BM_SUCCESS) {
        std::cerr << "bmcv_istft failed: " << st << std::endl;
        return -1;
    }
    // wav[0] = real part of reconstructed signal
    for (int i = 0; i < wav_L; ++i)
        wav_out[i] = yr[i];

    return 0;
}

int MpSenet::process(const std::string& wav_file, const std::string& output_file) {
    // ── 1. Read WAV ──
    WavReader reader(wav_file);
    if (!reader.is_valid()) {
        std::cerr << "Failed to read WAV: " << wav_file << std::endl;
        return -1;
    }
    std::cout << "Input: " << wav_file
              << "  sr=" << reader.sample_rate()
              << "  samples=" << reader.num_samples() << std::endl;

    // ── 2. Resample to 16 kHz ──
    std::vector<float> noisy_wav = reader.data();
    if (reader.sample_rate() != static_cast<uint32_t>(cfg_.sample_rate)) {
        noisy_wav = resample_linear(noisy_wav, reader.sample_rate(), cfg_.sample_rate);
        std::cout << "Resampled " << reader.sample_rate() << " → "
                  << cfg_.sample_rate << "Hz, " << noisy_wav.size() << " samples" << std::endl;
    }

    // ── 3. Normalize ──
    double sum_sq = 0.0;
    for (float v : noisy_wav) sum_sq += (double)v * v;
    norm_factor_ = std::sqrt((double)noisy_wav.size() / sum_sq);
    for (float& v : noisy_wav) v *= norm_factor_;

    // ── 4. Chunk into max_wav_length windows ──
    int freq = cfg_.n_fft / 2 + 1;  // 201
    std::vector<Chunk> chunks;
    noisy_end_len_ = 0;

    for (size_t start = 0; start < noisy_wav.size(); start += max_wav_length_) {
        int chunk_L = std::min((size_t)max_wav_length_, noisy_wav.size() - start);
        if (chunk_L < cfg_.n_fft) continue;  // skip too-short chunks

        const float* wav_ptr = noisy_wav.data() + start;
        std::vector<float> amp, pha;
        int T;
        if (mag_pha_stft(wav_ptr, chunk_L, amp, pha, T) != 0) return -1;
        noisy_end_len_ = T;

        // Pad to max_length if needed
        if (T < max_length_) {
            int pad_width = (int)(max_length_ * 0.1);  // 64
            // add = [zeros(pad_width), amp]  → length T+pad_width
            int add_len = T + pad_width;
            std::vector<float> add_amp(add_len, 0.0f), add_pha(add_len, 0.0f);
            // add[pad_width..] = amp[0..T-1]
            std::copy(amp.begin(), amp.end(), add_amp.begin() + pad_width);
            std::copy(pha.begin(), pha.end(), add_pha.begin() + pad_width);

            int repeat_n = (int)std::ceil((double)(max_length_ - T) / add_len);
            std::vector<float> padded_amp(amp), padded_pha(pha);
            for (int r = 0; r < repeat_n; ++r) {
                size_t old = padded_amp.size();
                padded_amp.resize(old + add_len);
                padded_pha.resize(old + add_len);
                std::copy(add_amp.begin(), add_amp.end(), padded_amp.begin() + old);
                std::copy(add_pha.begin(), add_pha.end(), padded_pha.begin() + old);
            }
            amp.assign(padded_amp.begin(), padded_amp.begin() + max_length_);
            pha.assign(padded_pha.begin(), padded_pha.begin() + max_length_);
        }

        Chunk ch;
        ch.amp = std::move(amp);
        ch.pha = std::move(pha);
        ch.wav_stft_L = chunk_L;
        chunks.push_back(std::move(ch));
    }

    std::cout << chunks.size() << " chunks, stft T=" << noisy_end_len_ << std::endl;
    if (chunks.empty()) {
        std::cerr << "No valid chunks to process" << std::endl;
        return -1;
    }

    // ── 5. Inference + postprocess ──
    int input_bytes  = freq * max_length_ * sizeof(float);   // 201*640*4
    int output_bytes = freq * max_length_ * sizeof(float);

    std::vector<float> clean_wav_all;

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        Chunk& ch = chunks[ci];

        // Create input tensors on device
        bm_tensor_t in_tensors[2];
        bm_shape_t in_shape;
        int in_dims[4] = {1, freq, max_length_};
        bmrt_shape(&in_shape, in_dims, 3);
        bmrt_tensor(&in_tensors[0], p_bmrt_, BM_FLOAT32, in_shape);
        bmrt_tensor(&in_tensors[1], p_bmrt_, BM_FLOAT32, in_shape);

        bm_status_t st;
        st = bm_memcpy_s2d_partial(bm_handle_, in_tensors[0].device_mem,
                                  ch.amp.data(), input_bytes);
        assert(st == BM_SUCCESS);
        st = bm_memcpy_s2d_partial(bm_handle_, in_tensors[1].device_mem,
                                  ch.pha.data(), input_bytes);
        assert(st == BM_SUCCESS);

        // Output tensors: model has 3 outputs (amp_g, pha_g, com_g);
        // we use the first 2, but must provide all output tensors to launch.
        int out_dims[3] = {1, freq, max_length_};
        bm_shape_t out_shape;
        bmrt_shape(&out_shape, out_dims, 3);
        int n_outputs = net_info_->output_num;  // 3
        std::vector<bm_tensor_t> out_tensors(n_outputs);
        for (int oi = 0; oi < n_outputs; ++oi)
            bmrt_tensor(&out_tensors[oi], p_bmrt_, BM_FLOAT32, out_shape);

        if (ts_) ts_->save("mp_senet inference");

        bool ret = bmrt_launch_tensor_ex(p_bmrt_, net_name_,
                                          in_tensors, 2,
                                          out_tensors.data(), n_outputs, true, false);
        assert(ret);

        st = bm_thread_sync(bm_handle_);
        assert(st == BM_SUCCESS);

        if (ts_) ts_->save("mp_senet inference");

        // Copy outputs back (first 2 outputs: amp_g, pha_g)
        std::vector<float> amp_g(freq * max_length_), pha_g(freq * max_length_);
        bm_memcpy_d2s_partial(bm_handle_, amp_g.data(), out_tensors[0].device_mem, output_bytes);
        bm_memcpy_d2s_partial(bm_handle_, pha_g.data(), out_tensors[1].device_mem, output_bytes);

        // Free device memory
        bm_free_device(bm_handle_, in_tensors[0].device_mem);
        bm_free_device(bm_handle_, in_tensors[1].device_mem);
        for (int oi = 0; oi < n_outputs; ++oi)
            bm_free_device(bm_handle_, out_tensors[oi].device_mem);

        // ── Postprocess: istft ──
        int real_len = (ci == chunks.size() - 1) ? noisy_end_len_ : max_length_;
        std::vector<float> wav_out;
        if (mag_pha_istft(amp_g.data(), pha_g.data(), real_len,
                          ch.wav_stft_L, wav_out) != 0)
            return -1;

        // Denormalize
        for (float& v : wav_out) v /= norm_factor_;
        clean_wav_all.insert(clean_wav_all.end(), wav_out.begin(), wav_out.end());
    }

    // ── 6. Write output WAV ──
    if (!write_wav(output_file, clean_wav_all.data(), clean_wav_all.size(),
                   cfg_.sample_rate)) {
        std::cerr << "Failed to write output WAV" << std::endl;
        return -1;
    }
    std::cout << "Wrote: " << output_file
              << " (" << clean_wav_all.size() << " samples)" << std::endl;

    return 0;
}
