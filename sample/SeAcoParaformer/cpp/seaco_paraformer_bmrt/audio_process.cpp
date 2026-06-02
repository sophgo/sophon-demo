//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "audio_process.h"

static int next_power_of_2(int x) {
    if (x <= 0) return 1;
    x--;
    x |= x >> 1; x |= x >> 2;
    x |= x >> 4; x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

static arma::fvec hamming_window(int window_size) {
    arma::fvec w(window_size);
    for (int i = 0; i < window_size; i++) {
        w(i) = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (window_size - 1));
    }
    return w;
}

static double mel_scale_scalar(double freq) {
    return 1127.0 * std::log(1.0 + freq / 700.0);
}

static arma::fmat get_mel_banks(int num_bins, int window_length_padded,
                                 double sample_freq, double low_freq, double high_freq) {
    int num_fft_bins = window_length_padded / 2;
    double low_mel = mel_scale_scalar(low_freq);
    double high_mel = mel_scale_scalar(high_freq);
    arma::fvec mel_points = arma::linspace<arma::fvec>(low_mel, high_mel, num_bins + 2);
    arma::fmat mel_weights(num_bins, num_fft_bins, arma::fill::zeros);
    for (int i = 0; i < num_bins; i++) {
        double left_mel = mel_points(i), center_mel = mel_points(i + 1), right_mel = mel_points(i + 2);
        for (int j = 0; j < num_fft_bins; j++) {
            double mel = mel_scale_scalar(j * sample_freq / window_length_padded);
            if (mel > left_mel && mel < center_mel)
                mel_weights(i, j) = (mel - left_mel) / (center_mel - left_mel);
            else if (mel > center_mel && mel < right_mel)
                mel_weights(i, j) = (right_mel - mel) / (right_mel - center_mel);
        }
    }
    return mel_weights;
}

static arma::fmat real_fft_power(const arma::fmat& signal, int n_fft) {
    int n_rows = signal.n_rows, n_cols = signal.n_cols, n_out = n_fft / 2;
    arma::fmat result(n_rows, n_out, arma::fill::zeros);
    for (int r = 0; r < n_rows; r++) {
        for (int k = 0; k < n_out; k++) {
            double real_sum = 0.0, imag_sum = 0.0;
            for (int t = 0; t < n_cols; t++) {
                double theta = 2.0 * M_PI * k * t / n_fft;
                double val = signal(r, t);
                real_sum += val * std::cos(theta);
                imag_sum += val * std::sin(theta);
            }
            result(r, k) = std::sqrt(real_sum * real_sum + imag_sum * imag_sum) / std::sqrt((double)n_fft);
        }
    }
    return result;
}

arma::fmat fbank(arma::fmat input, int num_mel_bins, int frame_length_samples,
                  int frame_shift_samples, int sample_frequency,
                  double dither, double energy_floor, bool use_power,
                  bool use_log_fbank, bool use_signal_log_energy) {
    int num_samples = input.n_cols;
    arma::fvec waveform(num_samples);
    for (int i = 0; i < num_samples; i++) waveform(i) = input(0, i);

    double preemph = 0.97;
    for (int i = num_samples - 1; i > 0; i--)
        waveform(i) -= preemph * waveform(i - 1);

    int frame_length_padded = next_power_of_2(frame_length_samples);
    arma::fvec window = hamming_window(frame_length_samples);
    arma::fvec window_padded(frame_length_padded, arma::fill::zeros);
    for (int i = 0; i < frame_length_samples; i++) window_padded(i) = window(i);

    int num_frames = std::max(0, (num_samples - frame_length_samples) / frame_shift_samples + 1);
    if (num_frames == 0) num_frames = 1;

    arma::fmat frames(num_frames, frame_length_samples, arma::fill::zeros);
    for (int i = 0; i < num_frames; i++) {
        int start = i * frame_shift_samples;
        for (int j = 0; j < frame_length_samples && (start + j) < num_samples; j++)
            frames(i, j) = waveform(start + j) * window(j);
    }

    arma::fmat power_spectrum = real_fft_power(frames, frame_length_padded);
    arma::fmat mel_banks = get_mel_banks(num_mel_bins, frame_length_padded, sample_frequency, 20.0, sample_frequency / 2.0);

    arma::fmat mel_feats(num_frames, num_mel_bins, arma::fill::zeros);
    int n_fft_bins = power_spectrum.n_cols;
    for (int i = 0; i < num_frames; i++)
        for (int m = 0; m < num_mel_bins; m++) {
            double sum = 0.0;
            for (int j = 0; j < n_fft_bins; j++) sum += power_spectrum(i, j) * mel_banks(m, j);
            mel_feats(i, m) = sum;
        }

    if (!use_power) mel_feats = arma::sqrt(mel_feats);
    if (use_log_fbank) {
        float eps = std::numeric_limits<float>::epsilon();
        for (arma::uword i = 0; i < mel_feats.n_elem; i++)
            mel_feats(i) = std::log(std::max(mel_feats(i), eps));
    }
    return mel_feats;
}

arma::fmat apply_lfr(const arma::fmat& inputs, int lfr_m, int lfr_n) {
    int T = inputs.n_rows, D = inputs.n_cols;
    int T_lfr = (int)std::ceil((float)T / lfr_n);
    int pad_left = (lfr_m - 1) / 2;
    arma::fmat padded = inputs;
    for (int p = 0; p < pad_left; p++)
        padded.insert_rows(0, inputs.row(0));

    int last_idx = (padded.n_rows - lfr_m) / lfr_n;
    int num_pad = (2 * lfr_m - 2 * (int)padded.n_rows + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx);
    if (num_pad > 0) {
        for (int p = 0; p < num_pad; p++)
            padded.insert_rows(padded.n_rows, padded.row(padded.n_rows - 1));
    }

    arma::fmat result(T_lfr, lfr_m * D, arma::fill::zeros);
    for (int i = 0; i < T_lfr; i++) {
        for (int m = 0; m < lfr_m; m++) {
            int row_idx = i * lfr_n + m;
            if (row_idx < (int)padded.n_rows) {
                for (int d = 0; d < D; d++)
                    result(i, m * D + d) = padded(row_idx, d);
            }
        }
    }
    return result;
}

CMVN load_cmvn(const std::string& path) {
    CMVN cmvn;
    std::ifstream file(path);
    if (!file.is_open()) { cmvn.means.set_size(0); cmvn.vars.set_size(0); return cmvn; }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) lines.push_back(line);
    file.close();

    std::vector<float> means_l, vars_l;
    for (size_t i = 0; i < lines.size(); i++) {
        std::istringstream iss(lines[i]);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) tokens.push_back(token);
        if (tokens.empty()) continue;

        if (tokens[0] == "<AddShift>" && i + 1 < lines.size()) {
            std::istringstream niss(lines[i + 1]);
            std::vector<std::string> ni;
            while (niss >> token) ni.push_back(token);
            if (!ni.empty() && ni[0] == "<LearnRateCoef>")
                for (size_t k = 3; k + 1 < ni.size(); k++)
                    means_l.push_back(std::stof(ni[k]));
        } else if (tokens[0] == "<Rescale>" && i + 1 < lines.size()) {
            std::istringstream niss(lines[i + 1]);
            std::vector<std::string> ni;
            while (niss >> token) ni.push_back(token);
            if (!ni.empty() && ni[0] == "<LearnRateCoef>")
                for (size_t k = 3; k + 1 < ni.size(); k++)
                    vars_l.push_back(std::stof(ni[k]));
        }
    }

    if (!means_l.empty()) { cmvn.means.set_size(means_l.size()); for (size_t i = 0; i < means_l.size(); i++) cmvn.means(i) = means_l[i]; }
    else cmvn.means.set_size(0);
    if (!vars_l.empty()) { cmvn.vars.set_size(vars_l.size()); for (size_t i = 0; i < vars_l.size(); i++) cmvn.vars(i) = vars_l[i]; }
    else cmvn.vars.set_size(0);
    return cmvn;
}

void apply_cmvn(arma::fmat& feats, const CMVN& cmvn) {
    if (cmvn.means.n_elem == 0 || cmvn.vars.n_elem == 0) return;
    int D = feats.n_cols, d = std::min(D, (int)cmvn.means.n_elem);
    for (int i = 0; i < d; i++)
        for (int t = 0; t < (int)feats.n_rows; t++)
            feats(t, i) = (feats(t, i) + cmvn.means(i)) * cmvn.vars(i);
}
