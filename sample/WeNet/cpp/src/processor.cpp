//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "processor.h"
#include "wrapper.h"
#include "bmruntime_cpp.h"
#include <chrono>
#define MILLISECONDS_TO_SECONDS 0.001

using namespace bmruntime;

// todo
arma::fmat resample(arma::fmat input, int sample_rate, int resample_rate) {
    return input; 
}

int bit_length(int n) {
    // Find the position of the most significant bit set
    int position = 0;
    while (n) {
        position++;
        n >>= 1;
    }

    return position;
}

int next_power_of_2(int x) {
    return (x == 0 ? 1 : std::pow(2.0, (double)bit_length(x - 1)));
}

double get_epsilon() {
    return 1.1921e-07;
}
arma::vec get_waveform_and_window_properties(arma::fmat& input, int channel, int sample_frequency, int frame_shift, int frame_length, bool round_to_power_of_two = true, double preemphasis_coefficient = 0.97) {
    channel = std::max(channel, 0);
    assert(channel < static_cast<int>(input.n_rows) && "Invalid channel num!");
    input = input.submat(channel, 0, channel, input.n_cols - 1);
    
    int window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS);
    int window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS);
    int padded_window_size = round_to_power_of_two ? next_power_of_2(window_size) : window_size;
    arma::vec res = {static_cast<double>(window_shift), static_cast<double>(window_size), static_cast<double>(padded_window_size)};
    return res;
}

void get_strided(arma::fmat& input, int window_size, int window_shift, bool snip_edges) {
    int num_samples = static_cast<int>(input.n_cols);
    int stride = 1; // tensor.stride(0)
    int m;
    if(snip_edges) {
        assert(num_samples >= window_size && "num_samples can't greater than window_size!");
        m = 1 + floor(((double)num_samples - (double)window_size) / (double)window_shift);
    }
    else {
        std::cout << "todo!!!" << std::endl;
        exit(0);
    }
    input = as_strided(input, m, window_size, window_shift * stride, stride);
}

arma::fmat get_log_energy(const arma::fmat& input, double epsilon, double energy_floor) {
    // arma::fmat log_energy = arma::log(arma::max(sum(square(input), 1), arma::fmat(input.n_rows, 1, arma::fill::value(epsilon))));
    // if (energy_floor == 0.0) {
    //     return log_energy;
    // }
    // return arma::max(log_energy, arma::fmat(log_energy.n_rows, log_energy.n_cols, arma::fill::value(log(epsilon))));
    arma::fmat log_energy = arma::log(arma::max(sum(square(input), 1), epsilon * arma::fmat(input.n_rows, 1, arma::fill::ones)));
    if (energy_floor == 0.0) {
        return log_energy;
    }
    return arma::max(log_energy, log(epsilon) * arma::fmat(size(log_energy), arma::fill::ones));
}

arma::fmat feature_window_function(const std::string& window_type, int window_size, double blackman_coeff) {
    if(window_type == "povey") {
        return hann_window(window_size);
    }
    else {
        std::cerr << "Invalid window type " + window_type<< std::endl;
        exit(1);
    }
    return arma::fmat();
}

arma::fmat hann_window(int window_size) {
    // generate symmetric hann window
    assert(window_size > 0 && "window size must greater than 0!");
    /* plan A: matrix operation */
    arma::frowvec window_v = arma::linspace<arma::frowvec>(0, window_size - 1, window_size);
    arma::fmat window(window_v);
    window = arma::pow(0.5 * (1 - arma::cos(2 * M_PI * window_v / (window_size - 1))), 0.85);

    /* plan B: element operation */
    // arma::fmat window(1, window_size, arma::fill::zeros);
    // for(int i = 0; i < window_size; i++) {
    //     window(0, i) = pow(0.5 * (1 - cos(2 * M_PI * i / (window_size - 1))), 0.85);
    // }
    return window;
}

arma::fmat get_window(arma::fmat& input, int padded_window_size, int window_size, int window_shift, double energy_floor, const std::string& window_type, double blackman_coeff, bool snip_edges, bool raw_energy, double dither, bool remove_dc_offset, double preemphasis_coefficient, bool use_signal_log_energy) {
    arma::fmat signal_log_energy;
    double epsilon = get_epsilon();
    get_strided(input, window_size, window_shift);
    if(dither != 0.0) {
        std::cout << "todo!!!" << std::endl;
        exit(0);       
    }
    if(remove_dc_offset) {
        arma::fmat mean_value = arma::mean(input, 1);
        input = input.each_col() - mean_value;
    }
    if(use_signal_log_energy && raw_energy) {
        signal_log_energy = get_log_energy(input, epsilon, energy_floor);
    }
    if(preemphasis_coefficient != 0.0) {
        /* plan A */
        // arma::fmat first_col = input.col(0);
        // arma::fmat offset_strided_input = arma::join_rows(first_col, input);
        // input = input - preemphasis_coefficient * offset_strided_input.submat(0, 0, offset_strided_input.n_rows - 1, offset_strided_input.n_cols - 2);

        /* plan B */
        arma::fmat offset_strided_input = preemphasis_coefficient * input;
        input.col(0) = input.col(0) - offset_strided_input.col(0);
        for(arma::uword i = 1; i < input.n_cols; i++) {
            input.col(i) = input.col(i) - offset_strided_input.col(i - 1);
        }
    }
    auto window = feature_window_function(window_type, window_size, 0.0);

    /*plan A: 9ms */
    for(arma::uword i = 0; i < input.n_rows; i++) {
        input.row(i) = input.row(i) % window;
    }
    /* plan B: 52ms */
    //input = input % pad(window, input.n_rows - 1, 0);
    if(padded_window_size != window_size) {
        int padding_right = padded_window_size - window_size;
        arma::fmat pad_matrix(input.n_rows, padding_right, arma::fill::zeros);
        input = arma::join_rows(input, pad_matrix);
    }

    // Compute energy after window function (not the raw one)
    if(use_signal_log_energy && !raw_energy) {
        signal_log_energy = get_log_energy(input, epsilon, energy_floor);
    }

    return signal_log_energy;
}

arma::fmat inverse_mel_scale(const arma::fmat& mel_freq) {
    return 700.0 * (arma::exp(mel_freq / 1127.0) - 1.0);
}

arma::fmat mel_scale(const arma::fmat& freq) {
    return 1127.0 * arma::log(1.0 + freq / 700.0);
}

double mel_scale_scalar(double freq) {
    return 1127.0 * log(1.0 + freq / 700.0);
}

arma::fmat get_mel_banks(int num_bins, int window_length_padded, double sample_freq, double low_freq, double high_freq, double vtln_low, double vtln_high, double vtln_warp_factor) {
    assert(num_bins > 3 && "Must have at least 3 mel bins!");
    assert(window_length_padded % 2 == 0);
    int num_fft_bins = window_length_padded / 2;
    double nyquist = 0.5 * sample_freq;

    if(high_freq <= 0.0) {
        high_freq += nyquist;
    }

    assert ((0.0 <= low_freq && low_freq < nyquist) && (0.0 < high_freq && high_freq <= nyquist) && (low_freq < high_freq) && "Bad values in options: low-freq, high-freq and nyquist!");
    
    double fft_bin_width = sample_freq / (double)window_length_padded;
    double mel_low_freq = mel_scale_scalar(low_freq);
    double mel_high_freq = mel_scale_scalar(high_freq);

    double mel_freq_delta = (mel_high_freq - mel_low_freq) / ((double)num_bins + 1);

    if(vtln_high < 0.0) {
        vtln_high += nyquist;
    }

    assert ((vtln_warp_factor == 1.0 || ((low_freq < vtln_low && vtln_low < high_freq) && (0.0 < vtln_high && vtln_high < high_freq) && (vtln_low < vtln_high))) && "Bad values in options: vtln-low, vtln-high, low-freq and high-freq!");

    arma::fmat bin = arange(num_bins);
    arma::fmat left_mel = mel_low_freq + bin * mel_freq_delta;
    arma::fmat center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta;
    arma::fmat right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta;

    if(vtln_warp_factor != 1.0) {
        std::cout << "todo!!!" << std::endl;
        exit(0);      
    }

    arma::fmat center_freqs = inverse_mel_scale(center_mel);
    arma::fmat mel = arma::trans(mel_scale(fft_bin_width * arange(num_fft_bins)));
    
    left_mel = pad(left_mel, mel.n_cols - left_mel.n_cols, 1);
    right_mel = pad(right_mel, mel.n_cols - right_mel.n_cols, 1);
    mel = pad(mel, left_mel.n_rows - mel.n_rows, 0);
    center_mel = pad(center_mel, mel.n_cols - center_mel.n_cols, 1);
    arma::fmat up_slope = (mel - left_mel) / (center_mel - left_mel);
    arma::fmat down_slope = (right_mel - mel) / (right_mel - center_mel);

    arma::fmat bins;
    if(vtln_warp_factor == 1.0) {
        bins = arma::max(arma::fmat(size(up_slope), arma::fill::zeros), arma::min(up_slope, down_slope));
    }
    else {
        std::cout << "todo!!!" << std::endl;
        exit(0);    
    }
    return bins;
}

arma::fmat fbank(arma::fmat input, int num_mel_bins, int frame_length, int frame_shift, int sample_frequency, double dither, double energy_floor, bool use_power, bool use_log_fbank, bool use_signal_log_energy) {
    /* get_waveform_and_window_properties: 1ms */
    auto window_paras = get_waveform_and_window_properties(input, 0, sample_frequency, frame_shift, frame_length);

    /* get_window: 93ms */
    auto signal_log_energy = get_window(input, window_paras(2), window_paras(1), window_paras(0), 0.0);
 
    // Real Fast Fourier Transform
    /* plan A: 220ms */
    // arma::cx_fmat fft_out = arma::fft(trans(input)); // 129ms
    // fft_out = trans(fft_out); // 36ms
    // arma::fmat spectrum = arma::pow(arma::pow(arma::real(fft_out), 2.0) + arma::pow(arma::imag(fft_out), 2.0), 0.5); // 55ms
    // spectrum = spectrum.submat(0, 0, spectrum.n_rows - 1, floor(spectrum.n_cols / 2));
    bm_handle_t handle;
    if(bm_dev_request(&handle, 0) != BM_SUCCESS) {
        std::cerr << "get bm handle failed!" << std::endl;
        exit(1);
    }
    bm_misc_info pmisc_info;
    if(bm_get_misc_info(handle, &pmisc_info) != BM_SUCCESS) {
        std::cerr << "get bm misc info failed!" << std::endl;
        exit(1);
    }
    bm_dev_free(handle);

    arma::fmat spectrum;
    if(pmisc_info.chipid_bit_mask == BM1684_CHIPID_BIT_MASK) {
        /*plan C: 145 ms */
        spectrum = bm_fft(input);
    }
    else {
        /*plan B: 212ms */
        spectrum = arma::fmat(input.n_rows, floor(input.n_cols / 2) + 1, arma::fill::zeros);
        for(arma::uword i = 0; i < spectrum.n_rows; i++) {
            arma::frowvec v = input.row(i);
            arma::cx_frowvec fft_v = arma::fft(v);
            fft_v = fft_v.subvec(0, floor(fft_v.n_cols / 2));
            arma::frowvec real_v = arma::pow(arma::pow(arma::real(fft_v), 2.0) + arma::pow(arma::imag(fft_v), 2.0), 0.5);
            spectrum.row(i) = real_v;
        }
    }

    /* rest: 22 ms */   
    if(use_power) {
        spectrum = arma::pow(spectrum, 2.0);
    }

    arma::fmat mel_energies = get_mel_banks(num_mel_bins, window_paras(2), sample_frequency, 20.0, 0.0, 100.0, -500.0, 1.0);
    mel_energies = arma::join_rows(mel_energies, arma::fmat(mel_energies.n_rows, 1, arma::fill::zeros));

    mel_energies = arma::trans(mel_energies);
    // mel_energies = matmul(spectrum, mel_energies);
    mel_energies = spectrum * mel_energies;
    
    if(use_log_fbank) {
        mel_energies = arma::log(arma::max(mel_energies, get_epsilon() * arma::fmat(size(mel_energies), arma::fill::ones)));
    }

    // todo:use_energy
    return mel_energies;
}
