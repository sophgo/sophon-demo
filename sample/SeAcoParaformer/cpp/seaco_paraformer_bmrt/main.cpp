//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sndfile.h>
#include "seaco_paraformer.h"
#include "json.hpp"

using json = nlohmann::json;

// Read WAV file -> float32 mono samples
static std::vector<float> read_wav(const std::string& path, int target_sr = 16000) {
    SF_INFO sfinfo;
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &sfinfo);
    if (!sf) {
        std::cerr << "Error: cannot open " << path << ": " << sf_strerror(nullptr) << std::endl;
        return {};
    }

    std::vector<float> samples(sfinfo.frames * sfinfo.channels);
    sf_readf_float(sf, samples.data(), sfinfo.frames);
    sf_close(sf);

    // Convert to mono by averaging channels
    if (sfinfo.channels > 1) {
        std::vector<float> mono(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; i++) {
            float sum = 0.0f;
            for (int c = 0; c < sfinfo.channels; c++)
                sum += samples[i * sfinfo.channels + c];
            mono[i] = sum / sfinfo.channels;
        }
        samples = std::move(mono);
    }

    // Simple resampling via linear interpolation (if needed)
    if (sfinfo.samplerate != target_sr) {
        double ratio = (double)target_sr / sfinfo.samplerate;
        int new_len = (int)(samples.size() * ratio);
        std::vector<float> resampled(new_len);
        for (int i = 0; i < new_len; i++) {
            double src_idx = i / ratio;
            int s0 = (int)src_idx;
            int s1 = std::min(s0 + 1, (int)samples.size() - 1);
            double frac = src_idx - s0;
            resampled[i] = (float)(samples[s0] * (1.0 - frac) + samples[s1] * frac);
        }
        samples = std::move(resampled);
    }

    return samples;
}

int main(int argc, char** argv) {
    std::string model_dir = "../models/BM1684X";
    std::string input_path;
    int dev_id = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model_dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--dev_id" && i + 1 < argc) {
            dev_id = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model_dir DIR   Model directory (default: ../models/BM1684X)\n"
                      << "  --input WAV       Input WAV file (16kHz mono)\n"
                      << "  --dev_id ID       TPU device ID (default: 0)\n";
            return 0;
        }
    }

    if (input_path.empty()) {
        std::cerr << "Error: --input WAV is required\n";
        return 1;
    }

    // Load audio
    std::vector<float> audio = read_wav(input_path);
    if (audio.empty()) {
        std::cerr << "Error: failed to read audio\n";
        return 1;
    }
    float audio_dur = audio.size() / 16000.0f;
    std::cout << "Audio duration: " << audio_dur << " s" << std::endl;

    // Load model
    std::cout << "Loading SeacoParaformer from " << model_dir << std::endl;
    TimeStamp ts;
    ts.save("model init");
    SeacoParaformer model(model_dir, dev_id);
    ts.save("model init");

    // Inference
    ts.save("inference");
    auto result = model.infer(audio);
    ts.save("inference");

    // Print results
    std::cout << "\n====================================================" << std::endl;
    std::cout << "TEXT: " << result.text << std::endl;
    std::cout << "====================================================" << std::endl;

    if (!result.sentence_info.empty()) {
        for (auto& si : result.sentence_info) {
            std::cout << "  [" << si.start_ms << "][" << si.end_ms << "]  "
                      << si.text << std::endl;
        }
    }

    // Timing summary
    std::cout << "\n------------------ Inference Time ----------------------" << std::endl;
    std::cout << "  preprocess:  " << model.t_pre() << " s" << std::endl;
    std::cout << "  encoder:     " << model.t_enc() << " s" << std::endl;
    std::cout << "  cif:         " << model.t_cif() << " s" << std::endl;
    std::cout << "  decoder:     " << model.t_dec() << " s" << std::endl;
    std::cout << "  predictor:   " << model.t_pred() << " s" << std::endl;
    std::cout << "  decode:      " << model.t_tok() << " s" << std::endl;
    double total = model.t_pre() + model.t_enc() + model.t_cif() +
                   model.t_dec() + model.t_pred() + model.t_tok();
    std::cout << "  total:       " << total << " s" << std::endl;
    if (audio_dur > 0)
        std::cout << "  RTF:         " << total / audio_dur << std::endl;

    // Save result JSON
    std::string basename = input_path;
    size_t slash = basename.find_last_of("/\\");
    if (slash != std::string::npos) basename = basename.substr(slash + 1);
    size_t dot = basename.find_last_of(".");
    if (dot != std::string::npos) basename = basename.substr(0, dot);

    std::string out_dir = "./results";
    std::string cmd = "mkdir -p " + out_dir;
    system(cmd.c_str());

    json result_json;
    result_json["audio_file"] = input_path;
    result_json["duration_s"] = audio_dur;
    result_json["text"] = result.text;
    result_json["tokens"] = result.tokens;
    result_json["wall_time_s"] = total;
    result_json["rtf"] = total / audio_dur;

    json sentence_info = json::array();
    for (auto& si : result.sentence_info) {
        json si_j;
        si_j["start"] = si.start_ms;
        si_j["end"] = si.end_ms;
        si_j["text"] = si.text;
        sentence_info.push_back(si_j);
    }
    result_json["sentence_info"] = sentence_info;

    std::string out_path = out_dir + "/" + basename + "_asr.json";
    std::ofstream of(out_path);
    of << std::setw(4) << result_json << std::endl;
    std::cout << "Result saved -> " << out_path << std::endl;

    return 0;
}
