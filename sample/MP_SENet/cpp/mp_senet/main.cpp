//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "mp_senet.hpp"

#include <sys/stat.h>
#include <iostream>
#include <string>
#include "cxxopts.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options options("mp_senet", "MP_SENet audio denoising demo on SOPHON devices");
    options.add_options()
        ("bmodel", "bmodel file path",
            cxxopts::value<std::string>()->default_value("../models/BM1684X/mpsenet_vb_1b_bf16.bmodel"))
        ("wav", "input WAV file path",
            cxxopts::value<std::string>()->default_value("../datasets/test.wav"))
        ("output", "output WAV file path",
            cxxopts::value<std::string>()->default_value("./results/test_denoised.wav"))
        ("dev_id", "TPU device id",
            cxxopts::value<int>()->default_value("0"))
        ("compress_factor", "compress factor (bf16=0.27, fp32=0.30)",
            cxxopts::value<float>()->default_value("0.27"))
        ("sampling_rate", "sampling rate",
            cxxopts::value<int>()->default_value("16000"))
        ("n_fft", "n_fft",
            cxxopts::value<int>()->default_value("400"))
        ("hop_size", "hop size (bf16=150, fp32=100)",
            cxxopts::value<int>()->default_value("150"))
        ("win_size", "window size",
            cxxopts::value<int>()->default_value("400"))
        ("help", "print help");
    auto parser = options.parse(argc, argv);

    if (parser.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    MpSenetConfig cfg;
    cfg.bmodel_path     = parser["bmodel"].as<std::string>();
    cfg.wav_path        = parser["wav"].as<std::string>();
    cfg.output_path     = parser["output"].as<std::string>();
    cfg.dev_id          = parser["dev_id"].as<int>();
    cfg.compress_factor = parser["compress_factor"].as<float>();
    cfg.sample_rate     = parser["sampling_rate"].as<int>();
    cfg.n_fft           = parser["n_fft"].as<int>();
    cfg.hop_size        = parser["hop_size"].as<int>();
    cfg.win_size        = parser["win_size"].as<int>();

    // Auto-detect bf16 vs fp32 from bmodel filename
    std::string bmodel_lower = cfg.bmodel_path;
    std::transform(bmodel_lower.begin(), bmodel_lower.end(),
                   bmodel_lower.begin(), ::tolower);
    if (bmodel_lower.find("bf16") != std::string::npos) {
        cfg.is_bf16 = true;
        cfg.compress_factor = 0.27f;
        cfg.hop_size = 150;
    } else if (bmodel_lower.find("fp32") != std::string::npos) {
        cfg.is_bf16 = false;
        cfg.compress_factor = 0.30f;
        cfg.hop_size = 100;
    }

    // Validate paths
    struct stat info;
    if (stat(cfg.bmodel_path.c_str(), &info) != 0) {
        std::cerr << "Cannot find bmodel: " << cfg.bmodel_path << std::endl;
        return 1;
    }
    if (stat(cfg.wav_path.c_str(), &info) != 0) {
        std::cerr << "Cannot find input WAV: " << cfg.wav_path << std::endl;
        return 1;
    }

    // Create output directory if needed
    {
        size_t slash = cfg.output_path.find_last_of('/');
        if (slash != std::string::npos) {
            std::string dir = cfg.output_path.substr(0, slash);
            if (!dir.empty() && stat(dir.c_str(), &info) != 0) {
                std::string cmd = "mkdir -p " + dir;
                system(cmd.c_str());
            }
        }
    }

    std::cout << "======================================" << std::endl;
    std::cout << "MP_SENet C++ Demo" << std::endl;
    std::cout << "  bmodel:     " << cfg.bmodel_path << std::endl;
    std::cout << "  input wav:  " << cfg.wav_path << std::endl;
    std::cout << "  output wav: " << cfg.output_path << std::endl;
    std::cout << "  dev_id:     " << cfg.dev_id << std::endl;
    std::cout << "  mode:       " << (cfg.is_bf16 ? "bf16" : "fp32") << std::endl;
    std::cout << "  compress:   " << cfg.compress_factor << std::endl;
    std::cout << "  hop_size:   " << cfg.hop_size << std::endl;
    std::cout << "======================================" << std::endl;

    MpSenet mp_senet(cfg);
    if (mp_senet.Init() != 0) {
        std::cerr << "Init failed" << std::endl;
        return 1;
    }

    TimeStamp ts;
    mp_senet.enableProfile(&ts);

    int ret = mp_senet.process(cfg.wav_path, cfg.output_path);
    if (ret != 0) {
        std::cerr << "Process failed" << std::endl;
        return 1;
    }

    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts.calbr_basetime(base_time);
    ts.build_timeline("MP_SENet");
    ts.show_summary("MP_SENet");
    ts.clear();

    return 0;
}
