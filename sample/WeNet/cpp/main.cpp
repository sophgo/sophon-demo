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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>
#include "bmruntime_cpp.h"
#include "util.h"
#include "utils.hpp"
#include "wenet.h"
#include "opencv2/opencv.hpp"

using namespace bmruntime;

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout.setf(std::ios::fixed);
    // get params
    const char *keys="{encoder_bmodel | ../models/BM1684/wenet_encoder_fp32.bmodel | encoder bmodel file path}"
    "{decoder_bmodel |  | decoder bmodel file path}"
    "{dict_file | ../config/lang_char.txt | dictionary file path}"
    "{config_file | ../config/train_u2++_conformer.yaml | config file path}"
    "{result_file | ./result.txt | result file path}"
    "{input | ../datasets/aishell_S0764/aishell_S0764.list | input path, audio data list}"
    "{mode | ctc_prefix_beam_search | decoding mode}"
    "{dev_id | 0 | TPU device id}"
    "{help | 0 | print help information.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string encoder_bmodel = parser.get<std::string>("encoder_bmodel");
    std::string decoder_bmodel = parser.get<std::string>("decoder_bmodel");
    std::string dict_file = parser.get<std::string>("dict_file");
    std::string config_file = parser.get<std::string>("config_file");
    std::string result_file = parser.get<std::string>("result_file");
    std::string input = parser.get<std::string>("input");
    std::string mode = parser.get<std::string>("mode");
    int dev_id = parser.get<int>("dev_id");
    int decoding_chunk_size = 16;
    int subsampling_rate = 4;
    int context = 7;

    // check params
    struct stat info;
    if (stat(encoder_bmodel.c_str(), &info) != 0) {
        std::cout << "Cannot find valid encoder model file." << std::endl;
        exit(1);
    }
    if(mode == "attention_rescoring" && stat(decoder_bmodel.c_str(), &info) != 0) {
        std::cout << "Cannot find valid decoder model file." << std::endl;
        exit(1);
    }
    if (stat(dict_file.c_str(), &info) != 0){
        std::cout << "Cannot find valid dictionary file." << std::endl;
        exit(1);
    }
    if (stat(config_file.c_str(), &info) != 0){
        std::cout << "Cannot find valid config file." << std::endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0){
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    auto data_map = read_data_lists(input);
    std::vector<std::string> dict = read_dict(dict_file);

    // int sample_frequency = 16000;
    // int num_mel_bins = 80;
    // int frame_shift = 10;
    // int frame_length = 25;

    // load configuration file
    // cv::FileStorage fs(config_file, cv::FileStorage::READ);
    // int sample_frequency = (int)fs["dataset_conf"]["resample_conf"]["resample_rate"];
    // int num_mel_bins = (int)fs["dataset_conf"]["fbank_conf"]["num_mel_bins"];
    // int frame_shift = (int)fs["dataset_conf"]["fbank_conf"]["frame_shift"];
    // int frame_length = (int)fs["dataset_conf"]["fbank_conf"]["frame_length"];

    std::ifstream fin(config_file);
    YAML::Node doc = YAML::Load(fin);
    fin.close();
    int sample_frequency = doc["dataset_conf"]["resample_conf"]["resample_rate"].as<int>();
    int num_mel_bins = doc["dataset_conf"]["fbank_conf"]["num_mel_bins"].as<int>();
    int frame_shift = doc["dataset_conf"]["fbank_conf"]["frame_shift"].as<int>();
    int frame_length = doc["dataset_conf"]["fbank_conf"]["frame_length"].as<int>();

    std::FILE* file_exists = std::fopen(result_file.c_str(), "r");
    if (file_exists) {
        // File exists, delete it
        std::fclose(file_exists);
        std::remove(result_file.c_str());
    }

    // load model
    auto encoder_ctx = std::make_shared<Context>(dev_id);
    bm_status_t status = encoder_ctx->load_bmodel(encoder_bmodel.c_str());
    assert(BM_SUCCESS == status);

    std::shared_ptr<Context> decoder_ctx;
    if(mode == "attention_rescoring") {
        decoder_ctx = std::make_shared<Context>(dev_id);
        bm_status_t status = decoder_ctx->load_bmodel(decoder_bmodel.c_str());
        assert(BM_SUCCESS == status);
    }

    WeNet wenet(encoder_ctx, decoder_ctx);
    wenet.Init(dict, sample_frequency, num_mel_bins, frame_shift, frame_length, decoding_chunk_size, subsampling_rate, context, mode);

    // profiling
    TimeStamp wenet_ts;
    wenet.enableProfile(&wenet_ts);

    for (const auto& pair : data_map) {
        const char* file_path = pair.second.c_str();
        auto result = wenet.Recognize(file_path);

        std::cout << "Key: " << pair.first << " , Result: " << result << std::endl;

        // Open file for writing
        std::ofstream result_file_stream(result_file, std::ios::app);

        if (!result_file_stream.is_open()) {
            // Failed to open file, handle error
            std::cerr << "Failed to open file " << result_file << " for writing." << std::endl;
            return 1;
        }
        result_file_stream << pair.first + " " + result + "\n";
        result_file_stream.close();
    }
    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    wenet_ts.calbr_basetime(base_time);
    wenet_ts.build_timeline("wenet test");
    wenet_ts.show_summary("wenet test");
    wenet_ts.clear();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "程序运行时间为 " << duration.count() << " 毫秒" << std::endl;
    
    return 0;
}