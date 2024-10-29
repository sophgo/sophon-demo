//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "SimpleTokenizer.hpp"
#include "clip.hpp"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
using namespace std;

void process_images(const std::vector<std::string>& image_paths, const std::vector<std::string>& text_inputs, CLIP& model, SimpleTokenizer& tokenizer) {
    // calculate text features
    std::vector<std::vector<float>> text_features;
    for (const auto& text : text_inputs){
        std::wstring wstr = std::wstring(text.begin(), text.end());
        std::vector<int64_t> tokenlized_text = tokenizer.tokenlize(wstr);
        std::vector<float> text_feature = model.encode_text(tokenlized_text);
        text_features.push_back(text_feature);
    }

    for (const auto& filename : image_paths) {
        std::cout << "Filename: " << filename << std::endl;
        cv::Mat image = cv::imread(filename);
        if (image.empty()) {
            std::cerr << "Error reading image: " << filename << std::endl;
            continue;
        }
        std::vector<float> image_input = model.preprocess(image);
        std::vector<float> image_feature = model.encode_image(image_input);
        // calculate similarity
        std::vector<float> similarity(text_inputs.size());
        similarity = model.calculate_similarity(image_feature, text_features);

        auto [values, indices] = model.topk(similarity, std::min(text_inputs.size(), static_cast<size_t>(model.top_k)));
        for (size_t i = 0; i < text_inputs.size(); ++i) {
            std::cout << "Text: " << text_inputs[indices[i]] << ", Similarity: " << values[i] << std::endl;
        }
    }
}

std::vector<std::string> get_image_paths(const std::string& image_path) {
    std::vector<std::string> image_paths;
    if (std::filesystem::is_regular_file(image_path)) {
        image_paths.push_back(image_path);
    } else if (std::filesystem::is_directory(image_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(image_path)) {
            if (entry.is_regular_file()) {
                image_paths.push_back(entry.path().string());
            }
        }
    }
    return image_paths;
}

std::vector<std::string> split(const std::string& input) {
    std::vector<std::string> result;
    std::istringstream stream(input);
    std::string item;

    while (std::getline(stream, item, ',')) {
        item.erase(std::remove(item.begin(), item.end(), '\"'), item.end());
        result.push_back(item);
    }

    return result;
}

std::vector<int> parse_devices(const std::string& device_str) {
    std::vector<int> devices;
    std::string trimmed = device_str.substr(1, device_str.size() - 2);
    std::istringstream iss(trimmed);
    std::string token;

    while (std::getline(iss, token, ',')) {
        devices.push_back(std::stoi(token));
    }
    return devices;
}

int main(int argc, char *argv[]){
    // get params
    const char *keys =
        "{image_path | ../../datasets | path to the image directory}"
        "{text | \"a diagram,a dog\" | text inputs for prediction (multiple texts can be separated by spaces and must be quoted)}"
        "{image_model | ../../models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel | path to the image model file}"
        "{text_model | ../../models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel | path to the text model file}"
        "{dev_id | 0 | TPU device ids (comma-separated list)}"
        "{help | 0 | print help information.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
    }
    
    std::cout << "argc: " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    }

    std::string image_path = parser.get<string>("image_path");
    std::string text_input = parser.get<string>("text");
    std::vector<std::string> text_vector = split(text_input);
    std::string image_model = parser.get<string>("image_model");
    std::string text_model = parser.get<string>("text_model");
    int dev_id = parser.get<int>("dev_id");

    // check params
    struct stat info;
    if (stat(image_model.c_str(), &info) != 0) {
        cout << "Cannot find valid image model file: " << image_model << endl;
        exit(1);
    }
    if (stat(text_model.c_str(), &info) != 0) {
        cout << "Cannot find valid text model file: " << text_model << endl;
        exit(1);
    }
    if (stat(image_path.c_str(), &info) != 0) {
        cout << "Cannot find input path: " << image_path << endl;
        exit(1);
    }

    //  Load bmodel
    CLIP clip;
    printf("Init Environment ...\n");
    clip.init(image_model, text_model, dev_id);
    printf("==========================\n");
    SimpleTokenizer tokenizer;

    // predict
    auto image_paths = get_image_paths(image_path);
    process_images(image_paths, text_vector, clip, tokenizer);

    // Logging average times
    size_t image_num = image_paths.size();
    std::cout << "-------------------Image num " << image_num << ", Preprocess average time ------------------------" << std::endl;
    std::cout << "preprocess(ms): " << (clip.preprocess_time / image_num * 1000) << std::endl;

    std::cout << "------------------ Image num " << image_num << ", Image Encoding average time ----------------------" << std::endl;
    std::cout << "image_encode(ms): " << (clip.encode_image_time / image_num * 1000) << std::endl;

    std::cout << "------------------ Image num " << image_num << ", Text Encoding average time ----------------------" << std::endl;
    std::cout << "text_encode(ms): " << (clip.encode_text_time / image_num * 1000) << std::endl;

    std::cout << "All done." << std::endl;
    clip.deinit();
    return 0;
}