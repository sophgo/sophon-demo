//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "vila.hpp"


void opencv_extract_frames(std::vector<cv::Mat>& images, std::string video_file, int num_frames, int dev_id=0) {
  // open video
  auto vidcap = cv::VideoCapture(video_file, cv::CAP_FFMPEG, dev_id);
  if (!vidcap.isOpened()){
    std::cerr << "Error: open video src failed in channel " << std::endl;
    exit(1);
  }

  // get frames
  int fps = vidcap.get(cv::CAP_PROP_FPS);
  int frame_count = (int)vidcap.get(cv::CAP_PROP_FRAME_COUNT);
  std::vector<int> frame_indices;
  frame_indices.push_back(0);
  for (int i = 1; i < num_frames - 1; i++) {
    frame_indices.push_back((int)((float)(frame_count - 1.0) / (num_frames - 1.0) * i));
  }
  if (num_frames - 1 > 0)
    frame_indices.push_back(frame_count-1);

  int count = 0;
  while (true) {
    cv::Mat image;
    if (frame_count >= num_frames) {
      vidcap.read(image);
      auto it = std::find(frame_indices.begin(), frame_indices.end(), count);
      if (it != frame_indices.end()) {
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        images.push_back(rgb_image);
        if (images.size() >= num_frames)
          break;
      }
      count += 1;
    }
    else {
      vidcap.read(image);
      if (image.empty()) {
        break;
      }
      cv::Mat rgb_image;
      cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
      images.push_back(rgb_image);
      count += 1;
    }
  }
  vidcap.release();
}

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{llm | ../../models/BM1684X/llama_int4_seq2560.bmodel | path of llm model}"
        "{vision | ../../models/BM1684X/vision_embedding_1batch.bmodel | path of vision_embedding model}"
        "{video | ../../datasets/test_car_person_1080P.mp4 | path of video}"
        "{num_input_frames | 6 | the number of sampled frames to infer}"
        "{tokenizer_path | ../../python/config/llm_token/tokenizer.model | path of tokenizer config}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string llm_bmodel_file = parser.get<std::string>("llm");
    std::string vision_bmodel_file = parser.get<std::string>("vision");
    std::string input = parser.get<std::string>("video");
    int num_input_frames = parser.get<int>("num_input_frames");
    std::string tokenizer_path = parser.get<std::string>("tokenizer_path");
    int dev_id = parser.get<int>("dev_id");
    std::vector<int> devids;
    devids.push_back(dev_id);

    // check params
    struct stat info;
    if (stat(llm_bmodel_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file: " << llm_bmodel_file << std::endl;
        exit(1);
    }
    if (stat(vision_bmodel_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file: " << vision_bmodel_file << std::endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0) {
        std::cout << "Cannot find valid input file: " << input << std::endl;
        exit(1);
    }

    // init
    auto vila = VILA(devids, llm_bmodel_file, vision_bmodel_file, tokenizer_path);

    // read video
    std::vector<cv::Mat> images;
    opencv_extract_frames(images, input, num_input_frames, dev_id);

    // get vision feat
    std::vector<std::map<int, std::shared_ptr<sail::Tensor>>> vision_feat;
    vila.vision_process(vision_feat, images);

    while (true) {
        std::cout << "\nQuestion for this video: ";
        std::string input_str;
        std::getline(std::cin, input_str);
        if (input_str == "exit")
            break;
        std::cout << "\nAnswer: ";
        std::string images_prompt = "";
        for (int i = 0; i < num_input_frames; i++)
            images_prompt += "<image>\n";
        std::string prompt = "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. USER: " 
                    + images_prompt + "<video>\\n " + input_str + ". ASSISTANT:";
        std::vector<int> tokens;
        vila.tokenizer_image_token(tokens, prompt);
        auto t0 = std::chrono::system_clock::now();
        int token = vila.forward_first(tokens, vision_feat);
        auto t1 = std::chrono::system_clock::now();
        int token_num = 0;
        std::string result;
        int pre_token = 0;
        while (token != vila.get_eos_id()){
            token_num += 1;
            // std::vector<int> token_ids;
            // token_ids.push_back(29871);
            // token_ids.push_back(token);
            // vila.decode(word, token_ids);
            // std::cout << word;
            std::string word;
            std::string pre_word;
            std::vector<int> pre_ids = {pre_token};
            std::vector<int> ids = {pre_token, token};
            vila.decode(pre_word, pre_ids);
            vila.decode(word, ids);
            std::string diff = word.substr(pre_word.size());
            result += diff;
            std::cout << diff << std::flush;
            token = vila.forward_next();  
        }
        
        // print speed
        auto t2 = std::chrono::system_clock::now();
        auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        printf("\n\nfirst token latency: %f s", (use0.count() * 1e-6));
        printf("\nspeed: %f token/s\n", token_num / (use1.count() * 1e-6));
    }

    return 0;
}