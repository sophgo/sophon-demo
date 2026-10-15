//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <getopt.h>

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bmnn_utils.h"
#include "sam3.hpp"
#include "utils.hpp"

namespace {
bool fileExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}
}  // namespace

int main(int argc, char* argv[]) {
  const char* keys =
      "{model_dir | ../../models/BM1684X_504 | bmodel directory}"
      "{bpe | ./tokenizer/bpe_simple_vocab_16e6.txt.gz | BPE vocab (gz)}"
      "{post_npz | ../../models/post_process_weights.npz | post-process weights}"
      "{seg_npz | ../../models/seg_head_weights.npz | mask-decoder weights}"
      "{input_image | ../../datasets/truck.jpg | input image}"
      "{prompt | a truck | text prompt(s), comma-separated}"
      "{score_thresh | 0.3 | score threshold}"
      "{output | ./sam3_output.jpg | output image path}"
      "{dev_id | 0 | device id}"
      "{streaming | false | stream ViT part1-4 bmodels (less device memory)}"
      "{help | | print help information.}";

  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  const std::string modelDir = parser.get<std::string>("model_dir");
  const std::string bpePath = parser.get<std::string>("bpe");
  const std::string postNpz = parser.get<std::string>("post_npz");
  const std::string segNpz = parser.get<std::string>("seg_npz");
  const std::string imagePath = parser.get<std::string>("input_image");
  const std::string promptArg = parser.get<std::string>("prompt");
  const float scoreThresh = parser.get<float>("score_thresh");
  const std::string outputPath = parser.get<std::string>("output");
  const int devId = parser.get<int>("dev_id");
  const bool streaming = parser.get<bool>("streaming");

  if (!fileExists(modelDir)) {
    std::cout << "model dir not found: " << modelDir << std::endl;
    return 1;
  }
  if (!fileExists(bpePath) || !fileExists(postNpz) ||
      !fileExists(segNpz)) {
    std::cout << "bpe / npz weight file not found (see --help for paths)"
              << std::endl;
    return 1;
  }
  cv::Mat image = cv::imread(imagePath);
  if (image.empty()) {
    std::cout << "cannot read image: " << imagePath << std::endl;
    return 1;
  }

  // comma-separated prompts
  std::vector<std::string> prompts;
  {
    std::string cur;
    for (char ch : promptArg) {
      if (ch == ',') {
        if (!cur.empty()) prompts.push_back(cur);
        cur.clear();
      } else {
        cur += ch;
      }
    }
    if (!cur.empty()) prompts.push_back(cur);
  }
  if (prompts.empty()) {
    std::cout << "no prompt given" << std::endl;
    return 1;
  }

  TimeStamp ts;
  auto handle = std::make_shared<BMNNHandle>(devId);
  std::cout << "set device id: " << devId << std::endl;

  Sam3 sam3(handle, modelDir, bpePath, postNpz, segNpz, devId);
  sam3.enableProfile(&ts);
  if (sam3.Init(streaming) != 0) {
    std::cout << "sam3 init failed" << std::endl;
    return 1;
  }

  std::vector<Sam3Result> results;
  if (sam3.predict(image, prompts, scoreThresh, results) != 0) {
    std::cout << "sam3 predict failed" << std::endl;
    return 1;
  }

  for (const auto& r : results) {
    std::cout << "prompt: " << r.prompt << ", score: " << r.score
              << ", query: " << r.best_idx << ", box(xyxy): [" << r.box_xyxy[0]
              << "," << r.box_xyxy[1] << "," << r.box_xyxy[2] << ","
              << r.box_xyxy[3] << "]";
    if (r.score >= scoreThresh)
      std::cout << ", mask pixels: " << r.mask_pixels;
    std::cout << std::endl;
  }

  Sam3::drawResults(image, results);
  cv::imwrite(outputPath, image);
  std::cout << "result saved to " << outputPath << std::endl;

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("SAM3");
  ts.show_summary("SAM3");
  ts.clear();
  return 0;
}
