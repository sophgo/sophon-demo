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
#include "qwen.hpp"


void split(const std::string &s, const std::string &delim,
                  std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (last < s.length()) {
    ret.push_back(s.substr(last));
  }
}

std::vector<int> parseCascadeDevices(const std::string &str) {
  std::vector<int> devices;
  std::vector<std::string> sub_str;
  split(str, ",", sub_str);
  for (auto &s : sub_str) {
    devices.push_back(std::atoi(s.c_str()));
  }
  return devices;
}

void Usage() {
  printf("Usage:\n"
         "  --help                  : Show help info.\n"
         "  --bmodel_path           : Set bmodel path \n"
         "  --tokenizer_path        : Set tokenizer path \n"
         "  --dev_id                : Set devices to run for model, e.g. 1,2, if not provided, use 0\n"
         "\n");
}

void processArguments(int argc, char *argv[], std::string &bmodel_path,
                      std::string &tokenizer_path, std::vector<int> &devices) {
  struct option longOptions[] = {
      {"bmodel_path", required_argument, nullptr, 'm'},
      {"tokenizer_path", required_argument, nullptr, 't'},
      {"dev_id", required_argument, nullptr, 'd'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:t:d:h", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      bmodel_path = optarg;
      break;
    case 't':
      tokenizer_path = optarg;
      break;
    case 'd':
      devices = parseCascadeDevices(optarg);
      break;
    case 'h':
      Usage();
      exit(EXIT_SUCCESS);
    case '?':
      Usage();
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::fixed);
    // get params
    std::string bmodel_path = "../../models/BM1684X/qwen2.5-1.5b_int4_seq512_1dev.bmodel";
    std::string tokenizer_path = "../../python/token_config/tokenizer.json";
    std::vector<int> devids = {0};
    processArguments(argc, argv, bmodel_path, tokenizer_path, devids);

    // check params
    struct stat info;
    if (stat(bmodel_path.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file: " << bmodel_path << std::endl;
        exit(1);
    }
    if (stat(tokenizer_path.c_str(), &info) != 0) {
        std::cout << "Cannot find tokenizer file: " << tokenizer_path << std::endl;
        exit(1);
    }

    // init
    auto qwen = Qwen();
    qwen.init(bmodel_path, devids, tokenizer_path);
    std::vector<std::pair<std::string, std::string>> history_vector;

    while (true) {
        std::cout << "\nQuestion: ";

        std::string input_str;
        std::getline(std::cin, input_str);
        if (input_str == "exit")
            break;

        std::cout << "\nAnswer: " << std::flush;
        qwen.answer(input_str, history_vector);
        std::cout << std::endl;
    }

    qwen.deinit();
    return 0;
}