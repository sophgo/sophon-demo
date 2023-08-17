//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chatglm2.hpp"

#include <iostream>
#include <string>
#include <cstdlib> // For atoi

int main(int argc, char **argv) {
 
    std::string bmodel = "../models/BM1684X/chatglm2-6b.bmodel";
    std::string token = "../models/BM1684X/tokenizer.model";
    int dev_id = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bmodel" && i + 1 < argc) {
            bmodel = argv[++i];
        } else if (arg == "--token" && i + 1 < argc) {
            token = argv[++i];
        } else if (arg == "--dev_id" && i + 1 < argc) {
            dev_id = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--bmodel <bmodel_path>] [--token <token_path>] [--dev_id <device_id>] [--help]" << std::endl;
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }


  printf("Demo for ChatGLM2-6B in BM1684X\n");

  ChatGLM2 glm;
  printf("Init Environment ...\n");
  glm.init(dev_id,bmodel,token);
  printf("==========================\n");
  glm.chat();
  glm.deinit();
  return 0;
}
