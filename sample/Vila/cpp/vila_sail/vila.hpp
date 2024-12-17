//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef VILA_H
#define VILA_H

#include <iostream>
#include <cstdlib>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include "memory.h"
#include "sentencepiece/sentencepiece_processor.h"
#include <opencv2/opencv.hpp>
#include "engine_llm.h"
#include "cvwrapper.h"
#include <getopt.h>
#include <fstream>
#include <map>
#include <random>
#include <vector>


class VILA {
public:
  VILA(const std::vector<int> devids, std::string llm_model_path, std::string vision_model_path, std::string tokenizer_path);
  ~VILA();
  int forward_first(std::vector<int> &tokens, std::vector<std::map<int, std::shared_ptr<sail::Tensor>>>& vision_feat);
  int forward_next();
  void decode(std::string& word, std::vector<int>& tokens);
  int get_eos_id();
  void tokenizer_image_token(std::vector<int>& input_ids, std::string& prompt, int image_token_index=-200, bool lstrip=false);
  void vision_process(std::vector<std::map<int, std::shared_ptr<sail::Tensor>>>& res, std::vector<cv::Mat>& images);
  
private:
  void load_sentencepiece(std::string tokenizer_path);
  void process_images(std::vector<std::vector<float>>& processed_images, std::vector<cv::Mat>& images);

private:
  sentencepiece::SentencePieceProcessor sentencepiece;

  std::vector<int> devids;
  std::shared_ptr<sail::EngineLLM> llm_model;
  std::shared_ptr<sail::EngineLLM> vision_model;
  std::vector<std::string> llm_graph_names;
  int num_layers; // read from bmodel

  std::vector<std::shared_ptr<sail::Tensor>> past_k;
  std::vector<std::shared_ptr<sail::Tensor>> past_v;
  std::shared_ptr<sail::Tensor> first_k;
  std::shared_ptr<sail::Tensor> first_v;
  std::map<int, sail::Handle> handles;
  std::shared_ptr<sail::Tensor> position_ids_next;
  std::shared_ptr<sail::Tensor> attention_mask_next;
  std::map<int, std::shared_ptr<sail::Tensor>> input_vision_embed;
  std::map<int, std::shared_ptr<sail::Tensor>> output_vision_embed;
  std::map<int, std::shared_ptr<sail::Tensor>> input_llm_embed;
  std::map<int, std::shared_ptr<sail::Tensor>> output_llm_embed;
  std::map<int, std::shared_ptr<sail::Tensor>> input_block;
  std::map<int, std::shared_ptr<sail::Tensor>> input_lm;
  std::map<int, std::shared_ptr<sail::Tensor>> output_lm;
  std::map<int, std::shared_ptr<sail::Tensor>> output_llm_embed_cache;
  std::map<std::string, std::map<int, sail::Tensor*>> input_tensors;
  std::map<std::string, std::map<int, sail::Tensor*>> output_tensors;

  std::string name_vision_embed = "vision_embedding";
  std::string name_llm_embed = "embedding";
  std::string name_llm_embed_cache = "embedding_cache";
  std::string name_lm = "lm_head";
  std::vector<std::string> name_block;
  std::vector<std::string> name_block_cache;

  int num_frames;
  int vision_token_length;
  int seqlen;     // read from bmodel
  int hidden_size; // read from bmodel
  int num_head; // read from bmodel
  int head_dim; // read from bmodel
  int EOS;
  int SOS;
  int token_length;

  // image process
  std::vector<int> crop_size;
  std::vector<float> mean;
  std::vector<float> std;
};

#endif  //! VILA_H