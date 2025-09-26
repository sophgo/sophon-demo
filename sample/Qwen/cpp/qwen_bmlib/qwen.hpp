//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef QWEN_H
#define QWEN_H

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <tokenizers-cpp/tokenizers_cpp.h>
#include <random>
#include <stdio.h>
#include <vector>
#include <dlfcn.h>
// #include "utils.h"
using tokenizers::Tokenizer;

static const uint16_t ATTENTION_MASK = 0xC61C;

class Qwen {
public:
  // init deinit
  void init(std::string bmodel_path, const std::vector<int> &dev_ids, std::string tokenizer_path);
  void deinit();

  // infer
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void answer(std::string input_str, std::vector<std::pair<std::string, std::string>>& history_vector);

  // token encode & decode
  void decode_tokens(std::vector<int>& tokens, std::string& word);
  void encode_tokens(std::string& prompt, std::vector<int>& tokens);

  // end judge
  std::pair<bool, bool> is_end(int token);

  // generate prompt
  std::string build_prompt(std::string query, std::vector<std::pair<std::string, std::string>>& history_vector);

  // get vals
  int get_max_length();

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int sample_head(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

  void load_sentencepiece(std::string tokenizer_path);

public:
  std::string version;

  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  int TOKEN_LEN;
  bool io_alone;
  bool is_dynamic;
  std::vector<int> visited_tokens;

  // generation
  float temperature = 0.8;
  float top_p = 0.8;
  float top_k = 50;
  float repeat_penalty = 1.1;
  int repeat_last_n;
  int max_new_tokens;
  std::string generation_mode;
  std::string prompt_mode;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head, *net_sample_head;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;

  // tokenizer
  std::unique_ptr<Tokenizer> tok;
  int EOS;
  int ID_IM_END;

  // system prompt
  std::string sys_config;
};

#endif  //! QWEN_H