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

using tokenizers::Tokenizer;

static const uint16_t ATTENTION_MASK = 0xC61C;

class MiniCPM4
{
public:
    // init deinit
    void init(std::string bmodel_path, const std::vector<int> &dev_ids, std::string tokenizer_path);
    void deinit();

    // infer
    int forward_first(std::vector<int> &tokens);
    int forward_next();
    void answer(std::string input_str, std::vector<std::pair<std::string, std::string>> &history_vector);

    // token encode & decode
    void decode_tokens(std::vector<int> &tokens, std::string &word);
    void encode_tokens(std::string &prompt, std::vector<int> &tokens);

    // end judge
    std::pair<bool, bool> is_end(int token);

    // generate prompt
    std::string build_prompt(std::string query, std::vector<std::pair<std::string, std::string>> &history_vector);

    // get vals
    int get_max_length();

    std::mt19937 sgen;
    MiniCPM4() : sgen(std::random_device()()) {};

private:
    void net_launch(const bm_net_info_t *net, int stage_idx = 0);
    void net_launch_dyn(const bm_net_info_t *net, int stage_idx = 0);

    inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
    inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);
    inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size);

    int greedy_search(bm_device_mem_t &logits_mem);
    int penalty_sample(bm_device_mem_t &logits_mem);

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
    bool lmhead_with_topk;
    std::vector<int> visited_tokens;

    // generation
    float temperature;
    float penalty;
    int top_k;
    float top_p;
    std::string generation_mode;

private:
    std::vector<bm_handle_t> handles;
    bm_handle_t bm_handle;
    void *p_bmrt;
    std::vector<const bm_net_info_t *> net_blocks;
    std::vector<const bm_net_info_t *> net_blocks_cache;
    const bm_net_info_t *net_embed;
    const bm_net_info_t *net_embed_cache;
    const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
    std::vector<bm_device_mem_t> past_key;
    std::vector<bm_device_mem_t> past_value;
    bm_device_mem_t dev_buffer;

    // tokenizer
    std::unique_ptr<Tokenizer> tok;
    int EOS;

    // system prompt
    std::string sys_config;
};

#endif //! QWEN_H