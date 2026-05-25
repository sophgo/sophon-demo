//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef VITS_PREPROCESS_HPP
#define VITS_PREPROCESS_HPP

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include "bmruntime_interface.h"
#include "bmlib_runtime.h"
#include "utils.hpp"

class VitsPreprocessor {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    int m_max_length;
    int m_embed_dim;

    // hanzi -> pinyin (TONE3)
    std::unordered_map<std::string, std::string> m_pinyin_map;

    // pinyin -> (initial, final)
    std::unordered_map<std::string, std::pair<std::string, std::string>> m_pinyin_dict;

    // BERT vocab: token string -> id
    std::unordered_map<std::string, int> m_vocab;

    // phoneme symbol string -> id
    std::unordered_map<std::string, int> m_symbol_to_id;

    TimeStamp* m_ts = NULL;
    TimeStamp tmp_ts;

    float* get_cpu_data(bm_tensor_t* tensor);

    bool is_chinese(uint32_t cp);
    std::string clean_chinese(const std::string& text);
    std::string char_to_pinyin(const std::string& ch);
    std::vector<std::string> get_phoneme4pinyin(const std::vector<std::string>& pinyins,
                                                  std::vector<int>& count_phone);

    void load_pinyin_map(const std::string& filepath);
    void load_vocab(const std::string& filepath);
    void build_pinyin_dict();
    void build_symbol_table();

public:
    VitsPreprocessor(const std::string& bert_model,
                     const std::string& pinyin_map_file,
                     const std::string& vocab_file,
                     int dev_id = 0);
    ~VitsPreprocessor();

    int max_length() const { return m_max_length; }

    int process(const std::string& text,
                std::vector<int32_t>& input_ids,
                std::vector<float>& char_embeds);
};

#endif // VITS_PREPROCESS_HPP