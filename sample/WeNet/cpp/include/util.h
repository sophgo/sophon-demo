//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <system_error>
#include <thread>
#include <map>
#include "ctcdecode.h"

std::vector<std::string> read_dict(const std::string& dict_file);

std::vector<std::string> ctc_decoding(void* log_probs, 
    void* log_probs_idx, 
    void* chunk_out_lens, 
    int beam_size, 
    int batch_size, 
    const std::vector<std::string> &vocabulary, 
    std::vector<std::vector<std::pair<double, std::vector<int>>>>& score_hyps, 
    const std::string& mode = "ctc_prefix_beam_search"
);

std::map<std::string, std::string> read_data_lists(const std::string& data_lists_file);