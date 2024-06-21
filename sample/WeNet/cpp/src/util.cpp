//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "util.h"

std::vector<std::string> read_dict(const std::string& dict_file) {
    std::ifstream infile(dict_file); // Replace "input.txt" with the filename of your input file.
    if (!infile.is_open() || !infile.good()) {
        std::cerr << "Failed to open file: " << dict_file << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<std::string> vocabulary;
    while (std::getline(infile, line)) { // Read each line of the file.
        std::istringstream iss(line);
        std::vector<std::string> words;
        std::string word;
        while (iss >> word) { // Split the line into words based on spaces.
            words.push_back(word);
        }
        vocabulary.push_back(words[0]);
    }
    infile.close();
    return vocabulary;
}

std::vector<std::string> ctc_decoding(void* log_probs, void* log_probs_idx, void* chunk_out_lens, int beam_size, int batch_size, const std::vector<std::string> &vocabulary, std::vector<std::vector<std::pair<double, std::vector<int>>>>& score_hyps, const std::string& mode) {
    int num_cores = std::thread::hardware_concurrency();
    size_t num_processes = std::min(num_cores, batch_size);
    std::vector<std::string> hyps;

    // Parsing the output
    std::vector<std::vector<std::vector<int>>> log_probs_idx_vector;
    std::vector<std::vector<std::vector<double>>> log_probs_vector;
    std::vector<int> chunk_out_lens_vector(batch_size);
    
    int32_t *log_probs_idx_ptr = static_cast<int32_t*>(log_probs_idx);
    float* log_probs_ptr = static_cast<float*>(log_probs);
    float *chunk_out_lens_ptr = static_cast<float*>(chunk_out_lens);
    for(int i = 0; i < batch_size; i++) {
        chunk_out_lens_vector[i] = chunk_out_lens_ptr[i];
    }
    
    int cur_pos = 0;
    for(int i = 0; i < batch_size; i++) {
        int out_feat_length = chunk_out_lens_vector[i];
        log_probs_idx_vector.push_back(std::vector<std::vector<int>>(out_feat_length, std::vector<int>(beam_size, 0)));
        log_probs_vector.push_back(std::vector<std::vector<double>>(out_feat_length, std::vector<double>(beam_size, 0)));
        for(int j = 0; j < out_feat_length; j++) {   
            for(int k = 0; k < beam_size; k++) {
                log_probs_idx_vector[i][j][k] = log_probs_idx_ptr[cur_pos];
                log_probs_vector[i][j][k] = static_cast<double>(log_probs_ptr[cur_pos]);
                cur_pos++;
            }
        }
    }

    // decoding
    if(mode == "ctc_greedy_search") {
        std::vector<std::vector<int>> batch_sents;
        for(int i = 0; i < batch_size; i++) {
            std::vector<int> tmp;
            for(int j = 0; j < chunk_out_lens_vector[i]; j++) {
                tmp.push_back(log_probs_idx_vector[i][j][0]);
            }
            batch_sents.push_back(tmp);
        }
        hyps = map_batch(batch_sents, vocabulary, num_processes, true, 0);
    }
    else if(mode == "ctc_prefix_beam_search" || mode == "attention_rescoring"){
        std::vector<bool> batch_start(batch_size, true);
        score_hyps = ctc_beam_search_decoder_batch(log_probs_vector, log_probs_idx_vector, batch_start, beam_size, num_processes, 0, -2, 0.99999);
        if(mode == "ctc_prefix_beam_search") {
            std::vector<std::vector<int>> batch_sents;
            for(const auto& cand_hyps : score_hyps) {
                batch_sents.push_back(cand_hyps[0].second);
            }
            hyps = map_batch(batch_sents, vocabulary, num_processes, false, 0);
        }
    }
    return hyps;
}

std::map<std::string, std::string> read_data_lists(const std::string& data_lists_file) {
    std::ifstream file(data_lists_file);
    if (!file.is_open() || !file.good()) {
        std::cerr << "Failed to open file: " << data_lists_file << std::endl;
        exit(1);
    }

    std::string line;
    std::map<std::string, std::string> data_map;

    while (std::getline(file, line)) {
        std::string key;
        std::string wav;

        // find the "key" value
        std::size_t prefix_length = 5;
        std::size_t key_pos = line.find("\"key\":");
        if (key_pos != std::string::npos) {
            std::size_t key_start = line.find('"', key_pos + 1 + prefix_length);
            std::size_t key_end = line.find('"', key_start + 1);
            key = line.substr(key_start + 1, key_end - key_start - 1);
        }

        // find the "wav" value
        std::size_t wav_pos = line.find("\"wav\":");
        if (wav_pos != std::string::npos) {
            std::size_t wav_start = line.find('"', wav_pos + 1 + prefix_length);
            std::size_t wav_end = line.find('"', wav_start + 1);
            wav = line.substr(wav_start + 1, wav_end - wav_start - 1);
        }

        if (!key.empty() && !wav.empty()) {
            data_map[key] = wav;
        }
    }
    file.close();
    
    return data_map;
}

