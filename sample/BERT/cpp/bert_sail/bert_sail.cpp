//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bert_sail.hpp"
#include <stdio.h>
#include <sail/cvwrapper.h>
#include <sail/tensor.h>
#include <sail/engine.h>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"
map<string, int> text2id;
map<int, string> id2text;
BERT::BERT(string model_path, string pre_train_path, int id) {
    device_id = id;
    mode = sail::SYSO;
    // 1. Initialize handle
    handle = std::make_shared<sail::Handle>(device_id);
    // 2. Initialize bmcv
    bmcv = std::make_shared<sail::Bmcv>(*handle);
    // 3. Initialize engine
    engine = std::make_shared<sail::Engine>(model_path, *handle, mode);
    graph_name = engine->get_graph_names()[0];
    input_name = engine->get_input_names(graph_name)[0];
    output_names = engine->get_output_names(graph_name);
    input_shape = engine->get_input_shape(graph_name, input_name);
    batch_size = input_shape[0];
    input_dtype = engine->get_input_dtype(graph_name, input_name);
    input_scale = (engine->get_input_scale(graph_name, input_name));

    tokenizer.add_vocab(pre_train_path.c_str());
    tokenizer.maxlen_ = 256;
    tokenizer.do_lower_case_ = 1;
    input_tensor = sail::Tensor(*handle, input_shape, input_dtype, 1, 1);
    for (int i = 0; i < output_names.size(); i++) {
        output_shape.push_back(
            engine->get_output_shape(graph_name, output_names[i]));
        output_dtype.push_back(
            engine->get_output_dtype(graph_name, output_names[i]));
        output_scale.push_back(
            engine->get_output_scale(graph_name, output_names[i]));
        output_tensors.push_back(
            sail::Tensor(*handle, output_shape[i], output_dtype[i], 1, 1));
    }
    id2label.push_back("O");
    id2label.push_back("B-LOC");
    id2label.push_back("I-LOC");
    id2label.push_back("B-PER");
    id2label.push_back("I-PER");
    id2label.push_back("B-ORG");
    id2label.push_back("I-ORG");
};
void BERT::pre_process(string text) { // pre_process
    /*
    input : text
    output : { tokens , token_ids }
    */
    text = "[CLS] " + text + " [SEP]";
    tokens = tokenizer.tokenize(text);
    token_ids = tokenizer.convert_tokens_to_ids(tokens);
    for (int i = token_ids.size(); i < 256; i++) {
        token_ids.push_back(0);
        tokens.push_back("[PAD]");
    }
    token_ids.resize(256);
    tokens.resize(256);
    return;
}
void BERT::pre_process(std::vector<string> texts) { // pre_process
    /*
    input : text
    output : { tokens , token_ids }
    */
    if (texts.size() != 8) {
        std::cout << "Unsupport batch size!" << std::endl;
    }
    token_ids.clear();
    tokens.clear();
    for (int j = 0; j < 8; j++) {
        string text = "[CLS] " + texts[j] + " [SEP]";

        vector<string> tmp_tokens;
        tmp_tokens = tokenizer.tokenize(text);

        vector<float> tmp;
        tmp = tokenizer.convert_tokens_to_ids(tmp_tokens);
        for (int i = 0; i < min((int)tmp.size(), 256); i++) {
            token_ids.push_back(tmp[i]);
            tokens.push_back(tmp_tokens[i]);
        }
        for (int i = tmp.size(); i < 256; i++) {
            token_ids.push_back(0);
            tokens.push_back("[PAD]");
        }
    }

    return;
}
bool BERT::Detect() // process
{
    /*
    input : token_ids
    output : output_tensors
    */
    std::map<std::string, sail::Tensor *> input =
        engine->create_input_tensors_map(graph_name, -1);
    input_tensor.reset_sys_data(token_ids.data(), input_shape);
    input_tensor.sync_s2d();
    input[input_name] = &input_tensor;
    std::map<std::string, sail::Tensor *> output =
        engine->create_output_tensors_map(graph_name, -1);
    for (int i = 0; i < output_names.size(); i++) {
        output[output_names[i]] = &output_tensors[i];
    }
    engine->process(graph_name, input, output);
    for (int i = 0; i < output_names.size(); i++) {
        output_tensors[i].sync_d2s();
    }
    return true;
}
void BERT::softmax(float *x, int length) // softmax for the x input of length
{
    float sum = 0;
    int i = 0;
    for (i = 0; i < length; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (i = 0; i < length; i++) {
        x[i] /= sum;
    }
}
vector<pair<vector<string>, vector<string>>>
BERT::post_process() // post_process
{
    /*
        input : output_tensors
    */
    float *tmp =
        (float *)output_tensors[0].sys_data(); // get output_tensors' dates
    string s = "";
    vector<pair<vector<string>, vector<string>>> ans;
    for (int k = 0; k < batch_size; k++) {
        pair<vector<string>, vector<string>> TMP;
        for (int i = 0; i < 256; i++) {
            if (token_ids[i + k * 256] == 0) {
                tmp += (256 - i) * 7;
                break;
            }
            softmax(tmp, 7);

            int id = 0;
            for (int j = 1; j < 7; j++)
                if (tmp[id] < tmp[j]) id = j; // find the maxarg
            TMP.second.push_back(id2label[id]);
            if (id2label[id][0] == 'B') { // token_ids to string
                if (s.length()) TMP.first.push_back(s), s = "";
                s = tokens[i + k * 256];

            } else if (id2label[id][0] == 'I') {
                s += tokens[i + k * 256];

            } else {
                if (s.length()) TMP.first.push_back(s), s = "";
            }
            tmp += 7;
        }
        ans.push_back(TMP);
    }

    return ans;
}

int BERT::get_batch_size() { return batch_size; }