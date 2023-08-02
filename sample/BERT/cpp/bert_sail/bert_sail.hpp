//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1
#include <stdio.h>
#include <sail/cvwrapper.h>
#include <sail/tensor.h>
#include <sail/engine.h>
#include <iostream>
#include <string>
#include "tokenizer.h"
class BERT {
public:
    BERT(std::string model_path, std::string pre_train_path, int id);
    void pre_process(std::string text);
    void pre_process(std::vector<string> texts);
    bool Detect();
    std::vector<pair<std::vector<std::string>, std::vector<std::string>>>
    post_process();
    void softmax(float* x, int length);
    int get_batch_size();

private:
    sail::IOMode mode;
    std::shared_ptr<sail::Handle> handle;
    std::shared_ptr<sail::Bmcv> bmcv;
    std::shared_ptr<sail::Engine> engine;
    int device_id;
    std::string graph_name;
    std::string input_name;
    std::vector<string> output_names;
    std::vector<int> input_shape;
    bm_data_type_t input_dtype;
    float input_scale;
    int batch_size;
    std::vector<std::vector<int>> output_shape;
    std::vector<bm_data_type_t> output_dtype;
    bm_image_data_format_ext img_dtype;
    std::vector<float> output_scale;
    BertTokenizer tokenizer;
    sail::Tensor input_tensor;
    std::vector<sail::Tensor> output_tensors;
    std::vector<std::string> tokens;
    std::vector<float> token_ids;
    std::vector<std::string> id2label;
};
