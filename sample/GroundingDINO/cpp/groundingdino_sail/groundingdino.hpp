//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef GROUNDINGDINO_H
#define GROUNDINGDINO_H

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Tokenizer.hpp"
#include "engine.h"
#include "utils.hpp"
#include "cvwrapper.h"

struct Object
{
	cv::Rect box;
	std::string text;
	float prob;
};

static inline float sigmoid(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

class GroundingDINO
{
public:
	GroundingDINO(string bmodel_file, int dev_id, float box_threshold, string vocab_path, float text_threshold);
	vector<Object> detect(sail::BMImage& srcimg, string text_prompt);
	// vector<Object> detect(cv::Mat& srcimg, string text_prompt);
	void enableProfile(TimeStamp* ts);

private:
	void opencv_preprocess(cv::Mat img);
	// void bmcv_preprocess(const bm_image& image);
	void sail_preprocess(sail::BMImage& input);
	void gen_encoder_output_proposals();
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };

	std::shared_ptr<TokenizerBase> tokenizer;
	bool load_tokenizer(std::string vocab_path);

	float* img;
	int* input_ids;
	float* text_token_mask;
	int* token_type_ids;
	float* text_self_attention_masks;
	float* attention_mask;
	int* position_ids;
	float* proposals;

	std::shared_ptr<sail::Engine>              engine;
	std::shared_ptr<sail::Bmcv>                bmcv;
	std::vector<std::string>                   graph_names;    
	std::vector<std::string>                   input_names;    
	std::vector<std::vector<int>>              input_shapes;
	std::vector<std::string>                   output_names;   
	std::vector<std::vector<int>>              output_shapes;
	std::vector<bm_data_type_t>                input_dtypes;    
	std::vector<bm_data_type_t>                output_dtypes;   
	std::map<std::string, std::shared_ptr<sail::Tensor>>       input_tensors; 
	std::map<std::string, sail::Tensor*>       input_tensor_ptrs; 
	std::map<std::string, std::shared_ptr<sail::Tensor>>       output_tensors; 
	std::map<std::string, sail::Tensor*>       output_tensor_ptrs; 

	int m_net_h;
	int m_net_w;
	int max_batch;
	float ab[6];
	int max_text_len;
	
	float box_threshold;
	float text_threshold;
	const char* specical_texts[4] = { "[CLS]", "[SEP]", ".", "?" };
	std::vector<int64> specical_tokens = { 101, 102, 1012, 1029 };
	TimeStamp* m_ts;
};

#endif //!GROUNDINGDINO_H
