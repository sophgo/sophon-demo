//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#define RESIZE_STRATEGY BMCV_INTER_LINEAR
#include "groundingdino.hpp"
#include <fstream>

#define LEFTSTRIP 0
#define RIGHTSTRIP 1
#define BOTHSTRIP 2

std::string do_strip(const std::string &str, int striptype, const std::string&chars)
{
	std::string::size_type strlen = str.size();
	std::string::size_type charslen = chars.size();
	std::string::size_type i, j;

	if (0 == charslen)
	{
		i = 0;
		if (striptype != RIGHTSTRIP)
		{
			while (i < strlen&&::isspace(str[i]))
			{
				i++;
			}
		}
		j = strlen;
		if (striptype != LEFTSTRIP)
		{
			j--;
			while (j >= i && ::isspace(str[j]))
			{
				j--;
			}
			j++;
		}
	}
	else
	{
		const char*sep = chars.c_str();
		i = 0;
		if (striptype != RIGHTSTRIP)
		{
			while (i < strlen&&memchr(sep, str[i], charslen))
			{
				i++;
			}
		}
		j = strlen;
		if (striptype != LEFTSTRIP)
		{
			j--;
			while (j >= i && memchr(sep, str[j], charslen))
			{
				j--;
			}
			j++;
		}
		if (0 == i && j == strlen)
		{
			return str;
		}
		else
		{
			return str.substr(i, j - i);
		}
	}

}

std::string strip(const std::string & str, const std::string & chars = " ")
{
	return do_strip(str, BOTHSTRIP, chars);
}

std::string lstrip(const std::string & str, const std::string & chars = " ")
{
	return do_strip(str, LEFTSTRIP, chars);
}

std::string rstrip(const std::string & str, const std::string & chars = " ")
{
	return do_strip(str, RIGHTSTRIP, chars);
}

int startswith(std::string s, std::string sub) {
	return s.find(sub) == 0 ? 1 : 0;
}

int endswith(std::string s, std::string sub) {
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

GroundingDINO::GroundingDINO(std::string bmodel_file, int dev_id, float box_threshold, string vocab_path, float text_threshold)
{
    engine = std::make_shared<sail::Engine>(dev_id);
    if (!engine->load(bmodel_file)) {
        std::cout << "Engine load bmodel " << bmodel_file << "failed" << endl;
        exit(0);
    }
    // 1. Initialize bmcv
    sail::Handle handle(engine->get_device_id());
    bmcv = std::make_shared<sail::Bmcv>(handle);
    // 2. Initialize engine
    graph_names = engine->get_graph_names();
    std::string gh_info;
    for_each(graph_names.begin(), graph_names.end(), [&](std::string& s) { gh_info += "0: " + s + "; "; });
    std::cout << "grapgh name -> " << gh_info << "\n";
    if (graph_names.size() > 1) {
        std::cout << "NetworkNumError, this net only accept one network!" << std::endl;
        exit(1);
    }
    // input names of network
    input_names = engine->get_input_names(graph_names[0]);
    assert(input_names.size() > 0);
    std::string input_tensor_names;
    for_each(input_names.begin(), input_names.end(), [&](std::string& s) { input_tensor_names += "0: " + s + "; "; });
    std::cout << "net input name -> " << input_tensor_names << "\n";
    // output names of network
    output_names = engine->get_output_names(graph_names[0]);
    assert(output_names.size() > 0);
    std::string output_tensor_names;
    for_each(output_names.begin(), output_names.end(),
             [&](std::string& s) { output_tensor_names += "0: " + s + "; "; });
    std::cout << "net output name -> " << output_tensor_names << "\n";
    // input shapes of network 0
	input_shapes.resize(input_names.size());
	for (int i = 0; i < input_names.size(); i++) {
		input_shapes[i] = engine->get_input_shape(graph_names[0], input_names[i]);
		std::string input_tensor_shape;
		for_each(input_shapes[i].begin(), input_shapes[i].end(), [&](int s) { input_tensor_shape += std::to_string(s) + " "; });
		std::cout << "input tensor " << i << " shape -> " << input_tensor_shape << "\n";
	}
    // output shapes of network 0
    output_shapes.resize(output_names.size());
    for (int i = 0; i < output_names.size(); i++) {
        output_shapes[i] = engine->get_output_shape(graph_names[0], output_names[i]);
        std::string output_tensor_shape;
        for_each(output_shapes[i].begin(), output_shapes[i].end(),
                 [&](int s) { output_tensor_shape += std::to_string(s) + " "; });
        std::cout << "output tensor " << i << " shape -> " << output_tensor_shape << "\n";
    }
    // data type of network input.
	input_dtypes.resize(input_names.size());
	for (int i = 0; i < input_names.size(); i++) {
		input_dtypes[i] = engine->get_input_dtype(graph_names[0], input_names[i]);
		std::cout << "input dtype -> " << input_dtypes[i] << ", is fp32=" << ((input_dtypes[i] == BM_FLOAT32) ? "true" : "false")
				<< "\n";
	}
    // data type of network output.
	output_dtypes.resize(output_names.size());
	for (int i = 0; i < output_names.size(); i++) {
		output_dtypes[i] = engine->get_output_dtype(graph_names[0], output_names[i]);
		std::cout << "output dtype -> " << output_dtypes[i] << ", is fp32=" << ((output_dtypes[i] == BM_FLOAT32) ? "true" : "false")
				<< "\n";
		std::cout << "===============================" << std::endl;
	}
    // 3. Initialize Network IO
	for (int i = 0; i < input_names.size(); i++) {
		input_tensors[input_names[i]] = std::make_shared<sail::Tensor>(handle, input_shapes[i], input_dtypes[i], true, true);
		input_tensor_ptrs[input_names[i]] = input_tensors[input_names[i]].get();
	}
    for (int i = 0; i < output_names.size(); i++) {
        output_tensors[output_names[i]] = std::make_shared<sail::Tensor>(handle, output_shapes[i], output_dtypes[i], true, true);
		output_tensor_ptrs[output_names[i]] = output_tensors[output_names[i]].get();
    }
    engine->set_io_mode(graph_names[0], sail::SYSIO);
    // Initialize net utils
    max_batch = input_shapes[0][0];
    m_net_h = input_shapes[0][2];
    m_net_w = input_shapes[0][3];
    float input_scale = engine->get_input_scale(graph_names[0], input_names[0]);
    ab[0] = 1.0 / (255.0* std[0]) * input_scale;
    ab[1] = (0.0 - mean[0]) / std[0];
    ab[2] = 1.0 / (255.0* std[1]) * input_scale;
    ab[3] = (0.0 - mean[1]) / std[1];
    ab[4] = 1.0 / (255.0* std[2]) * input_scale;
    ab[5] = (0.0 - mean[2]) / std[2];
	max_text_len = input_shapes[3][1];

	this->load_tokenizer(vocab_path);
	this->box_threshold = box_threshold;
	this->text_threshold = text_threshold;

	// input
	this->img = reinterpret_cast<float*>(input_tensors[input_names[0]]->sys_data());
	this->position_ids = reinterpret_cast<int*>(input_tensors[input_names[1]]->sys_data());
	this->text_self_attention_masks = reinterpret_cast<float*>(input_tensors[input_names[2]]->sys_data());
	this->input_ids = reinterpret_cast<int*>(input_tensors[input_names[3]]->sys_data());
	this->token_type_ids = reinterpret_cast<int*>(input_tensors[input_names[4]]->sys_data());
	this->attention_mask = reinterpret_cast<float*>(input_tensors[input_names[5]]->sys_data());
	this->text_token_mask = reinterpret_cast<float*>(input_tensors[input_names[6]]->sys_data());
	this->proposals = reinterpret_cast<float*>(input_tensors[input_names[7]]->sys_data());
}

bool GroundingDINO::load_tokenizer(std::string vocab_path)
{
	tokenizer.reset(new TokenizerClip);
	return tokenizer->load_tokenize(vocab_path);
}

void GroundingDINO::enableProfile(TimeStamp* ts) {
    m_ts = ts;
}

void GroundingDINO::opencv_preprocess(cv::Mat img)
{
	cv::Mat rgbimg;
	cv::cvtColor(img, rgbimg, cv::COLOR_BGR2RGB);
	cv::resize(rgbimg, rgbimg, cv::Size(this->m_net_w, this->m_net_h));
	std::vector<cv::Mat> rgbChannels(3);
	cv::split(rgbimg, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0* std[c]), (0.0 - mean[c]) / std[c]);
	}

	const int image_area = m_net_h * m_net_w;
	size_t single_chn_size = image_area * sizeof(float);
	memcpy(this->img, (float*)rgbChannels[0].data, single_chn_size);
	memcpy(this->img + image_area, (float*)rgbChannels[1].data, single_chn_size);
	memcpy(this->img + image_area * 2, (float*)rgbChannels[2].data, single_chn_size);
}

/*
void GroundingDINO::bmcv_preprocess(const bm_image& image)
{
	std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
	//1. resize image
	int ret = 0;
	bm_image image1 = image;
	bm_image image_aligned;
	bool need_copy = image1.width & (64-1);
	if(need_copy){
		int stride1[3], stride2[3];
		bm_image_get_stride(image1, stride1);
		stride2[0] = FFALIGN(stride1[0], 64);
		stride2[1] = FFALIGN(stride1[1], 64);
		stride2[2] = FFALIGN(stride1[2], 64);
		bm_image_create(bm_ctx->handle(), image1.height, image1.width,
			image1.image_format, image1.data_type, &image_aligned, stride2);

		bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
		bmcv_copy_to_atrr_t copyToAttr;
		memset(&copyToAttr, 0, sizeof(copyToAttr));
		copyToAttr.start_x = 0;
		copyToAttr.start_y = 0;
		copyToAttr.if_padding = 1;
		bmcv_image_copy_to(bm_ctx->handle(), copyToAttr, image1, image_aligned);
	} else {
		image_aligned = image1;
	}
    ret = bmcv_image_vpp_convert(bm_ctx->handle(), 1, image_aligned, &m_resized_imgs[i]);
    assert(BM_SUCCESS == ret);
    if(need_copy) bm_image_destroy(image_aligned);
  
	//2. converto
	ret = bmcv_image_convert_to(bm_ctx->handle(), 1, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
	CV_Assert(ret == 0);

	//3. attach to tensor
	bm_device_mem_t input_dev_mem;
	bm_image_get_contiguous_device_mem(1, m_converto_imgs.data(), &input_dev_mem);
	input_tensor->set_device_mem(&input_dev_mem);
	input_tensor->set_shape_by_dim(0, 1);  // set real batch number
}
*/

void GroundingDINO::sail_preprocess(sail::BMImage& input) {
    int ret = 0;
    sail::BMImage rgb_img(engine->get_handle(), input.height(), input.width(), FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE);
    rgb_img.align();
    bmcv->convert_format(input, rgb_img);
    sail::BMImage convert_img(engine->get_handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                              bmcv->get_bm_image_data_format(input_dtypes[0]));
    sail::BMImage resized_img(engine->get_handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                               DATA_TYPE_EXT_1N_BYTE);
    resized_img.align();

	// resize
    ret = bmcv->crop_and_resize(rgb_img, resized_img, 0, 0, rgb_img.width(), rgb_img.height(), m_net_w, m_net_h, RESIZE_STRATEGY);
    CV_Assert(ret == 0);

    bmcv->convert_to(
        resized_img, convert_img,
        std::make_tuple(std::make_pair(ab[0], ab[1]), std::make_pair(ab[2], ab[3]), std::make_pair(ab[4], ab[5])));
    bmcv->bm_image_to_tensor(convert_img, *input_tensors[input_names[0]].get());
	input_tensors[input_names[0]]->sync_d2s();
}

void GroundingDINO::gen_encoder_output_proposals()
{
	vector<vector<int>> spatial_shapes{{100, 100}, {50, 50}, {25, 25}, {13, 13}};
	int indx = 0;
	for(int i = 0; i < spatial_shapes.size(); i++)
	{
		int valid_H = spatial_shapes[i][0], valid_W = spatial_shapes[i][1];
		float wh = 0.05 * pow(2.0, i);
		for(int h_i = 0; h_i < spatial_shapes[i][0]; h_i++) {
			for(int w_i = 0; w_i < spatial_shapes[i][1]; w_i++) {
				this->proposals[indx] = (float(w_i) + 0.5) / valid_W;
				indx += 1;
				this->proposals[indx] = (float(h_i) + 0.5) / valid_H;
				indx += 1;
				this->proposals[indx] = wh;
				indx += 1;
				this->proposals[indx] = wh;
				indx += 1;
			}
		}
	}
}

vector<Object> GroundingDINO::detect(sail::BMImage& srcimg, string text_prompt)
{
	m_ts->save("groundingdino preprocess");
	std::vector<Object> objects;
	const int srch = srcimg.height(), srcw = srcimg.width();
	this->sail_preprocess(srcimg);
	
	std::transform(text_prompt.begin(), text_prompt.end(), text_prompt.begin(), ::tolower); ////תСд
	string caption = strip(text_prompt); ////ȥ����β�ո��
	while (!caption.empty() && endswith(caption, ".") == 1) {
		caption.pop_back();
	}
	if (caption.empty()) 
		return objects;

	std::vector<int64> ids;
	tokenizer->encode_text(caption, ids);
	int len_ids = ids.size();
	int trunc_len = len_ids <= this->max_text_len ? len_ids : this->max_text_len;
	for (int i = 0; i < trunc_len; i++) {
		input_ids[i] = ids[i];
		token_type_ids[i] = 0;
		text_token_mask[i] = ids[i] > 0 ? true : false;
	}
	for (int i = trunc_len; i < this->max_text_len; i++) {
		input_ids[i] = 0;
		token_type_ids[i] = 0;
		text_token_mask[i] = false;
	}
	
	////generate_masks_with_special_tokens_and_transfer_map
	////const int bs = input_ids.size();
	const int num_token = trunc_len;
	vector<int> idxs; 
	for (int i = 0; i < num_token; i++) {
		for (int j = 0; j < this->specical_tokens.size(); j++) {
			if (input_ids[i] == this->specical_tokens[j]) {
				idxs.push_back(i);
			}
		}
	}
	
	len_ids = idxs.size();
	trunc_len = num_token <= this->max_text_len ? num_token : this->max_text_len;
	for (int i = 0; i < this->max_text_len; i++)
	{
		for (int j = 0; j < this->max_text_len; j++)
		{
			text_self_attention_masks[i*this->max_text_len + j] = (i == j ? true : false);
		}
		position_ids[i] = 0;
	}
	int previous_col = 0;
	for (int i = 0; i < len_ids; i++) {
		const int col = idxs[i];
		if (col == 0 || col == num_token - 1) {
			text_self_attention_masks[col*this->max_text_len + col] = true;
			position_ids[col] = 0;
		}
		else {
			for (int j = previous_col + 1; j <= col; j++) {
				for (int k = previous_col + 1; k <= col; k++) {
					text_self_attention_masks[j*this->max_text_len + k] = true;
				}
				position_ids[j] = j - previous_col - 1;

			}
		}
		previous_col = col;
	}

	gen_encoder_output_proposals();
	memcpy(attention_mask, text_self_attention_masks, this->max_text_len*this->max_text_len*sizeof(float));
	m_ts->save("groundingdino preprocess");
	
	m_ts->save("groundingdino inference");
	for (auto input_name: input_names) {
		input_tensors[input_name]->sync_s2d();
	}
	/*
	// write input data to txt files
	std::ofstream ofile;
	ofile.open(input_names[0]);
	for (int j = 0; j < 3*800*800; j++) {
		ofile << this->img[j] << " ";
	}
	ofile.close();
	ofile.open(input_names[1]);
	for (int j = 0; j < 256; j++) {
		ofile << this->position_ids[j] << " ";
	}
	ofile.close();
	ofile.open(input_names[2]);
	for (int j = 0; j < 256*256; j++) {
		ofile << this->text_self_attention_masks[j] << " ";
	}
	ofile.close();
	ofile.open(input_names[3]);
	for (int j = 0; j < 256; j++) {
		ofile << this->input_ids[j] << " ";
	}
	ofile.close();
	ofile.open(input_names[4]);
	for (int j = 0; j < 256; j++) {
		ofile << this->token_type_ids[j] << " ";
	}
	ofile.close();
	ofile.open(input_names[5]);
	for (int j = 0; j < 256*256; j++) {
		ofile << this->attention_mask[j] << " ";
	}
	ofile.close();
	ofile.open(input_names[6]);
	for (int j = 0; j < 256; j++) {
		ofile << this->text_token_mask[j] << " ";
	}
	ofile.close();
	ofile.open(input_names[7]);
	for (int j = 0; j < 13294*4; j++) {
		ofile << this->proposals[j] << " ";
	}
	ofile.close();
	*/
	engine->process(graph_names[0], input_tensor_ptrs, output_tensor_ptrs);
	for (auto output_name: output_names) {
		output_tensors[output_name]->sync_d2s();
	}
	m_ts->save("groundingdino inference");

	m_ts->save("groundingdino postprocess");
	const float *ptr_logits = reinterpret_cast<float*>(output_tensors[output_names[0]]->sys_data());
	std::vector<int> logits_shape = output_tensors[output_names[0]]->shape();
	const float *ptr_boxes = reinterpret_cast<float*>(output_tensors[output_names[1]]->sys_data()); ////cx,cy,w,h
	std::vector<int> boxes_shape = output_tensors[output_names[1]]->shape();
	const int outh = logits_shape[1];
	const int outw = logits_shape[2];
	vector<int> filt_inds;
	vector<float> scores;
	for (int i = 0; i < outh; i++)
	{
		float max_data = 0;
		for (int j = 0; j < outw; j++)
		{
			float x = sigmoid(ptr_logits[i*outw + j]);
			if (max_data < x)
			{
				max_data = x;
			}
		}
		if (max_data > this->box_threshold)
		{
			filt_inds.push_back(i);
			scores.push_back(max_data);
		}
	}

	for (int i = 0; i < filt_inds.size(); i++)
	{
		////get_phrases_from_posmap
		const int ind = filt_inds[i];
		const int left_idx = 0, right_idx = 255;
		for (int j = left_idx+1; j < right_idx; j++)
		{
			float x = sigmoid(ptr_logits[ind*outw + j]);
			if (x > this->text_threshold)
			{
				const int64 token_id = input_ids[j];
				Object obj;
				obj.text = this->tokenizer->tokenizer_idx2token[token_id];  
				obj.prob = scores[i];
				int xmin = int((ptr_boxes[ind * 4] - ptr_boxes[ind * 4 + 2] * 0.5)*srcw);
				int ymin = int((ptr_boxes[ind * 4 + 1] - ptr_boxes[ind * 4 + 3] * 0.5)*srch);
				///int xmax = int((ptr_boxes[ind * 4] + ptr_boxes[ind * 4 + 2] * 0.5)*srcw);
				///int ymax = int((ptr_boxes[ind * 4 + 1] + ptr_boxes[ind * 4 + 3] * 0.5)*srch);
				int w = int(ptr_boxes[ind * 4 + 2] * srcw);
				int h = int(ptr_boxes[ind * 4 + 3] * srch);
				obj.box = cv::Rect(xmin, ymin, w, h);
				objects.push_back(obj);

				break; 
			}
		}
	}
	m_ts->save("groundingdino postprocess");
	return objects;
}