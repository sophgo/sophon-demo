//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "vila.hpp"
#define IO_ALONE 1
static const uint16_t ATTENTION_MASK_VALUE = 0xF0E2;

inline uint16_t fp32_to_fp16_bits(float f) {
  uint32_t x = *((uint32_t *)&f);
  uint16_t h = ((x >> 16) & 0x8000) |
               ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
               ((x >> 13) & 0x03ff);

  return h;
}

void string_split(std::vector<std::string>& res, const std::string& str, const std::string& splits)
{
	if (str == "")
    return;

  // add splits to end
	std::string strs = str + splits;
	size_t pos = strs.find(splits);
	int step = splits.size();

	while (pos != strs.npos)
	{
		std::string temp = strs.substr(0, pos);
		res.push_back(temp);
    // delete
		strs = strs.substr(pos + step, strs.size());
		pos = strs.find(splits);
	}
}

void VILA::opencv_extract_frames(std::vector<cv::Mat>& images, std::string video_file, int num_frames) {
  // open video
  auto vidcap = cv::VideoCapture(video_file, cv::CAP_FFMPEG, devids[0]);
  if (!vidcap.isOpened()){
    std::cerr << "Error: open video src failed in channel " << std::endl;
    exit(1);
  }

  // get frames
  int fps = vidcap.get(cv::CAP_PROP_FPS);
  int frame_count = (int)vidcap.get(cv::CAP_PROP_FRAME_COUNT);
  std::vector<int> frame_indices;
  frame_indices.push_back(0);
  for (int i = 1; i < num_frames - 1; i++) {
    frame_indices.push_back((int)((float)(frame_count - 1.0) / (num_frames - 1.0) * i));
  }
  frame_indices.push_back(frame_count-1);

  int count = 0;
  while (true) {
    cv::Mat image;
    if (frame_count >= num_frames) {
      vidcap.read(image);
      auto it = std::find(frame_indices.begin(), frame_indices.end(), count);
      if (it != frame_indices.end()) {
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        images.push_back(rgb_image);
        if (images.size() >= num_frames)
          break;
      }
      count += 1;
    }
    else {
      vidcap.read(image);
      if (image.empty()) {
        break;
      }
      cv::Mat rgb_image;
      cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
      images.push_back(rgb_image);
      count += 1;
    }
  }
}

void VILA::process_images(std::vector<std::vector<float>>& processed_images, std::vector<cv::Mat>& images) {
  for (auto& image : images) {
    // preprocess
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(crop_size[1], crop_size[0]), 0, 0, cv::INTER_AREA);
    resized_img.convertTo(resized_img, CV_32FC1, 0.00392156862745098, 0);
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(resized_img, rgbChannels);
    for (int c = 0; c < 3; c++)
    {
      rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / std[c], (0.0 - mean[c]) / std[c]);
    }

    // convert to array
    const int image_area = crop_size[1] * crop_size[0];
    size_t single_chn_size = image_area * sizeof(float);
    std::vector<float> process_image;
    process_image.resize(image_area * 3);
    memcpy((void*)process_image.data(), (float*)rgbChannels[0].data, single_chn_size);
    memcpy((void*)(process_image.data() + image_area), (float*)rgbChannels[1].data, single_chn_size);
    memcpy((void*)(process_image.data() + image_area * 2), (float*)rgbChannels[2].data, single_chn_size);
    processed_images.push_back(process_image);
  }
}

void VILA::load_sentencepiece(std::string tokenizer_path) {
  printf("Load %s ... ", tokenizer_path.c_str());
  auto status = sentencepiece.Load(tokenizer_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  SOS = sentencepiece.bos_id();
  printf("Done!\n");
}

VILA::VILA(std::string video_path, const std::vector<int> devids, std::string llm_model_path, std::string vision_model_path, std::string tokenizer_path) {
  // image process
  crop_size.push_back(384); // height
  crop_size.push_back(384); // width
  mean.push_back(0.5);
  mean.push_back(0.5);
  mean.push_back(0.5);
  std.push_back(0.5);
  std.push_back(0.5);
  std.push_back(0.5);

  // init handles
  for (auto dev_id : devids) {
    handles.emplace(std::pair<int, sail::Handle>(dev_id, sail::Handle(dev_id)));
  }
  
  // load tokenizer
  load_sentencepiece(tokenizer_path);

  // init models
  this->devids = devids;
  llm_model = std::make_shared<sail::EngineLLM>(llm_model_path, devids);
  vision_model = std::make_shared<sail::EngineLLM>(vision_model_path, devids);

  // graph
  llm_graph_names = llm_model->get_graph_names();
  num_layers = (llm_graph_names.size() - 3) / 2;

  // create block names
  for (int i = 0; i < num_layers; i++) {
    name_block.push_back("block_" + std::to_string(i));
    name_block_cache.push_back("block_cache_" + std::to_string(i));
  }

  // init vision's input and output
  input_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_vision_embed, vision_model->create_max_input_tensors(name_vision_embed)));
  output_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_vision_embed, vision_model->create_max_output_tensors(name_vision_embed)));

  // init other vision val
  auto output_vision_embed_shape = output_tensors.at(name_vision_embed).at(0)->shape();
  num_frames = output_vision_embed_shape[0];
  vision_token_length = output_vision_embed_shape[1];

  // video preprocess
  std::vector<cv::Mat> images;
  opencv_extract_frames(images, video_path, num_frames);
  std::vector<std::vector<float>> processed_images;
  process_images(processed_images, images);

  // prepare vision input and process
  auto vision_embed_input_mem = input_tensors.at(name_vision_embed).at(0)->dev_data();
  int vision_embed_input_mem_offset = 0;
  for (auto& processed_image : processed_images) {
    std::vector<uint16_t> processed_image_fp16;
    for (int i = 0; i < processed_image.size(); i++) {
      processed_image_fp16.push_back(fp32_to_fp16_bits(processed_image[i]));
    }
    bm_memcpy_s2d_partial_offset(handles[vision_model->get_input_tensor_devid(name_vision_embed, 0)].data(), vision_embed_input_mem, (void*)processed_image_fp16.data(), sizeof(uint16_t)*processed_image_fp16.size(), vision_embed_input_mem_offset);
    vision_embed_input_mem_offset += sizeof(uint16_t)*processed_image_fp16.size();
  }
  vision_model->process(name_vision_embed, input_tensors.at(name_vision_embed), output_tensors.at(name_vision_embed));

  // init other llm val
  auto input_block_shape = llm_model->get_input_shape(name_block[0], 0);
  seqlen = input_block_shape[1];
  hidden_size = input_block_shape[2];
  auto output_block_shape = llm_model->get_output_shape(name_block[0], 1);
  num_head = output_block_shape[2];
  head_dim = output_block_shape[3];
  token_length = 0;

  // init llm's input and output
  input_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_llm_embed, llm_model->create_max_input_tensors(name_llm_embed)));
  output_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_llm_embed, llm_model->create_max_output_tensors(name_llm_embed)));
  auto block_input_tensors = llm_model->create_max_input_tensors(name_block[0]);
  for (int i = 0; i < num_layers; i++) {
    input_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_block[i], std::map<int, sail::Tensor*>()));
    output_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_block[i], std::map<int, sail::Tensor*>()));
    input_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(0, block_input_tensors.at(0)));
    input_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(1, block_input_tensors.at(1)));
    input_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(2, block_input_tensors.at(2)));
    output_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(0, block_input_tensors.at(0)));
#if IO_ALONE
    output_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(1, llm_model->get_input_tensors(name_block_cache[i]).at(3)));
    output_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(2, llm_model->get_input_tensors(name_block_cache[i]).at(4)));
#else
    first_k = std::make_shared<sail::Tensor>(
          handles.at(llm_model->get_output_tensor_devid(name_block_cache[i], 3)), 
          llm_model->get_output_shape(name_block_cache[i], 3), 
          llm_model->get_output_dtype(name_block_cache[i], 3), false, true);
    first_v = std::make_shared<sail::Tensor>(
          handles.at(llm_model->get_output_tensor_devid(name_block_cache[i], 4)), 
          llm_model->get_output_shape(name_block_cache[i], 4), 
          llm_model->get_output_dtype(name_block_cache[i], 4), false, true);
    output_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(1, first_k.get()));
    output_tensors.at(name_block[i]).emplace(std::pair<int, sail::Tensor*>(2, first_v.get()));
#endif
  }
  input_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_lm, llm_model->create_max_input_tensors(name_lm)));
  output_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_lm, llm_model->create_max_output_tensors(name_lm)));
  input_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_llm_embed_cache, output_tensors.at(name_lm)));

  output_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_llm_embed_cache, llm_model->create_max_output_tensors(name_llm_embed_cache)));
  position_ids_next = std::make_shared<sail::Tensor>(
          handles.at(llm_model->get_input_tensor_devid(name_block_cache[num_layers-1], 1)), 
          llm_model->get_input_shape(name_block_cache[num_layers-1], 1), 
          llm_model->get_input_dtype(name_block_cache[num_layers-1], 1), false, true);
  attention_mask_next = std::make_shared<sail::Tensor>(
          handles.at(llm_model->get_input_tensor_devid(name_block_cache[num_layers-1], 2)), 
          llm_model->get_input_shape(name_block_cache[num_layers-1], 2), 
          llm_model->get_input_dtype(name_block_cache[num_layers-1], 2), false, true);
  
  for (int i = 0; i < num_layers; i++) {
    input_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_block_cache[i], std::map<int, sail::Tensor*>()));
    output_tensors.emplace(std::pair<std::string, std::map<int, sail::Tensor*>>(name_block_cache[i], std::map<int, sail::Tensor*>()));
    input_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(0, output_tensors.at(name_llm_embed_cache).at(0)));
    input_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(1, position_ids_next.get()));
    input_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(2, attention_mask_next.get()));
    input_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(3, output_tensors.at(name_block[i]).at(1)));
    input_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(4, output_tensors.at(name_block[i]).at(2)));
    output_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(0, output_tensors.at(name_llm_embed_cache).at(0)));
    past_k.push_back(std::make_shared<sail::Tensor>(
          handles[llm_model->get_output_tensor_devid(name_block_cache[i], 1)], 
          llm_model->get_output_shape(name_block_cache[i], 1), 
          llm_model->get_output_dtype(name_block_cache[i], 1), false, true));
    past_v.push_back(std::make_shared<sail::Tensor>(
          handles[llm_model->get_output_tensor_devid(name_block_cache[i], 2)], 
          llm_model->get_output_shape(name_block_cache[i], 2), 
          llm_model->get_output_dtype(name_block_cache[i], 2), false, true));
    output_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(1, past_k[i].get()));
    output_tensors.at(name_block_cache[i]).emplace(std::pair<int, sail::Tensor*>(2, past_v[i].get()));
  }
}

VILA::~VILA() {
  std::cout << "VILA dtor ..." << std::endl;
}

int VILA::forward_first(std::vector<int> &tokens) {
  // init val
  token_length = tokens.size() + (vision_token_length - 1) * num_frames;
  std::vector<int> input_ids(seqlen, 0);
  std::vector<int> image_index;
  image_index.push_back(-1);
  for (int i = 0; i < tokens.size(); i++) {
    if (tokens[i] == -200) {
      image_index.push_back(i);
      input_ids[i] = 0;
    }
    else {
      input_ids[i] = tokens[i];
    }
  }
  image_index.push_back(tokens.size());

  // prepare input and process
  auto llm_embed_input_mem = input_tensors.at(name_llm_embed).at(0)->dev_data();
  bm_memcpy_s2d_partial(handles[llm_model->get_input_tensor_devid(name_llm_embed, 0)].data(), llm_embed_input_mem, (void*)input_ids.data(), sizeof(int)*input_ids.size());
  llm_model->process(name_llm_embed, input_tensors.at(name_llm_embed), output_tensors.at(name_llm_embed));

  // image d2d
  int offset = 0;
  for (int i = 0; i < image_index.size() - 1; i++) {
    input_tensors.at(name_block[0]).at(0)->sync_d2d(
        *output_tensors.at(name_llm_embed).at(0), 
        (image_index[i] + 1) * hidden_size,
        offset,
        (image_index[i + 1] - image_index[i] - 1) * hidden_size);
    offset += (image_index[i + 1] - image_index[i] - 1) * hidden_size;
    if (i < num_frames) {
      input_tensors.at(name_block[0]).at(0)->sync_d2d(
          *output_tensors.at(name_vision_embed).at(0), 
          i * vision_token_length * hidden_size,
          offset,
          vision_token_length * hidden_size);
      offset += vision_token_length * hidden_size;
    }
  }

  // prepare input and process
  std::vector<int> position_id(seqlen, 0);
  std::vector<uint16_t> attention_mask(seqlen * seqlen, ATTENTION_MASK_VALUE);
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < i + 1; j++) {
      attention_mask[i * seqlen + j] = 0;
    }
  }
  auto block_input_mem = input_tensors.at(name_block[0]).at(1)->dev_data();
  bm_memcpy_s2d_partial(handles[llm_model->get_input_tensor_devid(name_block[0], 1)].data(), block_input_mem, (void*)position_id.data(), sizeof(int)*position_id.size());
  auto att_mask_input_mem = input_tensors.at(name_block[0]).at(2)->dev_data();
  bm_memcpy_s2d_partial(handles[llm_model->get_input_tensor_devid(name_block[0], 2)].data(), att_mask_input_mem, (void *)attention_mask.data(), sizeof(uint16_t)*attention_mask.size());
  for (int i = 0; i < num_layers; i++) {
    llm_model->process(name_block[i], input_tensors.at(name_block[i]), output_tensors.at(name_block[i]));
  }
  input_tensors.at(name_lm).at(0)->sync_d2d(*output_tensors.at(name_block[num_layers - 1]).at(0), (token_length - 1) * hidden_size, 0, hidden_size);
  llm_model->process(name_lm, input_tensors.at(name_lm), output_tensors.at(name_lm));
  
  int res_token_id;
  bm_memcpy_d2s_partial(handles[llm_model->get_output_tensor_devid(name_lm, 0)].data(), (void*)&res_token_id, output_tensors.at(name_lm).at(0)->dev_data(), sizeof(int));
  return res_token_id;
}

int VILA::forward_next() {
  // prepare input and process
  token_length += 1;
  std::vector<int> position_id;
  position_id.push_back(token_length - 1);
  std::vector<uint16_t> attention_mask(seqlen + 1, 0);
  for (int i = token_length - 1; i < seqlen; i++) {
    attention_mask[i] = ATTENTION_MASK_VALUE;
  }
  llm_model->process(name_llm_embed_cache, input_tensors.at(name_llm_embed_cache), output_tensors.at(name_llm_embed_cache));
  auto block_cache_input_mem = input_tensors.at(name_block_cache[0]).at(1)->dev_data();
  bm_memcpy_s2d_partial(handles[llm_model->get_input_tensor_devid(name_block_cache[0], 1)].data(), block_cache_input_mem, (void*)position_id.data(), sizeof(int)*position_id.size());
  auto att_mask_input_mem = input_tensors.at(name_block_cache[0]).at(2)->dev_data();
  bm_memcpy_s2d_partial(handles[llm_model->get_input_tensor_devid(name_block_cache[0], 2)].data(), att_mask_input_mem, (void*)attention_mask.data(), sizeof(uint16_t)*attention_mask.size());
  for (int i = 0; i < num_layers; i++) {
    llm_model->process(name_block_cache[i], input_tensors.at(name_block_cache[i]), output_tensors.at(name_block_cache[i]));
    input_tensors.at(name_block_cache[i]).at(3)->sync_d2d(*output_tensors.at(name_block_cache[i]).at(1), 0, (token_length - 1) * num_head * head_dim, num_head * head_dim);
    input_tensors.at(name_block_cache[i]).at(4)->sync_d2d(*output_tensors.at(name_block_cache[i]).at(2), 0, (token_length - 1) * num_head * head_dim, num_head * head_dim);
  }
  input_tensors.at(name_lm).at(0) = output_tensors.at(name_block_cache[num_layers - 1]).at(0);
  llm_model->process(name_lm, input_tensors.at(name_lm), output_tensors.at(name_lm));

  int res_token_id;
  bm_memcpy_d2s_partial(handles[llm_model->get_output_tensor_devid(name_lm, 0)].data(), (void*)&res_token_id, output_tensors.at(name_lm).at(0)->dev_data(), sizeof(int));
  return res_token_id;
}

void VILA::decode(std::string& word, std::vector<int>& tokens) {
  sentencepiece.Decode(tokens, &word);
}

int VILA::get_eos_id() {
  return EOS;
}

void VILA::tokenizer_image_token(std::vector<int>& input_ids, std::string& prompt, int image_token_index, bool lstrip) {
  // token2id
  std::vector<std::string> split_strings;
  std::vector<std::vector<int>> prompt_chunks;
  string_split(split_strings, prompt, "<image>");
  for (auto& split_string : split_strings) {
    split_string = " " + split_string;
    std::vector<int> chunk;
    sentencepiece.Encode(split_string, &chunk);
    chunk.insert(chunk.begin(), SOS);
    prompt_chunks.push_back(chunk);
  }

  int offset = 0;
  if (lstrip) 
    offset = 1;
  else {
    if (prompt_chunks.size() > 0 && prompt_chunks[0].size() > 0 && prompt_chunks[0][0] == SOS) {
      offset = 1;
      input_ids.push_back(prompt_chunks[0][0]);
    }
  }

  std::vector<int> sep(offset + 1, image_token_index);
  std::vector<std::vector<int>> insert_separator;
  for (int chunk_id = 0; chunk_id < prompt_chunks.size(); chunk_id++) {
    insert_separator.push_back(prompt_chunks[chunk_id]);
    if (chunk_id != prompt_chunks.size() - 1)
      insert_separator.push_back(sep);
  }
  for (int chunk_id = 0; chunk_id < insert_separator.size(); chunk_id++) {
    if (chunk_id == 0 && lstrip) 
      for (int j = 0; j < insert_separator[chunk_id].size(); j++) {
        input_ids.push_back(insert_separator[chunk_id][j]);
      }
    else
      for (int j = offset; j < insert_separator[chunk_id].size(); j++) {
        input_ids.push_back(insert_separator[chunk_id][j]);
      }
  }
}
