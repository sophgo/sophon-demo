//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "qwen.hpp"
#include <fstream>
#include "utils.hpp"

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, bm_mem_get_device_size(src));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen::load_sentencepiece(std::string tokenizer_path) {
  printf("Load %s ... ", tokenizer_path.c_str());
  // Read blob from file.
  auto blob = LoadBytesFromFile(tokenizer_path);
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  tok = Tokenizer::FromBlobJSON(blob);
  EOS = tok->TokenToId("<|endoftext|>");
  ID_IM_END = tok->TokenToId("<|im_end|>");
  printf("Done!\n");
}

void Qwen::decode_tokens(std::vector<int>& tokens, std::string& word) {
  word = tok->Decode(tokens);
}

std::string Qwen::build_prompt(std::string query, std::vector<std::pair<std::string, std::string>>& history_vector) {
    std::string prompt = sys_config;
    for (const auto& item : history_vector) {
      prompt += "<|im_start|>assistant\n" + item.first + 
                  "<|im_end|>\n<|im_start|>user\n" + item.second + "<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

void Qwen::encode_tokens(std::string& prompt, std::vector<int>& tokens) {
  tokens = tok->Encode(prompt);
}

std::pair<bool, bool> Qwen::is_end(int token) {
    return std::pair<bool, bool>({token == EOS || token == ID_IM_END, token_length >= SEQLEN});
}

int Qwen::get_max_length() {
    return SEQLEN;
}

void Qwen::init(std::string bmodel_path, const std::vector<int> &dev_ids, std::string tokenizer_path) {
  version = "1.1.1";

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : dev_ids) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : dev_ids) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", bmodel_path.c_str());
  bool ret = false;
  ret = bmrt_load_bmodel(p_bmrt, bmodel_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head = bmrt_get_network_info(p_bmrt, "penalty_sample_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 5) / 2;
  generation_mode = "greedy";

  // resize
  visited_tokens.resize(SEQLEN);

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  hidden_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]);
  kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  is_dynamic = net_blocks[0]->is_dynamic;
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    if (io_alone) {
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    } else {
      auto ret = bm_malloc_device_byte(bm_handle, &past_key[i],
                                       net_blocks_cache[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret = bm_malloc_device_byte(bm_handle, &past_value[i],
                                  net_blocks_cache[i]->max_input_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
  }

  // load tokenizer
  load_sentencepiece(tokenizer_path);

  // init prompt
  sys_config = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n";
}

void Qwen::deinit() {
  if (false == io_alone) {
    for (int i = 0; i < NUM_LAYERS; i++) {
      bm_free_device(bm_handle, past_key[i]);
      bm_free_device(bm_handle, past_value[i]);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void Qwen::net_launch(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen::net_launch_dyn(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }

  int h_bytes = bm_mem_get_device_size(in_tensors[0].device_mem) / SEQLEN;
  bm_set_device_mem(&in_tensors[0].device_mem,
                    h_bytes * TOKEN_LEN,
                    bm_mem_get_device_addr(in_tensors[0].device_mem));
  int pid_bytes = bm_mem_get_device_size(in_tensors[1].device_mem) / SEQLEN;
  bm_set_device_mem(&in_tensors[1].device_mem,
                    pid_bytes * TOKEN_LEN,
                    bm_mem_get_device_addr(in_tensors[1].device_mem));
  int mask_bytes = bm_mem_get_device_size(in_tensors[2].device_mem) / SEQLEN / SEQLEN;
  bm_set_device_mem(&in_tensors[2].device_mem,
                    mask_bytes * TOKEN_LEN * TOKEN_LEN,
                    bm_mem_get_device_addr(in_tensors[2].device_mem));

  in_tensors[0].shape.dims[1] = TOKEN_LEN;
  in_tensors[1].shape.dims[1] = TOKEN_LEN;
  in_tensors[2].shape.dims[2] = TOKEN_LEN;
  in_tensors[2].shape.dims[3] = TOKEN_LEN;

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(
      &in_tensors[0], logits_mem,
      net->input_dtypes[0], net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[0].input_mems[i],
        net->input_dtypes[i], net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[0].output_mems[i],
        net->output_dtypes[i], net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

int Qwen::greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Qwen::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN, visited_tokens[token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, token_length);
  std::copy(visited_tokens.begin() + token_length - repeat_last_n, 
            visited_tokens.begin() + token_length,
            generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int Qwen::forward_first(std::vector<int> &tokens) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();
  TOKEN_LEN = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  if (is_dynamic) {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j < TOKEN_LEN; j++) {
        if (j <= i) {
          attention_mask[i * TOKEN_LEN + j] = 0;
        }
      }
    }
  } else {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j < SEQLEN; j++) {
        if (j <= i) {
          attention_mask[i * SEQLEN + j] = 0;
        }
      }
    }
  }
  // empty
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i]);
    empty_net(bm_handle, net_blocks_cache[i]);
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed); // prefil embedding

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    empty(bm_handle, net_blocks[idx]->stages[0].input_mems[0]);
    d2d(in0_mem, out_mem, 0, token_length * hidden_bytes);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    if (is_dynamic) net_launch_dyn(net_blocks[idx]);
    else net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1], 0,
        token_length * kv_bytes);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2], 0,
        token_length * kv_bytes);

  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * hidden_bytes, hidden_bytes);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int Qwen::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;
  // embedding
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);

  // blocks
  int token_offset = (token_length - 1) * kv_bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      } else {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    } else {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       kv_bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       kv_bytes);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

void Qwen::answer(std::string input_str, std::vector<std::pair<std::string, std::string>>& history_vector) {
  std::string sentence_input = build_prompt(input_str, history_vector);

  int tok_num = 1;
  std::vector<int> tokens;
  encode_tokens(sentence_input, tokens);
  if (int(tokens.size()) >= get_max_length() - 10) {
    std::cout << "The tokens you input exceeds MAX SEQ LENGTH" << std::endl;
    return;
  }

  int pre_token = 0;
  auto t0 = std::chrono::system_clock::now();
  int token = forward_first(tokens);
  auto t1 = std::chrono::system_clock::now();
  std::string result;
  auto end_flag = is_end(token);
  while (!(end_flag.first || end_flag.second)) {
    std::string pre_word;
    std::string word;
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    decode_tokens(pre_ids, pre_word);
    decode_tokens(ids, word);
    std::string diff = word.substr(pre_word.size());
    result += diff;
    std::cout << diff << std::flush;
    tok_num++;
    token = forward_next();
    end_flag = is_end(token);
  }
  auto t2 = std::chrono::system_clock::now();
  auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  printf("\n\nfirst token latency: %f s", (use0.count() * 1e-6));
  printf("\nspeed: %f token/s\n", tok_num / (use1.count() * 1e-6));
  if (end_flag.second) {
    history_vector.push_back({input_str, result}); 
    result.clear();

    size_t half_size = history_vector.size() / 2;
    printf("length exceed max sequence length, now will delete the half %f", float(half_size));
    history_vector.erase(history_vector.begin(), history_vector.begin() + half_size);
  } else {
    history_vector.push_back({input_str, result});
    result.clear();
  }
}
