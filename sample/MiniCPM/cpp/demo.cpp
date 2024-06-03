//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include "memory.h"
#include "sentencepiece/sentencepiece_processor.h"
#include "bmruntime_interface.h"
#include <getopt.h>

static const uint16_t ATTENTION_MASK = 0xC61C;

class MiniCPM {
public:
  void init(std::string model_path, std::string tokenizer_path, const std::vector<int> &devices);
  void chat();
  void deinit();

private:
  void answer(const std::string &input_str);
  int forward_first(std::vector<int> &tokens);
  int forward_next(int cur_token);
  void load_sentencepiece(std::string tokenizer_path);
  void build_system_prompt();
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

private:
  bm_handle_t bm_handle = 0;
  void *p_bmrt;
  sentencepiece::SentencePieceProcessor sentencepiece;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<bm_device_mem_t> past_key, past_value;
  std::string system_string = "";
  std::vector<int> history_tokens;
  std::vector<int> head_prompt{1, 1786, 4194, 95388};
  std::vector<int> system_prompt;
  int round = 0;
  int token_length;
  int EOS;
  int SEQLEN;
  int NUM_LAYERS;
  bool io_alone;
};

void MiniCPM::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void MiniCPM::load_sentencepiece(std::string tokenizer_path) {
  printf("Load %s ... ", tokenizer_path.c_str());
  auto status = sentencepiece.Load(tokenizer_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  printf("Done!\n");
}

void MiniCPM::init(std::string model_path, std::string tokenizer_path, const std::vector<int> &devices) {
  load_sentencepiece(tokenizer_path);

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
  bm_status_t status = bm_dev_request(&bm_handle, d);
  assert(BM_SUCCESS == status);
  }

  // decode system prompt
  sentencepiece.Encode(system_string, &system_prompt);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);

  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("\nDone!\n");

  // set NUM_LAYERS
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 3) / 2;

  // net infos
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // set SEQLEN
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];

  // resize
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);

  // net device mem
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  assert(io_alone == true);
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
  }
}

void MiniCPM::deinit() {
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void MiniCPM::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

int MiniCPM::forward_first(std::vector<int> &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);

  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  token_length = tokens.size();
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed); // prefil embedding

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
    bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);

  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (tokens.size() - 1) * bytes, bytes);
  net_launch(net_lm);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  return token;
}

int MiniCPM::forward_next(int cur_token) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // forward embedding
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  d2d(in_mem, lm_out_mem);
  net_launch(net_embed_cache);

  // forward blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    // auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    // auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);

    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    } else {
      d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
      d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
    }

    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  // forward lmhead
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  return token;
}

void MiniCPM::build_system_prompt() {
  history_tokens.clear();
  history_tokens.insert(history_tokens.end(), head_prompt.begin(),
                        head_prompt.end());
  history_tokens.insert(history_tokens.end(), system_prompt.begin(),
                        system_prompt.end());
}

void MiniCPM::chat() {
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str == "exit" || input_str == "quit") {
      break;
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

void MiniCPM::answer(const std::string &input_str) {
  int tok_num = 1;
  std::vector<int> tokens;
  std::vector<int> prompt{95396, 4194, 95388};
  std::vector<int> endwords{95396, 10850, 95388};
  
  sentencepiece.Encode(input_str, &tokens);

  tokens.insert(tokens.end(), endwords.begin(), endwords.end());
  if (history_tokens.size() == 0) {
    build_system_prompt();
  } else {
    tokens.insert(tokens.begin(), prompt.begin(), prompt.end());
  }
  history_tokens.insert(history_tokens.end(), tokens.begin(), tokens.end());
  if (input_str.empty()) {
    printf("Sorry: your question is too wierd!!\n");
    round = 0;
    history_tokens.clear();
    return;
  }
  // make sure token not too large
  if ((int)history_tokens.size() > SEQLEN - 10) {
    // reset
    history_tokens.clear();
    if (round == 0) {
      printf("Error: your question is too large!\n");
      return;
    }
    round = 0;
    answer(input_str);
    return;
  }
  int pre_token = 0;
  auto t0 = std::chrono::system_clock::now();
  int token = forward_first(history_tokens);
  auto t1 = std::chrono::system_clock::now();
  while (token != EOS && token_length < SEQLEN) {
    std::string pre_word;
    std::string word;
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    sentencepiece.Decode(pre_ids, &pre_word);
    sentencepiece.Decode(ids, &word);
    std::string diff = word.substr(pre_word.size());
    history_tokens.emplace_back(token);
    std::cout << diff << std::flush;
    if (token_length < SEQLEN) {
      token_length++;
    }
    tok_num++;
    token = forward_next(token);
  }
  auto t2 = std::chrono::system_clock::now();
  auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  printf("\n\nfirst token latency: %f s", (use0.count() * 1e-6));
  printf("\nspeed: %f token/s\n", tok_num / (use1.count() * 1e-6));
  if (token_length >= SEQLEN) {
    history_tokens.clear();
    round = 0;
  } else {
    round++;
  }
}

static void split(const std::string &s, const std::string &delim,
                  std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (last < s.length()) {
    ret.push_back(s.substr(last));
  }
}

static std::vector<int> parseCascadeDevices(const std::string &str) {
  std::vector<int> devices;
  std::vector<std::string> sub_str;
  split(str, ",", sub_str);
  for (auto &s : sub_str) {
    devices.push_back(std::atoi(s.c_str()));
  }
  return devices;
}

void Usage() {
  printf("Usage:\n"
         "  --help         : Show help info.\n"
         "  --model        : Set model path \n"
         "  --tokenizer    : Set tokenizer path \n"
         "  --devid        : Set devices to run for model, e.g. 1,2, if not provided, use 0\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &tokenizer_path, std::vector<int> &devices) {
  struct option longOptions[] = {
    {"model", required_argument, nullptr, 'm'},
    {"tokenizer", required_argument, nullptr, 't'},
    {"devid", required_argument, nullptr, 'd'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:t:d:h:", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 't':
      tokenizer_path = optarg;
      break;
    case 'd':
      devices = parseCascadeDevices(optarg);
      break;
    case 'h':
      Usage();
      exit(EXIT_FAILURE);
    case '?':
      Usage();
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  // set your bmodel path here
  printf("Demo for MiniCPM in BM1684X AND BM1688\n");
  std::string model_path;
  std::string tokenizer_path;
  std::vector<int> devices = {0};
  processArguments(argc, argv, model_path, tokenizer_path, devices);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  MiniCPM minicpm;
  printf("Init Environment ...\n");
  minicpm.init(model_path, tokenizer_path, devices);
  printf("==========================\n");
  minicpm.chat();
  minicpm.deinit();
  return 0;
}
