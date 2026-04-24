//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chat.hpp"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  if (ret != BM_SUCCESS) {
        throw std::runtime_error("BMRuntime 操作失败");
    }
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage].output_mems[i]);
  }
}

//===------------------------------------------------------------===//
// Generation Config
//===------------------------------------------------------------===//
struct GenerationConfig {
  std::vector<int> eos_token_id;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  int top_k = 50;
  float top_p = 1.0;
  std::vector<std::string> stop_strings;
  static GenerationConfig from_json(const std::string &path) {
    GenerationConfig config;
    std::ifstream in(path);
    nlohmann::json j;
    in >> j;
    if (j.contains("eos_token_id"))
      config.eos_token_id = j["eos_token_id"].get<std::vector<int>>();
    if (j.contains("repetition_penalty"))
      config.repetition_penalty = j["repetition_penalty"].get<float>();
    if (j.contains("temperature"))
      config.temperature = j["temperature"].get<float>();
    if (j.contains("top_k"))
      config.top_k = j["top_k"].get<int>();
    if (j.contains("top_p"))
      config.top_p = j["top_p"].get<float>();
    if (j.contains("stop_strings"))
      config.stop_strings = j["stop_strings"].get<std::vector<std::string>>();
    return config;
  }
};

//===------------------------------------------------------------===//
// LFM2_VL
//===------------------------------------------------------------===//
void LFM2_VL::init_tensors(const bm_net_info_t *net,
                            std::vector<bm_tensor_t> &in_tensors,
                            std::vector<bm_tensor_t> &out_tensors, int stage) {
  in_tensors.resize(net->input_num);
  out_tensors.resize(net->output_num);
  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[stage].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[stage].input_shapes[i]);
  }

  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[stage].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[stage].output_shapes[i]);
  }
}

static bool ends_with(const std::string &str, const std::string &suffix) {
  if (str.size() < suffix.size())
    return false;
  return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

bool LFM2_VL::check_stop(const std::string &text) {
  for (const auto &stop_str : stop_strings) {
    if (ends_with(text, stop_str)) {
      return true;
    }
  }
  return false;
}

void LFM2_VL::net_launch(const bm_net_info_t *net,
                          const std::vector<bm_tensor_t> &in_tensors,
                          std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void LFM2_VL::net_launch_full_attention(int idx, int idx_kv, int kv_offset,
                                 bm_device_mem_t &input_mem, const int *pos_id,
                                 std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors);

  // ===== prepare input tensors =====
  in_tensors[0].device_mem = input_mem;
  static int idx_kv_start = 0;
  if (idx_kv == 0) {
    idx_kv_start = idx;
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
  } else {
    in_tensors[1].device_mem = net_blocks_cache[idx_kv_start]->stages[0].input_mems[1];
    in_tensors[2].device_mem = net_blocks_cache[idx_kv_start]->stages[0].input_mems[2];
  }
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx_kv].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx_kv].u.device.device_addr + kv_offset, KV_BYTES);

  // ===== launch =====
  net_launch(net, in_tensors, out_tensors);
}

void LFM2_VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                   int size) {
  if (!size) {
    size = bm_mem_get_device_size(src);
  }
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void LFM2_VL::clear_history() {
  if (!support_history) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
    // Fix: also clear conv state cache
    if (!past_conv_state.empty()) {
      empty(bm_handle, past_conv_state[i]);
    }
  }
  history_length = 0;
  token_length = 0;
}

void LFM2_VL::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_greedy_head = nullptr;
  auto num_blocks =
      num_nets - 4; // 4 nets are embed, lm_head, embedding_cache, vit
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
    num_blocks--; // greedy_head is not a block
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
    num_blocks--; // sample_head is not a block
  }
  // 2 nets for each block, one for cache
  NUM_LAYERS = num_blocks / 2;

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, expected %d blocks.\n", NUM_LAYERS,
             num_blocks / 2);
      break;
    }
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }
  free(net_names);
  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  support_history = net_blocks[0]->input_num == 5; // with kv cache
  is_dynamic = net_blocks[0]->is_dynamic;
  vit_dynamic = net_vit->is_dynamic;
  history_length = 0;
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks[0]->stages[0].input_shapes[0].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  TOKENS_PER_TILE = net_vit->stages[0].output_shapes[0].dims[1];
  for(int i = 0; i < NUM_LAYERS; i++) {
    if(KV_BYTES != -1 && CONV_STATE_BYTES != -1) {
      break;
    }
    if (layer_types[i] == "full_attention") {
      KV_BYTES = bm_mem_get_device_size(net_blocks_cache[i]->stages[0].output_mems[1]);
    } else{
      CONV_STATE_BYTES = bm_mem_get_device_size(net_blocks_cache[i]->stages[0].output_mems[1]);
    }
  }
  
  for (int i = 0; i < net_vit->stage_num; i++) {
    VIT_PATCH_LIST.push_back(net_vit->stages[i].input_shapes[0].dims[0]);
  }
  if (net_blocks[0]->stage_num > 1) {
    for (int i = 0; i < net_blocks[0]->stage_num; i++) {
      INPUT_LENGTH_LIST.push_back(
          net_blocks[0]->stages[i].input_shapes[0].dims[1]);
    }
  } else {
    INPUT_LENGTH_LIST.push_back(MAX_INPUT_LENGTH);
  }
  printf("Num Layers:%d\n", NUM_LAYERS);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True\n");
  } else {
    printf("History Support: False\n");
  }
}

void LFM2_VL::init(int dev_id, std::string model_path, std::string config_path,
                    bool do_sample_) {

  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  init_by_names();

  visited_tokens.resize(SEQLEN);

  // kv cache
  for (int i = 0; i < NUM_LAYERS; i++) {
    if (layer_types[i] == "full_attention") {
      past_key.push_back(net_blocks_cache[i]->stages[0].input_mems[3]);
      past_value.push_back(net_blocks_cache[i]->stages[0].input_mems[4]);
      empty(bm_handle, past_key.back());
      empty(bm_handle, past_value.back());
    } else { // conv layer, use conv state as kv cache
      past_conv_state.push_back(net_blocks_cache[i]->stages[0].input_mems[1]);
      empty(bm_handle, past_conv_state.back());
    }
  }
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
  vit_run = false;
  do_sample = do_sample_;
  if (do_sample) {
    if (!net_sample_head) {
      std::cerr
          << "\nWarning: Sample head not found in the model. You need compile "
             "bmodel with --do_sample. Using greedy mode instead!\n";
    } else {
      std::string generation_path = config_path + "/generation_config.json";
      std::cout << "Generation Config [" << generation_path.c_str()
                << "] loading .... ";
      auto gen_config = GenerationConfig::from_json(generation_path);
      penalty = gen_config.repetition_penalty;
      temperature = gen_config.temperature;
      top_k = gen_config.top_k;
      top_p = gen_config.top_p;
      if (!gen_config.stop_strings.empty()) {
        stop_strings = gen_config.stop_strings;
      }
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[2],
                    (void *)&penalty);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[3],
                    (void *)&temperature);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[4],
                    (void *)&top_k);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[5],
                    (void *)&top_p);
      std::cout << "Done!" << std::endl;
    }
  }
}

void LFM2_VL::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int LFM2_VL::greedy_search(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_greedy_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;
  net_launch(net_greedy_head, in_tensors, out_tensors);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  return token;
}

void LFM2_VL::forward_embed(std::vector<int> const &tokens) {
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)visited_tokens.data(),
                        MAX_INPUT_LENGTH * sizeof(int));
  net_launch(net_embed, in_tensors, out_tensors);
  empty(bm_handle, dev_buffer);
  d2d(dev_buffer, out_tensors[0].device_mem, 0,
      tokens.size() * HIDDEN_SIZE * sizeof(uint16_t));
  token_length = tokens.size();
}

void LFM2_VL::forward_vit(const float *pixel_values, int vit_offset) {
  int num_pixels = 1;
  for (int i = 0; i < net_vit->stages[0].input_shapes[0].num_dims; i++) {
    num_pixels *= net_vit->stages[0].input_shapes[0].dims[i];
  }
  // select stage
  int stage = 0;
  stage = std::max(0, stage - 1);
  empty_net(bm_handle, net_vit, stage);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit, in_tensors, out_tensors, stage);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)pixel_values, num_pixels * sizeof(float));
  // float fin_pix[num_pixels];
  // std::fstream fin_pix_file("../pixel_values.bin", std::ios::in | std::ios::binary);
  // fin_pix_file.read((char *)fin_pix, num_pixels * sizeof(float));
  // fin_pix_file.close();
  // bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem, (void *)fin_pix,
  //                       num_pixels * sizeof(float));


  net_launch(net_vit, in_tensors, out_tensors);
  int vit_size = net_vit->stages[stage].output_shapes[0].dims[1] * HIDDEN_SIZE * sizeof(uint16_t);
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);

  // uint16_t fin[1783*2048];
  // std::fstream fin_file("/home/lihengfang/work/open-source/LFM2-VL-1.6B/lfm2_image_features.bin", 
  //                       std::ios::in | std::ios::binary);
  // fin_file.read((char *)fin, 1783*2048 * sizeof(uint16_t));
  // fin_file.close();
  // bm_memcpy_s2d_partial_offset(bm_handle, dev_buffer, (void *)fin,
  //                       1783*2048 * sizeof(uint16_t), dst_offset);

  // concatenante texting embedding and image embedding
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
  
  vit_run = true;
}

int LFM2_VL::generate(bm_device_mem_t &logits_mem) {
  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s_partial(bm_handle, (void *)&token, logits_mem, sizeof(int));
  } else if (do_sample) {
    token = penalty_sample(logits_mem);
  } else {
    token = greedy_search(logits_mem);
  }
  return token;
}

int LFM2_VL::penalty_sample(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_sample_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;

  // repeat_penalty + top_p + top_k + temperature
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)visited_tokens.data(),
                        token_length * sizeof(int));
  in_tensors[1].shape.dims[1] = token_length;

  // inference
  net_launch(net_sample_head, in_tensors, out_tensors);

  // get logit & token
  int candidate_num = top_k;
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, probs.data(),
                               out_tensors[0].device_mem, top_k * sizeof(float),
                               0);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, tokens.data(),
                               out_tensors[1].device_mem, top_k * sizeof(float),
                               0);

  // sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int LFM2_VL::forward_first(std::vector<int> const &position_ids) {
  // token_length = 1808; //debug
  // Fix: auto clear history if history_length > 0 to ensure clean state
  if (history_length > 0) {
    clear_history();
  }
  if (support_history) {
    return forward_first_with_kv(position_ids);
  }
  const int *p_ids = position_ids.data();
  std::vector<int> position_ids_pad;
  std::vector<uint16_t> attention_mask;
  int stage = 0;
  if (is_dynamic) {
    attention_mask.assign(token_length * token_length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
    position_ids_pad.assign(3 * token_length, 0);
    assert((int)position_ids.size() == token_length * 3);
    std::copy(p_ids, p_ids + token_length * 3, position_ids_pad.begin());
  } else {
    for (stage = 0; stage < net_blocks[0]->stage_num; stage++) {
      if (token_length > INPUT_LENGTH_LIST[stage]) {
        break;
      }
    }
    stage = std::max(0, stage - 1);
    int length = INPUT_LENGTH_LIST[stage];
    attention_mask.assign(length * length, mask_value);
    for (int i = 1; i < token_length; i++) {
      for (int j = 0; j < i; j++) {
        attention_mask[i * length + j] = 0;
      }
    }
    position_ids_pad.assign(3 * length, 0);
    int ori_length = position_ids.size() / 3;
    for (int i = 0; i < 3; i++) {
      int ori_offset = i * ori_length;
      int dst_offset = i * length;
      std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
                position_ids_pad.begin() + dst_offset);
    }
  }
  //debug start
  // uint16_t* fin = new uint16_t[2048 * 2048];
  // std::fstream fin_file("/home/lihengfang/work/open-source/LFM2-VL-1.6B/binarys/hidden_states_input.bin", 
  //                       std::ios::in | std::ios::binary);
  // fin_file.read((char *)fin, 2048*2048 * sizeof(uint16_t));
  // fin_file.close();
  // bm_memcpy_s2d_partial_offset(bm_handle, dev_buffer, (void *)fin,
  //                       2048*2048 * sizeof(uint16_t), 0);
  // delete [] fin;
  //debug end
  
  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0], stage);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  int idx_kv = 0;
  int idx_conv = 0;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors, stage);
    in_tensors[0].device_mem = out_mem;
    if (is_dynamic) {
      // unsupport dynamic now
      // if (idx == 0) {
      //   // only first time need copy
      //   bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
      //                         (void *)position_ids_pad.data(),
      //                         token_length * 3 * sizeof(int));
      //   bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
      //                         (void *)attention_mask.data(),
      //                         token_length * token_length * sizeof(uint16_t));
      // }
      // in_tensors[0].shape.dims[1] = token_length;
      // in_tensors[1].shape.dims[1] = token_length;
      // in_tensors[2].shape.dims[2] = token_length;
      // in_tensors[2].shape.dims[3] = token_length;
    } else {
        if (layer_types[idx] == "full_attention") {
          bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                        (void *)position_ids_pad.data());
          bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                        (void *)attention_mask.data());
        } else{
          int32_t conv_state_offsetsT[1] = {token_length - conv_L_cache};
          int32_t conv_state_endsT[1] = {token_length};
          bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                        (void *)conv_state_offsetsT);
          bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                        (void *)conv_state_endsT);
          out_tensors[1].shape.dims[2] = conv_L_cache;
        }
    }
    //debug start
    // uint16_t* hidden_states_in = new uint16_t[2048 * 2048];
    // bm_memcpy_d2s_partial(bm_handle, hidden_states_in, in_tensors[0].device_mem,
    //                       2048 * 2048 * sizeof(uint16_t));
    //debug end

    net_launch(net_blocks[idx], in_tensors, out_tensors);
    //debug start
    // if (layer_types[idx] == "full_attention"){
    //   uint16_t* hidden_states_out = new uint16_t[2048 * 2048];
    //   bm_memcpy_d2s_partial(bm_handle, hidden_states_out, out_tensors[0].device_mem,
    //                         2048 * 2048 * sizeof(uint16_t));
    //   uint16_t* key_cache_out = new uint16_t[8*64];
    //   bm_memcpy_d2s_partial(bm_handle, key_cache_out, out_tensors[1].device_mem,
    //                         8 * 64 * sizeof(uint16_t));
    //   // // 保存hidden_states_out到文件
    //   // std::string fname = "/home/lihengfang/work/open-source/LFM2-VL-1.6B/binarys/hidden_states_out_layer" + std::to_string(idx) + ".bin";
    //   // std::ofstream fout(fname, std::ios::binary);
    //   // fout.write((char*)hidden_states_out, 2048*2048*sizeof(uint16_t));
    //   // fout.close();
    //   // // 保存key_cache_out到文件
    //   // std::string kfname = "/home/lihengfang/work/open-source/LFM2-VL-1.6B/binarys/key_cache_out_layer" + std::to_string(idx) + ".bin";
    //   // std::ofstream kfout(kfname, std::ios::binary);
    //   // kfout.write((char*)key_cache_out, 8*64*sizeof(uint16_t));
    //   // kfout.close();
    //   // printf("Saved hidden_states_out and key_cache_out for layer %d\n", idx);
    //   delete [] hidden_states_out;
    //   delete [] key_cache_out;
    // }
    // delete [] hidden_states_in;
    //debug end
  
    out_mem = out_tensors[0].device_mem; //net_blocks[idx]->stages[stage].output_mems[0];
    if (layer_types[idx] == "full_attention") {
      bm_memcpy_d2d_byte(bm_handle, past_key[idx_kv], 0,
                        net_blocks[idx]->stages[stage].output_mems[1], 0,
                        KV_BYTES * token_length);
      bm_memcpy_d2d_byte(bm_handle, past_value[idx_kv], 0,
                        net_blocks[idx]->stages[stage].output_mems[2], 0,
                        KV_BYTES * token_length);
      idx_kv++;
    } else { // conv layer, use conv state as kv cache
      bm_memcpy_d2d_byte(bm_handle, past_conv_state[idx_conv], 0,
                        net_blocks[idx]->stages[stage].output_mems[1], 0,
                        CONV_STATE_BYTES);
      idx_conv++;
    }
  }
    // debug start
    // uint16_t* hidden_states_in = new uint16_t[2048 * 2048];
    // bm_memcpy_d2s_partial(bm_handle, hidden_states_in, out_tensors[0].device_mem,
    //                       2048 * 2048 * sizeof(uint16_t));
    // debug end

  //debug start
  // uint16_t* fin_out = new uint16_t[2048 * 2048];
  // std::fstream fin_file_out("/home/lihengfang/work/open-source/LFM2-VL-1.6B/binarys/hidden_states_forward_first_out.bin", 
  //                       std::ios::in | std::ios::binary);
  // fin_file_out.read((char *)fin_out, 2048*2048 * sizeof(uint16_t));
  // fin_file_out.close();
  // bm_memcpy_s2d_partial_offset(bm_handle, dev_buffer, (void *)fin_out,
  //                       2048*2048 * sizeof(uint16_t), 0);
  // delete [] fin_out;
  //debug end
  
  vit_run = false;
  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);
  auto token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length = token_length;
  return token;
}

int LFM2_VL::forward_first_with_kv(std::vector<int> const &position_ids) {
  //unsupport history now.
  return -1;
}

int LFM2_VL::forward_next(std::vector<int> const &position_ids) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 3);
  const int *p_ids = position_ids.data();
  // embedding
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  int token = visited_tokens[token_length - 1];
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_tensors, out_tensors);
  auto out_mem = out_tensors[0].device_mem;
  int idx_conv = 0;
  int idx_kv = 0;
  // blocks
  int token_offset = (history_length - 1) * KV_BYTES;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    if(layer_types[idx] == "full_attention") {
      net_launch_full_attention(idx, idx_kv, token_offset, out_mem, p_ids, attention_mask);
      idx_kv++;
    } else { // conv layer
      auto &net = net_blocks_cache[idx];
      std::vector<bm_tensor_t> in_tensors;
      std::vector<bm_tensor_t> out_tensors;
      init_tensors(net, in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      in_tensors[1].device_mem = past_conv_state[idx_conv];
      net_launch(net, in_tensors, out_tensors);
      bm_memcpy_d2d_byte(bm_handle, past_conv_state[idx_conv], 0,
                        out_tensors[1].device_mem, 0,
                        CONV_STATE_BYTES);
      idx_conv++;
    }
    out_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
  }

  // forward lmhead
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = out_mem;
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);

  token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}