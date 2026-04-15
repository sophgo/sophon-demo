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
//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
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
// PaddleOCR_VL
//===------------------------------------------------------------===//
void PaddleOCR_VL::init_tensors(const bm_net_info_t *net,
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

bool PaddleOCR_VL::check_stop(const std::string &text) {
  for (const auto &stop_str : stop_strings) {
    if (ends_with(text, stop_str)) {
      return true;
    }
  }
  return false;
}

void PaddleOCR_VL::net_launch(const bm_net_info_t *net,
                          const std::vector<bm_tensor_t> &in_tensors,
                          std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void PaddleOCR_VL::net_launch_decode(int idx, int kv_offset,
                                 bm_device_mem_t &input_mem, const int *pos_id,
                                 std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors);

  // ===== prepare input tensors =====
  in_tensors[0].device_mem = input_mem;
  if (idx == 0) {
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
  } else {
    in_tensors[1].device_mem = net_blocks_cache[0]->stages[0].input_mems[1];
    in_tensors[2].device_mem = net_blocks_cache[0]->stages[0].input_mems[2];
  }
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);

  // ===== launch =====
  net_launch(net, in_tensors, out_tensors);
}

void PaddleOCR_VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                   int size) {
  if (!size) {
    size = bm_mem_get_device_size(src);
  }
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void PaddleOCR_VL::clear_history() {
  if (!support_history) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void PaddleOCR_VL::init_by_names() {
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
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
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

void PaddleOCR_VL::init(int dev_id, std::string model_path, std::string config_path,
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
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
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

void PaddleOCR_VL::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int PaddleOCR_VL::greedy_search(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_greedy_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;
  net_launch(net_greedy_head, in_tensors, out_tensors);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  return token;
}

void PaddleOCR_VL::forward_embed(std::vector<int> const &tokens) {
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

void PaddleOCR_VL::forward_vit(const float *pixel_values, std::vector<int> grid_thw, int vit_offset) {
  int t = grid_thw[0];
  int h = grid_thw[1];
  int w = grid_thw[2];
  int hw = t * h * w;
  int num_pixels = 1;
  for (int i = 0; i < net_vit->stages[0].input_shapes[0].num_dims; i++) {
    num_pixels *= net_vit->stages[0].input_shapes[0].dims[i];
  }
  // select stage
  int stage = 0;
  for (stage = 0; stage < net_vit->stage_num; stage++) {
    if (hw > VIT_PATCH_LIST[stage]) {
      break;
    }
  }
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
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);

  // uint16_t fin[vit_size / 2];
  // std::fstream fin_file("../vit_out_3.bin", std::ios::in | std::ios::binary);
  // fin_file.read((char *)fin, vit_size);
  // fin_file.close();
  // bm_memcpy_s2d_partial(bm_handle, out_tensors[0].device_mem, (void *)fin,
  //                       vit_size);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
  
  // uint16_t temp[vit_size / 2];
  // bm_memcpy_d2s_partial(bm_handle, (void *)temp, out_tensors[0].device_mem, vit_size);
  vit_run = true;
}

int PaddleOCR_VL::generate(bm_device_mem_t &logits_mem) {
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

int PaddleOCR_VL::penalty_sample(bm_device_mem_t &logits_mem) {
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

int PaddleOCR_VL::forward_first(std::vector<int> const &position_ids) {
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
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
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

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0], stage);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors, stage);
    in_tensors[0].device_mem = out_mem;
    if (is_dynamic) {
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                              (void *)position_ids_pad.data(),
                              token_length * 3 * sizeof(int));
        bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                              (void *)attention_mask.data(),
                              token_length * token_length * sizeof(uint16_t));
      }
      in_tensors[0].shape.dims[1] = token_length;
      in_tensors[1].shape.dims[1] = token_length;
      in_tensors[2].shape.dims[2] = token_length;
      in_tensors[2].shape.dims[3] = token_length;
    } else {
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                      (void *)position_ids_pad.data());
        bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                      (void *)attention_mask.data());
      }
    }
    net_launch(net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[stage].output_mems[0];
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                       net_blocks[idx]->stages[stage].output_mems[1], 0,
                       KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                       net_blocks[idx]->stages[stage].output_mems[2], 0,
                       KV_BYTES * token_length);
  }
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

int PaddleOCR_VL::forward_first_with_kv(std::vector<int> const &position_ids) {
  int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length,
                                       mask_value);
  auto old_length = history_length;
  history_length += token_length;
  assert(history_length < SEQLEN);
  assert(old_length <= PREFILL_KV_LENGTH);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < old_length; j++) {
      attention_mask[i * max_kv_length + j] = 0;
    }
    for (int j = 0; j <= i; j++) {
      attention_mask[i * max_kv_length + j + PREFILL_KV_LENGTH] = 0;
    }
  }

  const int *p_ids = position_ids.data();

  std::vector<int> position_ids_pad(3 * MAX_INPUT_LENGTH, 0);
  int ori_length = position_ids.size() / 3;
  assert(ori_length == token_length);
  assert(ori_length <= MAX_INPUT_LENGTH);
  for (int i = 0; i < 3; i++) {
    int ori_offset = i * ori_length;
    int dst_offset = i * MAX_INPUT_LENGTH;
    std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
              position_ids_pad.begin() + dst_offset);
  }

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    in_tensors[0].device_mem = out_mem;
    if (old_length > 0) {
      d2d(in_tensors[3].device_mem, past_key[idx], 0, KV_BYTES * old_length);
      d2d(in_tensors[4].device_mem, past_value[idx], 0, KV_BYTES * old_length);
    } else if (idx == 0) {
      empty(bm_handle, in_tensors[3].device_mem);
      empty(bm_handle, in_tensors[4].device_mem);
    }
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                  (void *)position_ids_pad.data());
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
    net_launch(net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks[idx]->stages[0].output_mems[2];
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], old_length * KV_BYTES,
                       out1_mem, 0, KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], old_length * KV_BYTES,
                       out2_mem, 0, KV_BYTES * token_length);
  }

  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);
  int token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

int PaddleOCR_VL::forward_next(std::vector<int> const &position_ids) {
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

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    net_launch_decode(idx, token_offset, out_mem, p_ids, attention_mask);
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
