//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chat.hpp"
#include "cv_utils.h"
#include "tokenizers-cpp/tokenizers_cpp.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using tokenizers::Tokenizer;

static const int IMAGE_PAD_TOKEN = 101304;
static const int VIDEO_PAD_TOKEN = 101307;

// 从文件加载字节数据
static inline std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open [ " << path << " ]" << std::endl;
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
std::vector<float> convertIntToFloat(const std::vector<int> &position_ids) {
  std::vector<float> floatPositionIds;
  for (int val : position_ids) {
    floatPositionIds.push_back(static_cast<float>(val));
  }
  return floatPositionIds;
}

std::vector<int> convertFloatToInt(const std::vector<float> &position_ids) {
  std::vector<int> intPositionIds;
  for (int val : position_ids) {
    intPositionIds.push_back(static_cast<int>(val));
  }
  return intPositionIds;
}

class ChatPipe {
public:
  Config config;

  ChatPipe(int devid, const std::string &model_path, const std::string &config_path, bool do_sample = false);
  // 聊天主循环
  void chat();

private:
  LFM2_VL model;
  int ID_IM_END, ID_VISION_START, ID_EOS;
  int tokens_per_second;
  int spatial_merge_size;
  int num_grid_per_side;
  int spatial_merge_unit;
  bool support_history;
  // 分词器和处理器
  std::unique_ptr<Tokenizer> tok;

  // 获取媒体类型
  typedef enum { IMAGE, VIDEO, TEXT, UNKNOWN } MediaType;
  MediaType get_media_type(const std::vector<std::string> &file_path);

  // 构建提示
  std::string build_image_prompt(const std::string &input_str);

  // 查找分词偏移量
  std::vector<int> find_token_offset(const std::vector<int> &input_ids,
                                     int pad_id);

  // 获取位置编码
  std::vector<int> get_position_ids(int token_len);

  // 处理图像
  void vit_process_image(std::vector<float> &pixel_values, int vit_offset);

  // 编码输入
  std::vector<int> encode_input(const std::string &sentence_input);

  // 打印聊天说明
  void print_chat_instructions();

  // 推理
  int forward_prefill(std::vector<int> &position_ids_1d, int &max_posid,
                      int &history_max_posid);
};

// 获取媒体类型
ChatPipe::MediaType
ChatPipe::get_media_type(const std::vector<std::string> &medias) {
  if (medias.empty() || medias[0].empty()) {
    return TEXT;
  }
  auto type = UNKNOWN;
  for (auto &m : medias) {
    std::string ext = m.substr(m.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
        ext == "webp") {
      if (type == UNKNOWN) {
        type = IMAGE;
      } else if (type != IMAGE) {
        printf("Error:Mixed media types detected.\n");
        return UNKNOWN;
      }
    } else if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
               ext == "flv" || ext == "wmv") {
      if (type == UNKNOWN) {
        type = VIDEO;
      } else if (type != VIDEO) {
        printf("Error:Mixed media types detected.\n");
        return UNKNOWN;
      }
    }
  }
  return type;
}

std::vector<int> ChatPipe::get_position_ids(int token_len) {
  std::vector<int> position_ids(token_len * 3);
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < token_len; ++i) {
      position_ids[j * token_len + i] = i;
    }
  }
  return position_ids;
}

// ChatPipe 类构造函数
ChatPipe::ChatPipe(int devid, const std::string &model_path,
                   const std::string &config_path, bool do_sample) {
  model.init(devid, model_path, config_path, do_sample);
  spatial_merge_size = 2;
  spatial_merge_unit = spatial_merge_size * spatial_merge_size;
  tokens_per_second = 2;
  num_grid_per_side = 48;
  support_history = model.support_history;

  std::cout << "Processor [" << config_path.c_str() << "] loading .... ";
  auto blob = LoadBytesFromFile((config_path + "/tokenizer.json").c_str());
  tok = Tokenizer::FromBlobJSON(blob);
  ID_IM_END = tok->TokenToId("<|image_end|>");
  ID_VISION_START = tok->TokenToId("<|image_start|>");
  ID_EOS = tok->TokenToId("<|im_end|>");

  std::cout << "Done!" << std::endl;

  config.temporal_patch_size = 1;
  config.spatial_merge_size = 2;
  config.patch_size = 16;
  config.SEQLEN = model.SEQLEN;
  config.MAX_INPUT_LENGTH = model.MAX_INPUT_LENGTH;
  config.MAX_PATCHES = model.MAX_PATCHES;
}

// 线性等分，包含端点
static inline std::vector<float> linspace_inclusive(float start, float end,
                                                    int num) {
  std::vector<float> out;
  out.reserve(num);
  if (num == 1) {
    out.push_back(start);
    return out;
  }
  float step = (end - start) / float(num - 1);
  for (int i = 0; i < num; ++i) {
    out.push_back(start + step * i);
  }
  return out;
}

// 查找元素在向量中的所有索引
std::vector<size_t> argwhere(const std::vector<int> &vec, int value) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] == value) {
      indices.push_back(i);
    }
  }
  return indices;
}

// 生成从 start 到 end-1 的整数序列
std::vector<int> arange(int start, int end) {
  std::vector<int> result;
  for (int i = start; i < end; ++i) {
    result.push_back(i);
  }
  return result;
}

// 获取向量中的最大值
int max(const std::vector<int> &vec) {
  int max_val = vec[0];
  for (int val : vec) {
    if (val > max_val) {
      max_val = val;
    }
  }
  return max_val;
}

// 将二维向量在指定维度上拼接
std::vector<std::vector<int>>
cat(const std::vector<std::vector<std::vector<int>>> &vecs, int dim) {
  if (dim == 1) {
    std::vector<std::vector<int>> result(3);
    for (const auto &vec : vecs) {
      for (int i = 0; i < 3; ++i) {
        result[i].insert(result[i].end(), vec[i].begin(), vec[i].end());
      }
    }
    return result;
  }
  return {};
}

std::string strip(const std::string &s) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  // 找到第一个非空白字符位置
  size_t start = s.find_first_not_of(WHITESPACE);
  if (start == std::string::npos) {
    // 全是空白
    return "";
  }
  // 找到最后一个非空白字符位置
  size_t end = s.find_last_not_of(WHITESPACE);
  // substr(pos, len)，len = end-start+1
  return s.substr(start, end - start + 1);
}

int ChatPipe::forward_prefill(std::vector<int> &position_ids_1d, int &max_posid,
                              int &history_max_posid) {
  if (model.history_length == 0 || support_history == false) {
    history_max_posid = 0;
    return model.forward_first(position_ids_1d);
  }

  if (model.history_length + model.token_length + 128 > model.SEQLEN ||
      model.history_length > model.PREFILL_KV_LENGTH) {
    std::cerr << "Warning: History is full and clear it to continue."
              << std::endl;
    model.clear_history();
    history_max_posid = 0;
    return model.forward_first(position_ids_1d);
  }
  // all id should increase by history_max_posid
  for (auto &x : position_ids_1d) {
    x += history_max_posid;
  }
  max_posid += history_max_posid;
  return model.forward_first(position_ids_1d);
}

static std::vector<std::string> splitString(const std::string &s) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(strip(item));
  }
  return result;
}

// 聊天主循环
void ChatPipe::chat() {
  using clock = std::chrono::steady_clock;
  print_chat_instructions();
  int history_max_posid = 0;
  while (true) {
    std::string input_str;
    int token = 0;
    int max_posid = 0;
    std::cout << "\nQuestion: ";
    std::getline(std::cin, input_str);
    input_str = strip(input_str);
    if (input_str == "exit" || input_str == "q" || input_str == "quit") {
      break;
    }
    if (input_str == "clear" || input_str == "c" || input_str == "new") {
      model.clear_history();
      history_max_posid = 0;
      std::cout << "Chat history cleared." << std::endl;
      continue;
    }
    std::string media_path="../image.png";
    std::cout << "\nImage Path: ";
    std::getline(std::cin, media_path);
    auto medias = splitString(media_path);
    auto media_type = get_media_type(medias);
    if (media_type == ChatPipe::UNKNOWN) {
      std::cout
          << "Unsupported media type. Please provide a valid image"
          << std::endl;
      continue;
    }
    if (media_type != ChatPipe::TEXT) {
      // check file exists
      for (auto &m : medias) {
        if (!std::filesystem::exists(m)) {
          std::cerr << "File does not exist: " << m << std::endl;
          continue;
        }
      }
    }

    std::cout << "\nAnswer:\n";
    int64_t duration_prefill = 0, duration_vit = 0, duration_decode = 0;
    int input_token_num = 0;
    clock::time_point clock_start;
    switch (media_type) {
    case ChatPipe::IMAGE: {
      int num_medias = medias.size();
      std::vector<float> pixel_values[num_medias];
      for (int i = 0; i < num_medias; ++i) {
        auto ret = process_image(pixel_values[i], medias[i], config);
        if (ret == false) {
          std::cerr << "Error processing image: " << medias[i] << std::endl;
          continue;
        }
      }
      std::string sentence_input = build_image_prompt(input_str);
      std::vector<int> tokens = encode_input(sentence_input);
      // // 保存tokens到.bin文件
      // std::ofstream out("tokens.bin", std::ios::binary);
      // out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int));
      // out.close();
      // exit(1);
    
      if ((int)(tokens.size()) > model.MAX_INPUT_LENGTH) {
        std::cerr << "Input tokens exceed maximum length: "
                  << model.MAX_INPUT_LENGTH << std::endl;
        continue;
      }
      input_token_num = tokens.size();
      auto vit_offset = find_token_offset(tokens, ID_VISION_START);
      clock_start = clock::now();
      model.forward_embed(tokens);
      auto clock_vit_start = clock::now();
      for (int i = 0; i < num_medias; ++i) {
        vit_process_image(pixel_values[i], vit_offset[i] + 1);
      }
      auto clock_vit_end = clock::now();
      duration_vit = std::chrono::duration_cast<std::chrono::milliseconds>(
                         clock_vit_end - clock_vit_start)
                         .count();
      std::vector<int> position_ids_1d = get_position_ids(input_token_num);
      token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
    } break;
    default:
      std::cerr << "Unsupported media type." << std::endl;
      continue;
    }
    auto clock_prefill = clock::now();
    duration_prefill = std::chrono::duration_cast<std::chrono::milliseconds>(
                           clock_prefill - clock_start)
                           .count();
    // 后续分词
    std::vector<int> full_word_tokens;
    std::string text;
    int output_token_num = 0;
    while (token != ID_EOS && model.history_length < model.SEQLEN - model.conv_L_cache - 1) {
      // std::cout << "\nfull_word_tokens: " << token << "  " << std::endl;
      full_word_tokens.push_back(token);
      std::string word = tok->Decode(full_word_tokens);
      if (word.find("�") == std::string::npos) {
        if (full_word_tokens.size() == 1) {
          std::string pre_word = word;
          std::vector<int> double_token = {token, token};
          word = tok->Decode(double_token).substr(pre_word.length());
        }
        text += word;
        std::cout << word << std::flush;
        if (model.do_sample) {
          if (model.check_stop(text)) {
            break;
          }
        }
        full_word_tokens.clear();
      }
      max_posid++;
      std::vector<int> following_position_ids = {max_posid, max_posid,
                                                 max_posid};
      token = model.forward_next(following_position_ids);
      output_token_num++;
    }
    history_max_posid = max_posid + 2;
    std::cout << std::endl;
    auto clock_end = clock::now();
    duration_decode = std::chrono::duration_cast<std::chrono::milliseconds>(
                          clock_end - clock_prefill)
                          .count();
    std::cout << "FTL: " << duration_prefill / 1000.0f << " s" << std::endl;
    if (output_token_num > 0) {
      std::cout << "TPS: " << output_token_num * 1000.0f / duration_decode
                << " tokens/s" << std::endl;
    }
    if (duration_vit > 0) {
      std::cout << "Vision [" << config.nchw[0] << ", "
                << config.nchw[1] << ", " << config.nchw[2] << ", "
                << config.nchw[3] << "]: " << duration_vit / 1000.0f << " s" << std::endl;
    }
    std::cout << "Input Tokens: " << input_token_num
              << ", Output Tokens: " << output_token_num + 1 << std::endl;
  }
}

std::string
ChatPipe::build_image_prompt(const std::string &input_str) {
  std::string prompt = "<|startoftext|>";
  int num_images = 1; //目前只支持单图像，后续可以根据num_images构建多图像提示
  for (int i = 0; i < num_images; i++) {
    int pad_len = model.TOKENS_PER_TILE;
    std::string img_coord = "<img_row_1_col_1>";
    prompt += "<|im_start|>user\n<|image_start|>" + img_coord; //目前只支持单图像
    for (int j = 0; j < pad_len; j++) {
      prompt += "<image>";
    }
    prompt += "<|image_end|>";
  }
  prompt += input_str + "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

// 查找分词偏移量
std::vector<int> ChatPipe::find_token_offset(const std::vector<int> &input_ids,
                                             int pad_id) {
  std::vector<int> offsets;
  int num = input_ids.size();
  for (int i = 0; i < num; ++i) {
    if (input_ids[i] == pad_id) {
      offsets.push_back(i);
    }
  }
  return offsets;
}

// 处理图像
void ChatPipe::vit_process_image(std::vector<float> &pixel_values,
                                 int vit_offset) {
  model.forward_vit(pixel_values.data(), vit_offset);
}

// 编码输入
std::vector<int> ChatPipe::encode_input(const std::string &sentence_input) {
  return tok->Encode(sentence_input);
}

void ChatPipe::print_chat_instructions() {
  std::cout
      << "\n================================================================="
         "\n"
      << "1. If you want to quit, please enter one of [q, quit, exit]\n"
      << "2. To create a new chat session, please enter one of [clear, new]\n"
      << "================================================================="
         "\n";
}

void Usage() {
  printf(
      "Usage:\n"
      "  -h, --help        : Show help info \n"
      "  -m, --model       : Set model path \n"
      "  -c, --config      : Set config path \n"
      "  -s, --do_sample   : Enable sampling during generation\n"
      "  -d, --devid       : Set devices to run for model, default is '0'\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &config_path, std::string &image_path,
                      int &device, bool &do_sample) {
  struct option longOptions[] = {
      {"model", required_argument, nullptr, 'm'},
      {"config", required_argument, nullptr, 'c'},
      {"devid", required_argument, nullptr, 'd'},
      {"do_sample", no_argument, nullptr, 's'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;
  while ((option = getopt_long(argc, argv, "m:c:d:r:f:sh", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'd':
      device = atoi(optarg);
      break;
    case 's':
      do_sample = true;
      break;
    case 'h':
      Usage();
      exit(EXIT_SUCCESS);
    case '?':
      Usage();
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char *argv[]) {
  std::string model_path;
  std::string config_path;
  std::string image_path;
  int dev_id = 0;
  bool do_sample = false;

  processArguments(argc, argv, model_path, config_path, image_path, dev_id, do_sample);
  if (model_path.empty() || config_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }
  ChatPipe pipeline(dev_id, model_path, config_path, do_sample);
  pipeline.chat();
  return 0;
}