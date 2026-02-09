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
  PaddleOCR_VL model;
  int ID_IM_END, ID_VISION_START, ID_EOS;
  int tokens_per_second;
  int spatial_merge_size;
  int num_grid_per_side;
  int spatial_merge_unit;
  bool support_history;
  // 分词器和处理器
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Maker> maker;

  // 计算旋转位置编码
  std::vector<std::vector<int>>
  rot_pos(const std::vector<std::vector<int>> &grid_thw);

  // 获取媒体类型
  typedef enum { IMAGE, VIDEO, TEXT, UNKNOWN } MediaType;
  MediaType get_media_type(const std::vector<std::string> &file_path);

  // 构建提示
  std::string build_text_prompt(const std::string &input_str);
  std::string build_image_prompt(const std::string &input_str,
                                 const std::vector<std::vector<int>> &grid_thw);

  // 获取rope索引
  std::vector<std::vector<int>>
  get_rope_index(const std::vector<int> &input_ids,
                 const std::vector<std::vector<int>> &grid_thw, int pad_id);

  void fast_pos_embed_interpolate(const std::vector<int> &grid_thw,
                                  std::vector<int> &idx_out,
                                  std::vector<float> &weight_out);

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
  ID_IM_END = tok->TokenToId("<|IMAGE_END|>");
  ID_VISION_START = tok->TokenToId("<|IMAGE_START|>");
  ID_EOS = tok->TokenToId("</s>");

  std::cout << "Done!" << std::endl;

  config.temporal_patch_size = 1;
  config.spatial_merge_size = 2;
  config.patch_size = 14;
  config.SEQLEN = model.SEQLEN;
  config.MAX_INPUT_LENGTH = model.MAX_INPUT_LENGTH;
  config.MAX_PATCHES = model.MAX_PATCHES;
  config.MIN_PIXELS = 28 * 28 * 130;
  config.MAX_PIXELS = 784 * 784;
  maker = std::make_unique<Maker>(config);
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

// 主函数：返回两个向量
// idx_out: int32 等价（用 int 存储），长度 4 * t * h * w
// weight_out: float32，长度 4 * t * h * w
void ChatPipe::fast_pos_embed_interpolate(const std::vector<int> &grid_thw,
                                          std::vector<int> &idx_out,
                                          std::vector<float> &weight_out) {
  if (grid_thw.empty()) {
    throw std::invalid_argument("grid_thw must contain at least one element");
  }
  int t = 1;
  int h = grid_thw[1];
  int w = grid_thw[2];

  if (h <= 0 || w <= 0 || t <= 0) {
    throw std::invalid_argument("t, h, w must be positive");
  }

  auto h_idxs = linspace_inclusive(0.0f, float(num_grid_per_side - 1), h);
  auto w_idxs = linspace_inclusive(0.0f, float(num_grid_per_side - 1), w);

  std::vector<int> h_floor(h), h_ceil(h);
  std::vector<int> w_floor(w), w_ceil(w);
  std::vector<float> dh(h), dw(w);

  for (int i = 0; i < h; ++i) {
    int f = static_cast<int>(h_idxs[i]);
    int c = std::min(f + 1, num_grid_per_side - 1);
    h_floor[i] = f;
    h_ceil[i] = c;
    dh[i] = h_idxs[i] - float(f);
  }
  for (int j = 0; j < w; ++j) {
    int f = static_cast<int>(w_idxs[j]);
    int c = std::min(f + 1, num_grid_per_side - 1);
    w_floor[j] = f;
    w_ceil[j] = c;
    dw[j] = w_idxs[j] - float(f);
  }

  std::vector<int> base_h(h), base_h_ceil(h);
  for (int i = 0; i < h; ++i) {
    base_h[i] = h_floor[i] * num_grid_per_side;
    base_h_ceil[i] = h_ceil[i] * num_grid_per_side;
  }

  std::vector<int> idx00;
  idx00.reserve(h * w);
  std::vector<int> idx01;
  idx01.reserve(h * w);
  std::vector<int> idx10;
  idx10.reserve(h * w);
  std::vector<int> idx11;
  idx11.reserve(h * w);

  std::vector<float> w00;
  w00.reserve(h * w);
  std::vector<float> w01;
  w01.reserve(h * w);
  std::vector<float> w10;
  w10.reserve(h * w);
  std::vector<float> w11;
  w11.reserve(h * w);

  for (int i = 0; i < h; ++i) {
    float dh_i = dh[i];
    float one_dh_i = 1.0f - dh_i;
    int base_i = base_h[i];
    int base_i_ceil = base_h_ceil[i];
    for (int j = 0; j < w; ++j) {
      float dw_j = dw[j];
      float one_dw_j = 1.0f - dw_j;

      idx00.push_back(base_i + w_floor[j]);
      idx01.push_back(base_i + w_ceil[j]);
      idx10.push_back(base_i_ceil + w_floor[j]);
      idx11.push_back(base_i_ceil + w_ceil[j]);

      w00.push_back(one_dh_i * one_dw_j);
      w01.push_back(one_dh_i * dw_j);
      w10.push_back(dh_i * one_dw_j);
      w11.push_back(dh_i * dw_j);
    }
  }

  int msize = spatial_merge_size;
  int H_blk = h / msize;
  int W_blk = w / msize;

  // 调整输出大小为 [t*h*w, 4]
  idx_out.resize(t * h * w * 4);
  weight_out.resize(t * h * w * 4);

  // 构造重排顺序 (块重排)
  std::vector<int> out_order;
  out_order.reserve(h * w);
  for (int i_blk = 0; i_blk < H_blk; ++i_blk) {
    for (int j_blk = 0; j_blk < W_blk; ++j_blk) {
      for (int i2 = 0; i2 < msize; ++i2) {
        for (int j2 = 0; j2 < msize; ++j2) {
          int i = i_blk * msize + i2;
          int j = j_blk * msize + j2;
          int flat = i * w + j;
          out_order.push_back(flat);
        }
      }
    }
  }

  // 写入为列优先（最后一维为4通道）
  // 位置k的四通道分别是列0..3，对应 idx00/01/10/11
  for (int k = 0; k < (int)out_order.size(); ++k) {
    int src = out_order[k];
    int base = k * 4;

    idx_out[base + 0] = idx00[src];
    idx_out[base + 1] = idx01[src];
    idx_out[base + 2] = idx10[src];
    idx_out[base + 3] = idx11[src];

    weight_out[base + 0] = w00[src];
    weight_out[base + 1] = w01[src];
    weight_out[base + 2] = w10[src];
    weight_out[base + 3] = w11[src];
  }
}

// 计算旋转位置编码
std::vector<std::vector<int>>
ChatPipe::rot_pos(const std::vector<std::vector<int>> &grid_thw) {
  std::vector<std::vector<int>> pos_ids;

  for (const auto &thw : grid_thw) {
    int t = thw[0];
    int h = thw[1];
    int w = thw[2];

    // 生成 hpos_ids
    std::vector<int> hpos_ids;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        hpos_ids.push_back(i);
      }
    }

    // 重塑 hpos_ids
    std::vector<int> reshaped_hpos_ids;
    int h_merged = h / spatial_merge_size;
    int w_merged = w / spatial_merge_size;
    for (int i = 0; i < h_merged; ++i) {
      for (int j = 0; j < w_merged; ++j) {
        for (int k = 0; k < spatial_merge_size; ++k) {
          for (int l = 0; l < spatial_merge_size; ++l) {
            int src_idx = ((i * spatial_merge_size + k) * w) +
                          (j * spatial_merge_size + l);
            reshaped_hpos_ids.push_back(hpos_ids[src_idx]);
          }
        }
      }
    }

    // 生成 wpos_ids
    std::vector<int> wpos_ids;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        wpos_ids.push_back(j);
      }
    }

    // 重塑 wpos_ids
    std::vector<int> reshaped_wpos_ids;
    for (int i = 0; i < h_merged; ++i) {
      for (int j = 0; j < w_merged; ++j) {
        for (int k = 0; k < spatial_merge_size; ++k) {
          for (int l = 0; l < spatial_merge_size; ++l) {
            int src_idx = ((i * spatial_merge_size + k) * w) +
                          (j * spatial_merge_size + l);
            reshaped_wpos_ids.push_back(wpos_ids[src_idx]);
          }
        }
      }
    }

    // 合并 hpos_ids 和 wpos_ids
    std::vector<std::vector<int>> merged;
    for (size_t i = 0; i < reshaped_hpos_ids.size(); ++i) {
      merged.push_back({reshaped_hpos_ids[i], reshaped_wpos_ids[i]});
    }

    // 重复 t 次
    std::vector<std::vector<int>> repeated;
    for (int i = 0; i < t; ++i) {
      repeated.insert(repeated.end(), merged.begin(), merged.end());
    }

    pos_ids.insert(pos_ids.end(), repeated.begin(), repeated.end());
  }

  return pos_ids;
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
// 返回形状为 [3][seq_len] 的position_ids
std::vector<std::vector<int>>
ChatPipe::get_rope_index(const std::vector<int> &input_ids,
                         const std::vector<std::vector<int>> &grid_thw,
                         int pad_id) {

  size_t seq_length = input_ids.size();

  // 初始化 attention_mask 和 position_ids
  std::vector<std::vector<int>> position_ids(3,
                                             std::vector<int>(seq_length, 1));

  // 计算图像数量
  std::vector<size_t> vision_start_indices =
      argwhere(input_ids, ID_VISION_START);
  int image_nums = vision_start_indices.size();

  std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;
  size_t st = 0;
  int remain_images = image_nums;
  int second_per_grid_t = pad_id == VIDEO_PAD_TOKEN ? 1 : 0;
  for (int img_idx = 0; img_idx < image_nums; ++img_idx) {
    size_t ed_image = input_ids.size();
    if (remain_images > 0) {
      auto it = std::find(input_ids.begin() + st, input_ids.end(), pad_id);
      if (it != input_ids.end()) {
        ed_image = it - input_ids.begin();
      }
    }
    int t, h, w;
    if (pad_id == IMAGE_PAD_TOKEN) {
      t = grid_thw[img_idx][0];
      h = grid_thw[img_idx][1];
      w = grid_thw[img_idx][2];
    } else {
      t = 1;
      h = grid_thw[0][1];
      w = grid_thw[0][2];
    }
    --remain_images;
    size_t ed = ed_image;

    int llm_grid_t = t;
    int llm_grid_h = h / spatial_merge_size;
    int llm_grid_w = w / spatial_merge_size;
    size_t text_len = ed - st;

    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      int max_val = 0;
      for (const auto &row : llm_pos_ids_list.back()) {
        int row_max = max(row);
        if (row_max > max_val) {
          max_val = row_max;
        }
      }
      st_idx = max_val + 1;
    }

    // 处理文本部分的位置索引
    std::vector<std::vector<int>> text_pos(3);
    std::vector<int> text_range = arange(0, text_len);
    for (int j = 0; j < 3; ++j) {
      std::vector<int> temp(text_range);
      for (int &val : temp) {
        val += st_idx;
      }
      text_pos[j] = temp;
    }
    llm_pos_ids_list.push_back(text_pos);

    // 处理图像部分的位置索引

    std::vector<int> t_index;
    for (int i = 0; i < llm_grid_t; i++) {
      auto time_val = i * second_per_grid_t * tokens_per_second;
      t_index.insert(t_index.end(), llm_grid_h * llm_grid_w, time_val);
    }

    std::vector<int> h_index;
    for (int n = 0; n < llm_grid_t; ++n) {
      for (int p = 0; p < llm_grid_h; ++p) {
        for (int q = 0; q < llm_grid_w; ++q) {
          h_index.push_back(p);
        }
      }
    }

    std::vector<int> w_index;
    for (int n = 0; n < llm_grid_t; ++n) {
      for (int p = 0; p < llm_grid_h; ++p) {
        for (int q = 0; q < llm_grid_w; ++q) {
          w_index.push_back(q);
        }
      }
    }

    std::vector<std::vector<int>> grid_pos = {t_index, h_index, w_index};
    for (auto &row : grid_pos) {
      for (int &val : row) {
        val += text_len + st_idx;
      }
    }
    llm_pos_ids_list.push_back(grid_pos);

    st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
  }
  if (st < input_ids.size()) {
    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      int max_val = 0;
      for (const auto &row : llm_pos_ids_list.back()) {
        int row_max = max(row);
        if (row_max > max_val) {
          max_val = row_max;
        }
      }
      st_idx = max_val + 1;
    }
    size_t text_len = input_ids.size() - st;
    std::vector<std::vector<int>> text_pos(3);
    std::vector<int> text_range = arange(0, text_len);
    for (int j = 0; j < 3; ++j) {
      std::vector<int> temp(text_range);
      for (int &val : temp) {
        val += st_idx;
      }
      text_pos[j] = temp;
    }
    llm_pos_ids_list.push_back(text_pos);
  }

  std::vector<std::vector<int>> llm_positions = cat(llm_pos_ids_list, 1);
  return llm_positions;
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
    std::cout << "\nMode(ocr,table,formula,chart): ";
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
    if (input_str == "ocr"){
      input_str = "OCR:";
    }else if (input_str == "table"){
      input_str = "Table Recognition:";
    }else if (input_str == "formula"){
      input_str = "Formula Recognition:";
    }else if (input_str == "chart"){
      input_str = "Chart Recognition:";
    }else{
      std::cout << "Unsupported mode. Please enter one of the following modes: ocr, table, formula, chart." << std::endl;
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
      std::vector<std::vector<int>> grid_thws;
      for (int i = 0; i < num_medias; ++i) {
        auto ret = process_image(pixel_values[i], medias[i], config);
        if (ret == false) {
          std::cerr << "Error processing image: " << medias[i] << std::endl;
          continue;
        }
        grid_thws.push_back(config.grid_thw);
      }
      std::string sentence_input = build_image_prompt(input_str, grid_thws);
      std::vector<int> tokens = encode_input(sentence_input);
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
      auto position_ids = get_rope_index(tokens, grid_thws, IMAGE_PAD_TOKEN);

      // 找到三维数组position_ids中的最大值
      for (int val : position_ids[0]) {
        if (val > max_posid) {
          max_posid = val;
        }
      }

      // 将三维数组position_ids转换维1维
      std::vector<int> position_ids_1d;
      for (const auto &dim_tensor : position_ids) {
        position_ids_1d.insert(position_ids_1d.end(), dim_tensor.begin(),
                               dim_tensor.end());
      }
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
    while (token != ID_EOS && model.history_length < model.SEQLEN) {
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
      std::cout << "Vision [" << config.grid_thw[0] << ", "
                << config.grid_thw[1] << ", " << config.grid_thw[2]
                << "]: " << duration_vit / 1000.0f << " s" << std::endl;
    }
    std::cout << "Input Tokens: " << input_token_num
              << ", Output Tokens: " << output_token_num + 1 << std::endl;
  }
}

// 构建提示
std::string ChatPipe::build_text_prompt(const std::string &input_str) {
  std::string prompt = "<|im_start|>user\n";
  prompt += input_str + "\n<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

std::string
ChatPipe::build_image_prompt(const std::string &input_str,
                             const std::vector<std::vector<int>> &grid_thw) {
  std::string prompt = "<|begin_of_sentence|>User: ";
  int num_images = grid_thw.size();
  for (int i = 0; i < num_images; i++) {
    int h = grid_thw[i][1];
    int w = grid_thw[i][2];
    int pad_len = h * w / 4;
    prompt += "<|IMAGE_START|>";
    for (int j = 0; j < pad_len; j++) {
      prompt += "<|IMAGE_PLACEHOLDER|>";
    }
    prompt += "<|IMAGE_END|>";
  }
  prompt += input_str + "\nAssistant: ";
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
  model.forward_vit(pixel_values.data(), config.grid_thw, vit_offset);
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