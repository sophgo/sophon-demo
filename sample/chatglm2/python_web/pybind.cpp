//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../cpp/chatglm2.hpp"

class ChatGLM2_web : public ChatGLM2{
public:
    std::string predict_next_token();
    std::string predict_first_token(const std::string &input_str);
};

std::string ChatGLM2_web::predict_first_token(const std::string &input_str) {

  history += ("[Round " + std::to_string(round + 1) + "]\n\n问：" + input_str +
              "\n\n答：");
  //int tok_num = 1;
  std::vector<int> tokens;
  sentencepiece.Encode(history, &tokens);
  if (tokens.empty()) {
    round = 0;
    history = "Sorry: your question is too wierd!!\n";
    return history;
  }
  // make sure token not too large
  if (tokens.size() > MAX_LEN - 10) {
    // reset
    if (round == 0) {
      history = "Error: your question is too large!\n";
      return history;
    }
    round = 0;
    history = "";
    return predict_first_token(input_str);
  }
  int token = forward_first(tokens);
  int pre_token = 0;
  std::string pre_word;
  std::string word;
  std::vector<int> pre_ids = {pre_token};
  std::vector<int> ids = {pre_token,token};
  sentencepiece.Decode(pre_ids, &pre_word);
  sentencepiece.Decode(ids, &word);
  std::string diff = word.substr(pre_word.size());
#ifdef PRINT
  printf("token %d",token);
  printf("diff %s",diff.c_str());
#endif
  history += diff;
  if (token_length < MAX_LEN) {
    token_length++;
  }
  return diff;
}

std::string ChatGLM2_web::predict_next_token() {
  int pre_token;
  pre_token = 0;
  int token = forward_next();
  if(token == EOS){
    round = 0;
    history = history.substr(history.size()/2);
    return "_GETEOS_";
  }
  std::string pre_word;
  std::string word;
  std::vector<int> pre_ids = {pre_token};
  std::vector<int> ids = {pre_token, token};
  sentencepiece.Decode(pre_ids, &pre_word);
  sentencepiece.Decode(ids, &word);
  std::string diff = word.substr(pre_word.size());
#ifdef PRINT
  printf("token %d",token);
  printf("diff %s",diff.c_str());
#endif
  history += diff;
  if (token_length < MAX_LEN) {
    token_length++;
  }else{
    round = 0;
    return "_GETMAX_";
  }
  return diff;
}



extern "C" {

ChatGLM2_web *ChatGLM2_with_devid_and_model(int devid, const char *bmodel_path, const char *tokenizer_path) {
  ChatGLM2_web *chat = new ChatGLM2_web();
  chat->init(devid, bmodel_path, tokenizer_path);
  return chat;
}

void ChatGLM2_delete(ChatGLM2_web *chat) { delete chat; }

void ChatGLM2_deinit(ChatGLM2_web *chat) { 
  chat->deinit();
}

const char *get_history(ChatGLM2_web *chat) {
  std::string str = chat->history;
  return strdup(str.c_str());
}

const char *set_history(ChatGLM2_web *chat, const char *history) {
  chat->history = history;
  return strdup(history);
}

const char *ChatGLM2_predict_first_token(ChatGLM2_web *chat, const char *input_str) {
  std::string str = chat->predict_first_token(input_str);
  return strdup(str.c_str());
}

const char *ChatGLM2_predict_next_token(ChatGLM2_web *chat) {
  std::string str = chat->predict_next_token();
  return strdup(str.c_str());
}

const int get_eos(ChatGLM2_web *chat){
  const int res = chat->EOS;
  return res;
}
}

