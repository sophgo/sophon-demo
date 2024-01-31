//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef LPRNET_MULTI_H
#define LPRNET_MULTI_H

#include <dirent.h>
#include <iostream>
#include <map>
#include <mutex>
#include <unistd.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>
#include <sys/stat.h>

#include "json.hpp"
#include "datapipe.hpp"

#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include "bmruntime_cpp.h"
#include "bmruntime_interface.h"

// define data struct
struct bmimage {
  std::shared_ptr<cv::Mat> mat;
  std::shared_ptr<bm_image> bmimg;
  std::string filename;
  int channel_id;
  int frame_id;
};

struct bmtensor {
  std::shared_ptr<bm_tensor_t> bmtensor;
  std::vector<int> channel_ids;
  std::vector<int> frame_ids;
};

struct rec_data {
  std::string rec_res;
  int channel_id;
  int frame_id;
};

class LPRNet {
 public:
  LPRNet(int dev_id, std::string bmodel_path,
               int pre_thread_num,
               int post_thread_num, int queue_size);
  ~LPRNet();
  // start thread
  void run();
  // push data into queue where preprocess threads pop data
  void push_m_queue_decode(std::shared_ptr<bmimage> in);
  void set_preprocess_exit();
  // pop data where postprocess threads push data
  int pop_m_queue_post(std::shared_ptr<rec_data>& out);

 private:
  // thread-safe data queue
  DataPipe<std::shared_ptr<bmimage>> m_queue_decode;
  DataPipe<std::shared_ptr<bmtensor>> m_queue_pre;
  DataPipe<std::shared_ptr<bmtensor>> m_queue_infer;
  DataPipe<std::shared_ptr<rec_data>> m_queue_post;

  // inference
  int dev_id;
  bm_handle_t m_handle;
  const bm_net_info_t* m_netinfo;
  bool is_soc;
  std::unique_ptr<bmruntime::Context> m_ctx;
  std::unique_ptr<bmruntime::Network> m_net;
  std::vector<bmruntime::Tensor*> m_inputs;
  std::vector<bmruntime::Tensor*> m_outputs;

  // network info
  int m_net_h, m_net_w;
  int batch_size;
  int class_num;
  int seq_len;
  float input_scale;
  float output_scale;
  int channel_num;
  bm_image_data_format_ext img_dtype;
  std::vector<std::vector<bm_image>> m_resized_imgs;
  std::string bmodel_path;

  // thread
  std::vector<std::thread> threads;
  int pre_thread_num;
  int post_thread_num;

  // std and mean in preprocess
  bmcv_convert_to_attr converto_attr;

  // end flag
  int decode_activate_thread_num;
  int pre_activate_thread_num;
  int infer_activate_thread_num;
  int post_activate_thread_num;
  std::mutex m_mutex_decode_end;
  std::mutex m_mutex_pre_end;
  std::mutex m_mutex_infer_end;
  std::mutex m_mutex_post_end;

  // pipeline func
  void preprocess(int process_id);
  void inference();
  void postprocess(int process_id);

  // aux func
  std::shared_ptr<float> get_cpu_data(std::shared_ptr<bm_tensor_t> tensor);
  std::string get_res(int pred_num[]);
  bm_handle_t get_handle() {return m_handle;}
};

#endif