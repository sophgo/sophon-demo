//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "vlpr_bmcv.hpp"
#include "json.hpp"
#include <fstream>
using json = nlohmann::json;


void get_config(std::string& json_path, demo_config& config){
  std::ifstream istream;
  istream.open(json_path);
  assert(istream.is_open());
  
  json demo_json;
  istream >> demo_json;
  istream.close();

  config.dev_id = demo_json.find("dev_id")->get<int>();
  config.yolov5_bmodel_path = demo_json.find("yolov5_bmodel_path")->get<std::string>();
  config.lprnet_bmodel_path = demo_json.find("lprnet_bmodel_path")->get<std::string>();

  auto channel_list_it = demo_json.find("channels");
  std::vector<std::string> input_paths;
  std::vector<bool> is_videos;
  for (auto & channel_it : *channel_list_it){
    config.input_paths.emplace_back(channel_it.find("url")->get<std::string>());
    config.is_videos.emplace_back(channel_it.find("is_video")->get<bool>());
  }

  config.yolov5_num_pre = demo_json.find("yolov5_num_pre")->get<int>();
  config.yolov5_num_post = demo_json.find("yolov5_num_post")->get<int>();
  config.lprnet_num_pre = demo_json.find("lprnet_num_pre")->get<int>();
  config.lprnet_num_post = demo_json.find("lprnet_num_post")->get<int>();
  config.yolov5_queue_size = demo_json.find("yolov5_queue_size")->get<int>();
  config.lprnet_queue_size = demo_json.find("lprnet_queue_size")->get<int>();
  config.yolov5_conf_thresh = demo_json.find("yolov5_conf_thresh")->get<float>();
  config.yolov5_nms_thresh = demo_json.find("yolov5_nms_thresh")->get<float>();

  config.frame_sample_interval = demo_json.find("frame_sample_interval")->get<int>();
  config.in_frame_num = demo_json.find("in_frame_num")->get<int>();
  config.out_frame_num = demo_json.find("out_frame_num")->get<int>();
  config.crop_thread_num = demo_json.find("crop_thread_num")->get<int>();
  config.push_data_thread_num = demo_json.find("push_data_thread_num")->get<int>();
  config.perf_out_time_interval = demo_json.find("perf_out_time_interval")->get<int>();
}

int main(int argc, char** argv) {
  const char *keys="{config_path | configs/config_se7.json | config path}"
                    "{help | 0 | print help information.}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  std::string json_fpath = parser.get<std::string>("config_path");
  if (access( json_fpath.c_str(), F_OK ) == -1) {
    std::cout << "config file: " << json_fpath << " not exit!" << std::endl;
    return -1;
  }

  demo_config config;
  // parse cmd params
  get_config(json_fpath, config);

  VLPR vlpr(config);
  vlpr.run();

  return 0;
}

