//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include "yolov5.hpp"
#include "draw.hpp"
#include "VideoConsole.h"
#define DRAW_ACC 1

struct demo_config{
  int dev_id;
  std::string bmodel_path;
  std::vector<std::string> input_paths; // 图片以/结尾                   
  //std::vector<std::string> output_paths; // rtsp url                   
  std::vector<bool> is_videos; 
  std::vector<int> skip_frame_nums;
  int queue_size;
  int num_pre;
  int num_post;
  std::string classnames;
  float conf_thresh;
  float nms_thresh;
  int display_rows;
  int display_cols;
};

// 跳帧的属性在是图片的时候设置为0
// 图片路径以‘/’结尾
void get_config(std::string& json_path, demo_config& config){

  std::ifstream istream;
  istream.open(json_path);
  assert (istream.is_open());
  
  json demo_json;
  istream >> demo_json;
  istream.close();

  config.dev_id = demo_json.find("dev_id")->get<int>();

  config.bmodel_path = demo_json.find("bmodel_path")->get<std::string>();

  auto channel_list_it = demo_json.find("channels");

  for (auto & channel_it : *channel_list_it){
    config.input_paths.emplace_back(channel_it.find("url")->get<std::string>());
    //config.output_paths.emplace_back(channel_it.find("output_url")->get<std::string>());
    config.is_videos.emplace_back(channel_it.find("is_video")->get<bool>());
    config.skip_frame_nums.emplace_back(channel_it.find("skip_frame")->get<int>());
  }

  config.queue_size = demo_json.find("queue_size")->get<int>();
  config.num_pre = demo_json.find("num_pre")->get<int>();
  config.num_post = demo_json.find("num_post")->get<int>();
  config.classnames = demo_json.find("class_names")->get<std::string>();
  config.conf_thresh = demo_json.find("conf_thresh")->get<float>();
  config.nms_thresh = demo_json.find("nms_thresh")->get<float>();

  config.display_rows = demo_json.find("display_rows")->get<int>();
  config.display_cols = demo_json.find("display_cols")->get<int>();
}

void worker_sink(YOLOv5& yolov5){
  while (true){
    std::shared_ptr<DataPost> box_data;
    std::shared_ptr<cv::Mat> origin_image;
    std::string name;
    int ret = yolov5.get_post_data(box_data, origin_image);
    if (ret){
      return;
    }
  }
  
}

std::chrono::time_point<std::chrono::high_resolution_clock> before_cap_init;

int main(int argc, char *argv[]){

  QApplication app(argc, argv);
  const char *keys="{config | config.json | config path}"
                    "{help | 0 | print help information.}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  // 解析json
  demo_config config;
  std::string json_path = parser.get<std::string>("config");
  get_config(json_path, config);

  std::vector<json> results_json;
  std::vector<std::string> class_names;

  int display_channel_rows = config.display_rows;
  int display_channel_cols = config.display_cols;
  int channel_num = display_channel_rows * display_channel_cols;
  VideoConsole<cv::Mat> video_console(display_channel_rows,display_channel_cols,channel_num,channel_num);

#if DRAW_ACC
  struct stat info;
  auto coco_names_file = config.classnames;
  if (stat(coco_names_file.c_str(), &info) != 0) {
    std::cerr << "Cannot find classnames file." << std::endl;
    exit(1);
  }
  // coco names
  std::ifstream ifs(coco_names_file);
  if (ifs.is_open()) {
    std::string line;
    while(std::getline(ifs, line)) {
      line = line.substr(0, line.length() - 1);
      class_names.push_back(line);
    }
  }

#endif

  YOLOv5 yolov5(config.dev_id, 
                config.bmodel_path, 
                config.input_paths, 
                config.is_videos, 
                config.skip_frame_nums,
                config.queue_size,
                config.num_pre, 
                config.num_post, 
                config.conf_thresh, 
                config.nms_thresh);

#if DRAW_ACC
  auto thread_draw = std::thread([&]{worker_draw(yolov5, class_names, &video_console);});
  //不建议使用 join 函数：调用 join() 会使调用线程（通常是主线程）阻塞，直到被调用的线程完成执行。
  //但被调用的线程 thread_draw 内部是死循环，会导致主线程一直被阻塞，从而导致qt无法正常显示。
  thread_draw.detach();
#else
  auto thread_sink = std::thread([&]{worker_sink(yolov5);});
  if (thread_sink.joinable()){
    thread_sink.join();
  }

#endif
    return app.exec();
}
