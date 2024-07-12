//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef DWA_H
#define DWA_H
// #include <algorithm>
// #include "bmcv_api_ext.h"
#include <mutex>
#include "frame.h"
#include<queue>
#include <thread>
#include <unordered_map>
#include <fstream>

#define FFALIGN(x, a) (((x) + (a)-1) & ~((a)-1))
static std::string get_json(std::string path){
  std::string json;
  std::ifstream file(path);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            json += line;
        }
        file.close();
    }

    std::cout << json << std::endl;
    return json;
}
enum DwaMode {
  DWA_GDC_MODE,
  DWA_FISHEYE_MODE,
};

class Dwa{
 public:
  Dwa();
    Dwa(const Dwa& other) {
    }

  ~Dwa() ;

  int init(const std::string& json) ;
 

  int fisheye_work(std::shared_ptr<Frame> &input_image);
int dwa_gdc_work(std::shared_ptr<Frame> &mFrame);
  float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h,
                                bool* pIsAligWidth);
  const char* CONFIG_INTERNAL_IS_GRAY_FILED = "is_gray";
  const char* CONFIG_INTERNAL_IS_RESIZE_FILED = "is_resize";
  const char* CONFIG_INTERNAL_IS_ROT_FILED = "is_rot";
  const char* CONFIG_INTERNAL_DIS_MODE_FILED = "dis_mode";
  const char* CONFIG_INTERNAL_GRID_NAME_FILED = "grid_name";
  const char* CONFIG_INTERNAL_USE_GRIDE_FILED = "use_grid";
  const char* CONFIG_INTERNAL_GRIDE_SIZE_FILED = "grid_size";

  const char* CONFIG_INTERNAL_SRC_H_FILED = "src_h";
  const char* CONFIG_INTERNAL_SRC_W_FILED = "src_w";
  const char* CONFIG_INTERNAL_DST_H_FILED = "dst_h";
  const char* CONFIG_INTERNAL_DST_W_FILED = "dst_w";
  const char* CONFIG_INTERNAL_TMP_H_FILED = "resize_h";
  const char* CONFIG_INTERNAL_TMP_W_FILED = "resize_w";
  const char* CONFIG_INTERNAL_DWA_MODE_FILED = "dwa_mode";
  
//   const char* CONFIG_INTERNAL_SRC_H_FILED ;
//   const char* CONFIG_INTERNAL_SRC_W_FILED ;
//   const char* CONFIG_INTERNAL_DST_H_FILED ;
//   const char* CONFIG_INTERNAL_DST_W_FILED ;
//   const char* CONFIG_INTERNAL_TMP_H_FILED ;
//   const char* CONFIG_INTERNAL_TMP_W_FILED ;
//   const char* CONFIG_INTERNAL_DWA_MODE_FILED ;



// const char* CONFIG_INTERNAL_IS_GRAY_FILED;
//   const char* CONFIG_INTERNAL_IS_RESIZE_FILED;
//   const char* CONFIG_INTERNAL_IS_ROT_FILED;
//   const char* CONFIG_INTERNAL_DIS_MODE_FILED;
//   const char* CONFIG_INTERNAL_GRID_NAME_FILED;
//   const char* CONFIG_INTERNAL_USE_GRIDE_FILED;
//   const char* CONFIG_INTERNAL_GRIDE_SIZE_FILED ;

  int src_h, src_w, dst_h, dst_w, tmp_h, tmp_w;

  bm_image_format_ext_ src_fmt;
  int subId = 0;
  bool is_resize = false;
  bool is_rot = false;

  bmcv_usage_mode dis_mode;
  DwaMode dwa_mode;

  bmcv_rot_mode rot_mode;

  std::string grid_name;

  bmcv_gdc_attr ldc_attr = {0};
  bmcv_fisheye_attr_s fisheye_attr = {0};

  std::mutex dwa_lock;

 private:
//   ::sophon_stream::common::FpsProfiler mFpsProfiler;

  std::unordered_map<std::string, DwaMode> dwa_mode_map{
      {"DWA_GDC_MODE", DwaMode::DWA_GDC_MODE},
      {"DWA_FISHEYE_MODE", DwaMode::DWA_FISHEYE_MODE}};

  std::unordered_map<std::string, bmcv_usage_mode> fisheye_mode_map{
      {"BMCV_MODE_PANORAMA_360", bmcv_usage_mode::BMCV_MODE_PANORAMA_360},
      {"BMCV_MODE_PANORAMA_180", bmcv_usage_mode::BMCV_MODE_PANORAMA_180},
      {"BMCV_MODE_01_1O", bmcv_usage_mode::BMCV_MODE_01_1O},
      {"BMCV_MODE_02_1O4R", bmcv_usage_mode::BMCV_MODE_02_1O4R},
      {"BMCV_MODE_03_4R", bmcv_usage_mode::BMCV_MODE_03_4R},
      {"BMCV_MODE_04_1P2R", bmcv_usage_mode::BMCV_MODE_04_1P2R},
      {"BMCV_MODE_05_1P2R", bmcv_usage_mode::BMCV_MODE_05_1P2R},
      {"BMCV_MODE_06_1P", bmcv_usage_mode::BMCV_MODE_06_1P},
      {"BMCV_MODE_07_2P", bmcv_usage_mode::BMCV_MODE_07_2P},
      {"BMCV_MODE_STEREO_FIT", bmcv_usage_mode::BMCV_MODE_STEREO_FIT},
      {"BMCV_MODE_MAX", bmcv_usage_mode::BMCV_MODE_MAX}};
};


class DWA_PIPE  {
 public:
  DWA_PIPE(){};
  ~DWA_PIPE(){};
 int set_in_queue(std::shared_ptr<DatePipe> queue){
    input_frames=queue;
  } int set_in_lock(std::shared_ptr<std::mutex> m){
    input_queue_lock=m;
  }
   int set_out_queue(std::shared_ptr<DatePipe> queue){
    output_frames=queue;
  } int set_out_lock(std::shared_ptr<std::mutex> m){
    output_queue_lock=m;
  }
  int get_data(std::shared_ptr<Frame> &data1,std::shared_ptr<Frame> &data2){
    while(output_frames->frames[0].empty());
    data1=std::move(output_frames->frames[0].front());
    while(output_frames->frames[1].empty());
    data2=std::move(output_frames->frames[1].front());
    for(int i=0;i<output_frames->frames.size();i++){
      std::unique_lock<std::mutex> lock(*output_queue_lock);
      output_frames->frames[i].pop();

    }
  }
  int init(std::vector<std::string> &dwa_path);
  void start(int dwa_id);
  void uninit();
  
 private:
 std::shared_ptr<std::mutex> input_queue_lock;
    std::shared_ptr<std::mutex> output_queue_lock;

std::shared_ptr<DatePipe> output_frames;

std::shared_ptr<DatePipe> input_frames;
  std::vector<Dwa> dwas;

 
  std::vector<std::thread> threads;

  
};
#endif