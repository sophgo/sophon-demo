//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef CVDEMO_H
#define CVDEMO_H
#include "decoder.h"
#include "ff_decode.h"
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "blend.h"
#include "wss_boost.h"
#include "dwa.h"

#define USE_OPENCV 1
#define DEBUG 0



class CvDemo {
  public:
  CvDemo();
  
  ~CvDemo();
  CvDemo(int server_port, int mFps,std::string decode_path,std::vector<std::string> &dwa_path,std::string blend_path);
  int init();
  int Detect();
  int resize_work(
    std::shared_ptr<Frame> &resObj);

  private:
  
  Decoder decoder;
  DWA_PIPE dwa;
  Blend blend;
  std::vector<std::thread> mWSSThreads;
  std::shared_ptr<WebSocketServer> mwss;
  std::vector<std::queue<std::shared_ptr<Frame> >> output_frames;
  static std::mutex queue_lock;

};

#endif