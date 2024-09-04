//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef DECODER_H
#define DECODER_H
#include <dirent.h>

#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <unordered_map>

#include "ff_decode.h"
#include "frame.h"
#include "json.hpp"

class Decoder {
 public:
  Decoder();
  ~Decoder();

  int init(const std::string& json);
  int set_out_queue(std::shared_ptr<DatePipe> queue) { output_frames = queue; }
  int set_out_lock(std::shared_ptr<std::mutex> m) { output_queue_lock = m; }
  int process(std::shared_ptr<Frame>& mframe, int decode_id);
  void start(int decode_id);
  void uninit();

 private:
  std::shared_ptr<std::mutex> output_queue_lock;
  std::shared_ptr<DatePipe> output_frames;
  bm_handle_t m_handle;
  std::vector<VideoDecFFM> decoders;

  std::string mUrl;
  int mDeviceId = 0;
  int mLoopNum;
  int mImgIndex;
  int mFrameCount;
  std::vector<std::string> mImagePaths;
  bmcv_rect_t mRoi;
  bool mRoiPredefined = false;

  double mFps;
  int mSampleInterval;

  // camera synchronization
  std::mutex decoder_mutex;
  std::condition_variable decoder_cv;
  int numThreadsReady = 0;
  int numThreadsTotal = 2;
  std::vector<std::thread> threads;
};
#endif