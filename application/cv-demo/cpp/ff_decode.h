//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef FF_DECODER_H
#define FF_DECODER_H
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <queue>

#include <thread>
// for bmcv_api_ext.h
#include "libyuv.h"
#include "opencv2/opencv.hpp"
extern "C" {
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include "common_defs.h"

#include "frame.h"

#define QUEUE_MAX_SIZE 5
#define EXTRA_FRAME_BUFFER_NUM 2
#define USEING_MEM_HEAP2 4
#define USEING_MEM_HEAP1 2

static const int DISCONNECTED_ERROR_CODE = -22;

typedef struct {
  uint8_t* start;
  int size;
  int pos;
} bs_buffer_t;

int read_buffer(void* opaque, uint8_t* buf, int buf_size);

/**
 * @brief convert avframe pix format to bm_image pix format.
 * @return bm_image_format_ext.
 */
int map_avformat_to_bmformat(int avformat);

/**
 * @brief convert avformat to bm_image.
 */
bm_status_t avframe_to_bm_image(bm_handle_t& handle, AVFrame* in, bm_image* out,
                                bool is_jpeg);

/**
 * @brief picture decode. support jpg and png
 */
std::shared_ptr<bm_image> picDec(bm_handle_t& handle, const char* path);
std::shared_ptr<bm_image> pngDec(bm_handle_t& handle, std::string input_name);
std::shared_ptr<bm_image> jpgDec(bm_handle_t& handle, std::string input_name);
std::shared_ptr<bm_image> bmpDec(bm_handle_t& handle, std::string input_name);



/**
 * video decode class
 * support video file and rtsp stream.
 *
 * VideoDecFFM create a thread to decode, convert AVFrame to bm_image, push
 * bm_image into the cache queue. When the queue is full, for video file, the
 * decode thread will sleep. For rtsp stream, the decode thread will pop the
 * front element of the queue.
 *
 */
class VideoDecFFM {
 public:
  VideoDecFFM();
  VideoDecFFM(const VideoDecFFM& other);
  ~VideoDecFFM();

  /* open video decoder, decode, convert avFrame to bm_image, push it into the
   * cache queue  */
  int openDec(bm_handle_t* dec_handle, const char* input);

  /* grab a bm_image from the cache queue*/
  std::shared_ptr<bm_image> grab(int& frame_id, int& eof, int64_t& pts,
                                 int sampleInterval);

  /* get frame count */
  void mFrameCount(const char* video_file, int& mFrameCount);

  /* close video decoder */
  void closeDec();

  /* pic dec */
  std::shared_ptr<bm_image> picDec(bm_handle_t& handle, const char* path);

  /* set fps */
  void setFps(int f);

 private:
  bool quit_flag = false;

  int is_rtsp;
  int is_rtmp;
  int is_gb28181;
  int is_camera;
  const char* rtsp_url;
  const char* rtmp_url;
  const char* gb28181_url;
  const char* camera_url;
  int width;
  int height;
  int pix_fmt;

  int frame_id;

  int video_stream_idx;
  int refcount;
  double fps;
  double frame_interval_time;  // ms
  struct timeval last_time;
  struct timeval current_time;

  AVFrame* frame;
  AVPacket* pkt;
  AVFormatContext* ifmt_ctx;
  AVCodec* decoder;
  AVCodecContext* video_dec_ctx;
  AVCodecParameters* video_dec_par;

  bm_handle_t* handle;
  int dev_id;
  std::mutex lock;
  std::queue<bm_image*> queue;

  std::string inputUrl;

  int openCodecContext(int* stream_idx, AVCodecContext** dec_ctx,
                       AVFormatContext* fmt_ctx, enum AVMediaType type,
                       int sophon_idx);

  int isNetworkError(int ret);

  void reConnectVideoStream();

  AVFrame* flushDecoder();

  AVFrame* grabFrame(int& eof);
};

#endif