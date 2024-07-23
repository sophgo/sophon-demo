//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1

#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "ff_video_decode.h"
#include "ff_video_encode.h"
#include "opencv2/opencv.hpp"

using namespace std;

// 检查文件名是否为图像文件
bool is_image(const std::string& filename) {
  // 定义图像文件扩展名列表
  std::vector<std::string> image_extensions = {".jpg", ".png", ".bmp"};

  // 将文件名转换为小写
  std::string lower_filename = filename;
  std::transform(lower_filename.begin(), lower_filename.end(),
                 lower_filename.begin(), ::tolower);

  // 使用 std::any_of 和 lambda 表达式来检查文件名是否以任何一个扩展名结尾
  return std::any_of(image_extensions.begin(), image_extensions.end(),
                     [&lower_filename](const std::string& ext) {
                       return lower_filename.size() >= ext.size() &&
                              lower_filename.compare(
                                  lower_filename.size() - ext.size(),
                                  ext.size(), ext) == 0;
                     });
}

int picture_decode_encode(int dev_id, std::string filename) {
  VideoDec_FFMPEG decoder;
  int ret = decoder.openDec(filename.c_str(), 1, NULL, 100, dev_id);
  if (ret < 0) {
    std::cerr << "Avframe: Open input media failed" << std::endl;
    return -1;
  }

  AVFrame* picture = av_frame_alloc();
  decoder.grabFrame(picture);

  // 生成yuv mat，此时mat的释放和avframe是相互独立的，avframe需要单独释放
  cv::Mat ocv_frame(picture,
                    dev_id | BM_MAKEFLAG(cv::UMatData::AVFRAME_ATTACHED, 0, 0));

  cv::Mat bgr_mat;
  cv::bmcv::toMAT(ocv_frame, bgr_mat, true);
  cv::imwrite("output.jpg", bgr_mat);

  ocv_frame.release();
  picture = nullptr;

  return 0;
}

void video_push_stream(cv::CommandLineParser parser) {
  std::string input_path = parser.get<std::string>("input_path");
  std::string output_path = parser.get<std::string>("output_path");
  int dev_id = parser.get<int>("dev_id");
  bool compressed_nv12 = parser.get<bool>("compressed_nv12");
  int height = parser.get<int>("height");
  int width = parser.get<int>("width");
  std::string enc_fmt = parser.get<std::string>("enc_fmt");
  int bitrate = parser.get<int>("bitrate");
  std::string pix_fmt = parser.get<std::string>("pix_fmt");
  int gop = parser.get<int>("gop");
  int gop_preset = parser.get<int>("gop_preset");
  int framerate = parser.get<int>("framerate");
  int enc_id = AV_CODEC_ID_H264;
  int inputformat = AV_PIX_FMT_YUV420P;
  AVPixelFormat pix_fmt_;
  if (pix_fmt == "I420") {
    pix_fmt_ = AV_PIX_FMT_YUV420P;
  } else if (pix_fmt == "NV12") {
    pix_fmt_ = AV_PIX_FMT_NV12;
  } else {
    printf("only support I420, NV12, check pix_fmt: {} \n.", pix_fmt);
  }

  AVFrame* frame = av_frame_alloc();

  VideoDec_FFMPEG decoder;
  int ret = decoder.openDec(input_path.c_str(), 1, NULL, 100, dev_id);

  VideoEnc_FFMPEG encoder;

  ret = encoder.openEnc(output_path.c_str(), "h264_bm", 0, framerate, width,
                        height, pix_fmt_, bitrate, dev_id);

  int count = 0;
  while (true) {
    int got_frame = 0;

    got_frame = decoder.grabFrame(frame);
    count++;
    sleep(0.01);

    if (got_frame) {
      encoder.writeFrame(frame);
      printf("is encoding %d\n", count);
    } else {
      printf("Video read fail!\n");
      break;
    }
    av_frame_unref(frame);
  }
  encoder.closeEnc();
  printf("encode finish! \n");
}

int main(int argc, char* argv[]) {
  cout.setf(ios::fixed);
  // get params
  const char* keys =
      "{help h usage ? |      | print this message}"
      "{input_path     | ../datasets/test_car_person_1080P.mp4 | Path or rtsp "
      "url to the video/image file.}"
      "{output_path    |      | Local file path or stream url}"
      "{dev_id      | 0    | Device id}"
      "{compressed_nv12| true | Whether the format of decoded output is "
      "compressed NV12.}"
      "{height         | 1080 | The height of the encoded video}"
      "{width          | 1920 | The width of the encoded video}"
      "{enc_fmt        | h264_bm | encoded video format, h264_bm/hevc_bm}"
      "{bitrate        | 2000 | encoded bitrate}"
      "{pix_fmt        | NV12 | encoded pixel format}"
      "{gop            | 32   | gop size}"
      "{gop_preset     | 2    | gop_preset}"
      "{framerate      | 25   | encode frame rate}";
  cv::CommandLineParser parser(argc, argv, keys);

  std::string input_path = parser.get<std::string>("input_path");
  int dev_id = parser.get<int>("dev_id");

  if (!parser.check()) {
    parser.printErrors();
    return 1;
  }

  // init params
  if (is_image(input_path)) {
    picture_decode_encode(dev_id, input_path);

  } else {
    video_push_stream(parser);
  }

  return 0;
}
