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

#include "cvwrapper.h"
#include "encoder.h"
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

int picture_decode_encode(sail::Handle handle, sail::Decoder decoder,
                          sail::Encoder encoder, sail::Bmcv bmcv) {
  sail::BMImage img;
  int decode_ret = decoder.read(handle, img);
  if (decode_ret != 0) {
    printf("Video read fail!\n");
    return -1;
  }
  sleep(0.01);

  vector<u_char> img_data;
  string extension = ".jpg";
  int size = encoder.pic_encode(extension, img, img_data);

  // 写入文件
  std::ofstream file("picture_encode.jpg", std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Could not open the file for writing" << std::endl;
    return 1;
  }
  file.write(reinterpret_cast<const char*>(img_data.data()), img_data.size());
  file.close();

  return 0;
}

void video_push_stream(sail::Handle handle, sail::Decoder decoder,
                       sail::Encoder encoder, sail::Bmcv bmcv) {
  int count = 0;
  while (true) {
    sail::BMImage img;
    int decode_ret = decoder.read(handle, img);
    count++;
    if (decode_ret != 0) {
      printf("Video read fail!\n");
      break;
    }
    sleep(0.01);
    int encode_ret = encoder.video_write(img);
    printf("is encoding %d\n", count);
    while (encode_ret != 0) {
      encode_ret = encoder.video_write(img);
    }
  }
}

int main(int argc, char* argv[]) {
  cout.setf(ios::fixed);
  // get params
  const char* keys =
      "{help | 0 | print help information.}"
      "{input_path     | ../datasets/test_car_person_1080P.mp4 | Path or rtsp "
      "url to the video/image file.}"
      "{output_path    |  output.mp4  | Local file path or stream url}"
      "{dev_id         | 0    | Device id}"
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
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

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

  if (!parser.check()) {
    parser.printErrors();
    return 1;
  }

  // init params
  auto handle = sail::Handle(dev_id);
  sail::Bmcv bmcv(handle);

  sail::Decoder decoder(input_path, compressed_nv12, dev_id);

  if (is_image(input_path)) {
    sail::Encoder encoder;
    picture_decode_encode(handle, decoder, encoder, bmcv);
    encoder.release();
  } else {
    std::ostringstream enc_params_stream;
    enc_params_stream << "width=" << width << ":height=" << height
                      << ":bitrate=" << bitrate << ":gop=" << gop
                      << ":gop_preset=" << gop_preset
                      << ":framerate=" << framerate;
    std::string enc_params = enc_params_stream.str();

    sail::Encoder encoder(output_path, dev_id, enc_fmt, pix_fmt, enc_params,
                          10);
    video_push_stream(handle, decoder, encoder, bmcv);
    encoder.release();
  }

  decoder.release();

  return 0;
}
