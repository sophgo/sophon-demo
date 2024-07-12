//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cvdemo.hpp"

#include <fstream>
#include <string>
#include <vector>
#define BASE64_CPU 1

CvDemo::CvDemo(int server_port, int mFps, std::string decode_path,
               std::vector<std::string>& dwa_path, std::string blend_path) {
  std::cout << "CvDemo ctor .." << std::endl;
  // decoder.set_queue(output_frames);

  std::shared_ptr<std::mutex> decoder_output_lock =
      std::make_shared<std::mutex>();
  std::shared_ptr<DatePipe> decoder_output_frames =
      std::make_shared<DatePipe>();

  std::shared_ptr<std::mutex> dwa_output_lock = std::make_shared<std::mutex>();
  std::shared_ptr<DatePipe> dwa_output_frames = std::make_shared<DatePipe>();
  decoder_output_frames->frames.resize(dwa_path.size());
  dwa_output_frames->frames.resize(dwa_path.size());

  decoder.set_out_lock(decoder_output_lock);
  decoder.set_out_queue(decoder_output_frames);
  dwa.set_in_lock(decoder_output_lock);
  dwa.set_in_queue(decoder_output_frames);
  dwa.set_out_lock(dwa_output_lock);
  dwa.set_out_queue(dwa_output_frames);

  decoder.init(get_json(decode_path));
  // std::cout<<"size"<<decoder.output_frames.size()<<std::endl;
  // for(int i=0;i<decoder.output_frames.size();i++){
  //     Dwa mdwa;
  //     ldwa.init(get_json(dwa_path[i]));
  //     dwa.push_back(std::move(mdwa));
  // }
  dwa.init(dwa_path);
  // ldwa.init(get_json(dwa_path[0]));
  // rdwa.init(get_json(dwa_path[1]));
  blend.init(get_json(blend_path));
  auto wss = std::make_shared<WebSocketServer>(server_port, mFps);
  std::thread t([wss]() { wss->run(); });
  mwss = wss;
  mWSSThreads.push_back(std::move(t));
}

CvDemo::~CvDemo() {
  for (auto& thread : mWSSThreads) {
    if (thread.joinable()) thread.join();
  }
  std::cout << "CvDemo dtor ..." << std::endl;
}

const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

/// Encode a char buffer into a base64 string
/**
 * @param input The input data
 * @param len The length of input in bytes
 * @return A base64 encoded string representing input
 */
std::string base64_encode(unsigned char const* input, size_t len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (len--) {
    char_array_3[i++] = *(input++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] =
          ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] =
          ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; (i < 4); i++) {
        ret += base64_chars[char_array_4[i]];
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 3; j++) {
      char_array_3[j] = '\0';
    }

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] =
        ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] =
        ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++) {
      ret += base64_chars[char_array_4[j]];
    }

    while ((i++ < 3)) {
      ret += '=';
    }
  }

  return ret;
}

std::string frame_to_base64(Frame& frame) {
#if ENABLE_TIME_LOG
  timeval time1, time2, time3, time4, time5;
  gettimeofday(&time1, NULL);
#endif
  unsigned char* jpegData = nullptr;
  size_t nBytes = 0;
  bm_image bgr_;
  if (frame.mSpDataOsd != nullptr) {
    bgr_ = *(frame.mSpDataOsd);
  } else {
    bgr_ = *(frame.mSpData);
  }
  bm_handle_t handle_ = bm_image_get_handle(&bgr_);

  bm_image yuv_;

  bm_image_create(handle_, bgr_.height, bgr_.width, FORMAT_YUV420P,
                  bgr_.data_type, &yuv_);
  bm_image_alloc_dev_mem_heap_mask(yuv_, STREAM_VPU_HEAP_MASK);
#if ENABLE_TIME_LOG
  gettimeofday(&time2, NULL);
#endif
  // bmcv_image_storage_convert(handle_, 1, &bgr_, &yuv_);
  bmcv_rect_t rect_{0, 0, bgr_.width, bgr_.height};
  bmcv_image_vpp_convert(handle_, 1, bgr_, &yuv_, &rect_);
#if ENABLE_TIME_LOG
  gettimeofday(&time3, NULL);
#endif
  bmcv_image_jpeg_enc(handle_, 1, &yuv_, (void**)&jpegData, &nBytes);
#if ENABLE_TIME_LOG
  gettimeofday(&time4, NULL);
#endif
  bm_image_destroy(yuv_);

#if BASE64_CPU
  // for cpu
  std::string res = base64_encode(jpegData, nBytes);
#else
  // for bmcv
  unsigned long origin_len[2] = {nBytes, 0};
  unsigned long encode_len[2] = {(origin_len[0] + 2) / 3 * 4, 0};
  std::string res(encode_len[0], '\0');
  bmcv_base64_enc(handle_, bm_mem_from_system(jpegData),
                  bm_mem_from_system(const_cast<char*>(res.c_str())),
                  origin_len);
#endif

#if ENABLE_TIME_LOG
  gettimeofday(&time5, NULL);
  double time_delta1 =
      1000 * ((time5.tv_sec - time1.tv_sec) +
              (double)(time5.tv_usec - time1.tv_usec) / 1000000.0);
  double time_delta2 =
      1000 * ((time3.tv_sec - time2.tv_sec) +
              (double)(time3.tv_usec - time2.tv_usec) / 1000000.0);
  double time_delta3 =
      1000 * ((time4.tv_sec - time3.tv_sec) +
              (double)(time4.tv_usec - time3.tv_usec) / 1000000.0);
  double time_delta4 =
      1000 * ((time5.tv_sec - time4.tv_sec) +
              (double)(time5.tv_usec - time4.tv_usec) / 1000000.0);

  IVS_INFO(
      "storage convert time = {0}, jpeg_enc time = {1}, base64_enc time = {2}, "
      "total time = {3}",
      time_delta2, time_delta3, time_delta4, time_delta1);
#endif

  delete jpegData;
  return res;
}
int CvDemo::resize_work(std::shared_ptr<Frame>& resObj) {
  if (resObj != nullptr) {
    std::shared_ptr<bm_image> resize_image = nullptr;
    resize_image.reset(new bm_image, [](bm_image* p) {
      bm_image_destroy(*p);
      delete p;
      p = nullptr;
    });

    bm_status_t ret =
        bm_image_create(resObj->mHandle, resObj->mSpData->height * 0.4,
                        resObj->mSpData->width * 0.4, FORMAT_YUV420P,
                        DATA_TYPE_EXT_1N_BYTE, resize_image.get());
    bm_image_alloc_dev_mem(*resize_image, 1);

    bmcv_rect_t crop_rect{0, 0, (unsigned int)resObj->mSpData->width,
                          (unsigned int)resObj->mSpData->height};
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    padding_attr.dst_crop_h = (unsigned int)resObj->mSpData->height * 0.4;
    padding_attr.dst_crop_w = (unsigned int)resObj->mSpData->width * 0.4;

    ret = bmcv_image_vpp_convert_padding(resObj->mHandle, 1, *resObj->mSpData,
                                         resize_image.get(), &padding_attr,
                                         &crop_rect);

    resObj->mSpData = resize_image;

    resObj->mWidth = resObj->mSpData->width;
    resObj->mHeight = resObj->mSpData->height;
  }
}
int CvDemo::Detect() {
  int ret = 0;
  std::shared_ptr<Frame> blend_frame = std::make_shared<Frame>();
  std::shared_ptr<Frame> l_frame, r_frame;
  dwa.get_data(l_frame, r_frame);
  auto start3 = std::chrono::high_resolution_clock::now();

  ret = blend.blend_work(l_frame, r_frame, blend_frame);

  auto end3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration3 = end3 - start3;
  std::cout << "blend程序执行时间：" << duration3.count() << " ms" << std::endl;
  resize_work(blend_frame);
  std::string data = frame_to_base64(*blend_frame);
  mwss->pushImgDataQueue(data);

  CV_Assert(ret == 0);

  return ret;
}
