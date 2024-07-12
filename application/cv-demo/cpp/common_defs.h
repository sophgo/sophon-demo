//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//



#include <iostream>
#include <sstream>
#include <string>

#include "bmcv_api_ext.h"
#include "bmruntime_interface.h"

// BM1688/CV186AH
#if BMCV_VERSION_MAJOR > 1

// For SDK Version 1.7, Heap Mask is the Same With 1684/1684X
#ifdef BMCV_VERSION_MINOR

#include "bmcv_api.h"
#define STREAM_VPP_HEAP_MASK 4
#define STREAM_VPU_HEAP_MASK 2
#define STREAM_NPU_HEAP_MASK 1
#define STREAM_VPP_HEAP 2
#define STREAM_VPU_HEAP 1
#define STREAM_NPU_HEAP 0

#else

// For SDK Version 1.6, Heap Mask is 1 and 2
#define STREAM_VPP_HEAP_MASK 2
#define STREAM_VPU_HEAP_MASK 2
#define STREAM_NPU_HEAP_MASK 1
#define STREAM_VPP_HEAP 1
#define STREAM_VPU_HEAP 1
#define STREAM_NPU_HEAP 0

typedef bmcv_padding_attr_t bmcv_padding_atrr_t;
/**
 * @brief To solve incompatible issue in a2 sdk
 *
 * @param image input bm_image
 * @return bm_status_t BM_SUCCESS change success, other values: change
 failed.
 */
static inline bm_status_t bm_image_destroy(bm_image& image) {
  return bm_image_destroy(&image);
}

static inline bm_status_t bmcv_image_crop(bm_handle_t handle, int crop_num,
                                          bmcv_rect_t* rects, bm_image input,
                                          bm_image* output) {
  return bmcv_image_vpp_convert(handle, crop_num, input, output, rects,
                                BMCV_INTER_LINEAR);
}

#endif

#else

// For 1684/1684X
#define STREAM_VPP_HEAP_MASK 4
#define STREAM_VPU_HEAP_MASK 2
#define STREAM_NPU_HEAP_MASK 1
#define STREAM_VPP_HEAP 2
#define STREAM_VPU_HEAP 1
#define STREAM_NPU_HEAP 0

#endif




#if LIBAVCODEC_VERSION_MAJOR > 58
static int avcodec_decode_video2(AVCodecContext* dec_ctx, AVFrame* frame,
                                 int* got_picture, AVPacket* pkt) {
  int ret;
  *got_picture = 0;
  ret = avcodec_send_packet(dec_ctx, pkt);
  if (ret == AVERROR_EOF) {
    ret = 0;
  } else if (ret < 0) {
    char err[256] = {0};
    av_strerror(ret, err, sizeof(err));
    fprintf(stderr, "Error sending a packet for decoding, %s\n", err);
    return -1;
  }
  while (ret >= 0) {
    ret = avcodec_receive_frame(dec_ctx, frame);
    if (ret == AVERROR(EAGAIN)) {
      ret = 0;
      break;
    } else if (ret == AVERROR_EOF) {
      printf("File end!\n");
      avcodec_flush_buffers(dec_ctx);
      ret = 0;
      break;
    } else if (ret < 0) {
      fprintf(stderr, "Error during decoding\n");
      break;
    }
    *got_picture += 1;
    break;
  }
  if (*got_picture > 1) {
    printf("got picture %d\n", *got_picture);
  }
  return ret;
}
static int avcodec_encode_video2(AVCodecContext* avctx, AVPacket* avpkt,
                                 const AVFrame* frame, int* got_packet_ptr) {
  int ret = avcodec_send_frame(avctx, frame);
  if (ret < 0) {
    return ret;
  }
  ret = avcodec_receive_packet(avctx, avpkt);
  if (ret < 0) {
    *got_packet_ptr = 0;
    return ret;
  } else {
    *got_packet_ptr = 1;
  }
  return 0;
}

#define av_find_input_format(x) const_cast<AVInputFormat*>(av_find_input_format(x))
#define avcodec_find_decoder(x) const_cast<AVCodec*>(avcodec_find_decoder(x))
#define av_guess_format(x1, x2, x3)  const_cast<AVOutputFormat*>(av_guess_format(x1, x2, x3))
#define avcodec_find_decoder_by_name(x) const_cast<AVCodec*>(avcodec_find_decoder_by_name(x))
#define avcodec_find_encoder_by_name(x) const_cast<AVCodec*>(avcodec_find_encoder_by_name(x));



#endif

