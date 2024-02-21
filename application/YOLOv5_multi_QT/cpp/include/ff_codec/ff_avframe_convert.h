// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===


#ifndef __AVFRAME_CONVERT_H_
#define __AVFRAME_CONVERT_H_

extern "C" {
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libavformat/avformat.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include <stdio.h>
#include <unistd.h>
#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
}

//In SoC mode, heap distribution is the same as that of PCIe
/*
 * heap0   tpu
 * heap1   vpp
 * heap2   vpu
*/
#define USEING_MEM_HEAP2 4
#define USEING_MEM_HEAP1 2

typedef struct{
        bm_image *bmImg;
        uint8_t* buf0;
        uint8_t* buf1;
        uint8_t* buf2;
}transcode_t;


int map_bmformat_to_avformat(int bmformat);
int map_avformat_to_bmformat(int avformat);
int bm_image_sizeof_data_type(bm_image *image);
void bmBufferDeviceMemFree(void *opaque, uint8_t *data);
static void bmBufferDeviceMemFree2(void *opaque, uint8_t *data);
int avframe_to_bm_image(bm_handle_t &bm_handle,AVFrame &in, bm_image &out);
int bm_image_to_avframe(bm_handle_t &bm_handle,bm_image *in,AVFrame *out);

#endif
