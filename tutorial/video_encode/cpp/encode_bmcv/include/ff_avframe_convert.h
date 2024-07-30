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

#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include <iostream>
#include <string>
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
}

/*for multi version compatible*/
// #include "bmcv_api.h" //After BM1688&CV186AH SDK 1.7, this header file is implement for bmcv api's compatibility.  
#if BMCV_VERSION_MAJOR > 1
typedef bmcv_padding_attr_t bmcv_padding_atrr_t;
/**
 * @name    bm_image_destroy
 * @brief   To solve incompatible issue in a2 sdk.
 * @ingroup bmcv
 *
 * @param [image]        input bm_image
 * @retval BM_SUCCESS    change success.
 * @retval other values  change failed.
 */
static inline bm_status_t bm_image_destroy(bm_image& image){
  return bm_image_destroy(&image);
}
#endif

//In BM1684/BM1684X SoC mode, heap distribution is the same as that of PCIe
/*
 * heap0   tpu
 * heap1   vpp
 * heap2   vpu
*/
//In BM1688/CV186AH SoC mode, there is only 2 heaps
/*
 * heap0   tpu
 * heap1   vpp
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
