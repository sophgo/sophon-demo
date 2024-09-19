// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef ADAPT_H_
#define ADAPT_H_

#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include "bmruntime_cpp.h"
#include "bmruntime_interface.h"

// adapt for 1688
#if BMCV_VERSION_MAJOR > 1
static inline bm_status_t bm_image_destroy(bm_image& image){
  return bm_image_destroy(&image);
}
typedef bmcv_padding_attr_t bmcv_padding_atrr_t;
static inline bm_status_t bm_image_dettach_contiguous_mem(int image_num, bm_image *images){
  return bm_image_detach_contiguous_mem(image_num, images);
}
#endif

#endif