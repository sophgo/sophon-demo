//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef BLEND_H
#define BLEND_H
#include <mutex>

#include "bmcv_api_ext.h"
#include "frame.h"
void bm_read_bin(bm_image src, const char* input_name);
void bm_dem_read_bin(bm_handle_t handle, bm_device_mem_t* dmem,
                     const char* input_name, unsigned int size);
#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

class Blend {
 public:
  Blend();
  ~Blend();

  int init(const std::string& json);

  int blend_work(std::shared_ptr<Frame>& l_frame,
                 std::shared_ptr<Frame>& r_frame,
                 std::shared_ptr<Frame>& blendframe);

  const char* CONFIG_INTERNAL_HEIGHT_FILED = "src_h";

  const char* CONFIG_INTERNAL_WGT1_FILED = "wgt1";
  const char* CONFIG_INTERNAL_WGT2_FILED = "wgt2";

  const char* CONFIG_INTERNAL_OVLP_LX_FILED = "ovlp_lx";
  const char* CONFIG_INTERNAL_OVLP_RX_FILED = "ovlp_rx";
  const char* CONFIG_INTERNAL_BD_LX0_FILED = "bd_lx0";
  const char* CONFIG_INTERNAL_BD_RX0_FILED = "bd_rx0";
  const char* CONFIG_INTERNAL_BD_LX1_FILED = "bd_lx1";
  const char* CONFIG_INTERNAL_BD_RX1_FILED = "bd_rx1";

  const char* CONFIG_INTERNAL_WET_MODE_FILED = "wgt_mode";
  const char* CONFIG_INTERNAL_WIDTH_MINUS_DIS = "width_minus";

  bool isDwa = false;
  int dev_id = 0;
  bm_handle_t handle = NULL;

  int subId = 0;
  int input_num = 2;
  int src_h, src_w;
  int dst_h, dst_w;

  int width_minus;
  std::mutex mtx;
  struct stitch_param blend_config;
};

#endif