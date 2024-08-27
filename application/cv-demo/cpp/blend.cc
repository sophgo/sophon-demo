//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "blend.h"

#include "json.hpp"

extern void bm_read_bin(bm_image src, const char* input_name);

void bm_dem_read_bin(bm_handle_t handle, bm_device_mem_t* dmem,
                     const char* input_name, unsigned int size) {
  char* input_ptr = (char*)malloc(size);
  FILE* fp_src = fopen(input_name, "rb+");

  if (fread((void*)input_ptr, 1, size, fp_src) < (unsigned int)size) {
    printf("file size is less than %d required bytes\n", size);
  };
  fclose(fp_src);

  if (BM_SUCCESS != bm_malloc_device_byte(handle, dmem, size)) {
    printf("bm_malloc_device_byte failed\n");
  }

  if (BM_SUCCESS != bm_memcpy_s2d(handle, *dmem, input_ptr)) {
    printf("bm_memcpy_s2d failed\n");
  }

  free(input_ptr);
  return;
}

Blend::Blend() {}
Blend::~Blend() {}

int Blend::init(const std::string& json) {
  auto configure2 = nlohmann::json::parse(json, nullptr, false);
  if (!configure2.is_object()) {
    return -1;
  }
  // mFpsProfiler.config("fps_blend:", 100);
  auto configure = configure2["configure"];
  bm_status_t ret = bm_dev_request(&handle, dev_id);

  src_h = configure.find(CONFIG_INTERNAL_HEIGHT_FILED)->get<int>();

  auto wgt1 = configure.find(CONFIG_INTERNAL_WGT1_FILED)->get<std::string>();
  auto wgt2 = configure.find(CONFIG_INTERNAL_WGT2_FILED)->get<std::string>();
  char* wgt_name[2] = {NULL};
  wgt_name[0] = (char*)wgt1.c_str();
  wgt_name[1] = (char*)wgt2.c_str();

  memset(&blend_config, 0, sizeof(blend_config));
  blend_config.ovlap_attr.ovlp_lx[0] =
      configure.find(CONFIG_INTERNAL_OVLP_LX_FILED)->get<int>();
  blend_config.ovlap_attr.ovlp_rx[0] =
      configure.find(CONFIG_INTERNAL_OVLP_RX_FILED)->get<int>();
  blend_config.bd_attr.bd_lx[0] =
      configure.find(CONFIG_INTERNAL_BD_LX0_FILED)
          ->get<int>();  // left img, bd_attr from algo
  blend_config.bd_attr.bd_rx[0] =
      configure.find(CONFIG_INTERNAL_BD_RX0_FILED)->get<int>();
  blend_config.bd_attr.bd_lx[1] =
      configure.find(CONFIG_INTERNAL_BD_LX1_FILED)
          ->get<int>();  // right img, bd_attr from algo
  blend_config.bd_attr.bd_rx[1] =
      configure.find(CONFIG_INTERNAL_BD_RX1_FILED)->get<int>();

  // blend_config.wgt_mode =
  //     (bm_stitch_wgt_mode)configure.find(CONFIG_INTERNAL_BD_RX1_FILED)
  //         ->get<int>();
  blend_config.wgt_mode = BM_STITCH_WGT_YUV_SHARE;
  int wgtwidth = ALIGN(blend_config.ovlap_attr.ovlp_rx[0] -
                           blend_config.ovlap_attr.ovlp_lx[0] + 1,
                       16);
  int wgtheight = src_h;
  int wgt_len = wgtwidth * wgtheight;
  for (int i = 0; i < 2; i++) {
    bm_dem_read_bin(handle, &blend_config.wgt_phy_mem[0][i], wgt_name[i],
                    wgt_len);
  }

  width_minus = blend_config.ovlap_attr.ovlp_rx[0] -
                blend_config.ovlap_attr.ovlp_lx[0] + 1;
  return 1;
}

int Blend::blend_work(std::shared_ptr<Frame>& l_frame,
                      std::shared_ptr<Frame>& r_frame,
                      std::shared_ptr<Frame>& blendframe) {
  {
    auto start = std::chrono::high_resolution_clock::now();

    bool need_convert = (l_frame->mSpDataDwa->image_format != FORMAT_YUV420P ||
                         r_frame->mSpDataDwa->image_format != FORMAT_YUV420P);

    bm_image blend_img[2];
    if (need_convert) {
      bm_image_create(l_frame->mHandle, l_frame->mSpDataDwa->height,
                      l_frame->mSpDataDwa->width, FORMAT_YUV420P,
                      DATA_TYPE_EXT_1N_BYTE, &blend_img[0], NULL);
      bm_image_create(r_frame->mHandle, r_frame->mSpDataDwa->height,
                      r_frame->mSpDataDwa->width, FORMAT_YUV420P,
                      DATA_TYPE_EXT_1N_BYTE, &blend_img[1], NULL);

      bm_image_alloc_dev_mem(blend_img[0], 1);
      bm_image_alloc_dev_mem(blend_img[1], 1);

      bmcv_image_storage_convert(l_frame->mHandle, 1, l_frame->mSpDataDwa.get(),
                                 &blend_img[0]);
      bmcv_image_storage_convert(r_frame->mHandle, 1, r_frame->mSpDataDwa.get(),
                                 &blend_img[1]);
    } else {
      blend_img[0] = *l_frame->mSpDataDwa;
      blend_img[1] = *r_frame->mSpDataDwa;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::shared_ptr<bm_image> blend_image = nullptr;
    blend_image.reset(new bm_image, [](bm_image* p) {
      bm_image_destroy(p);
      delete p;
      p = nullptr;
    });

    bm_status_t ret = bm_image_create(
        handle, l_frame->mSpDataDwa->height,
        ALIGN(l_frame->mSpDataDwa->width + r_frame->mSpDataDwa->width -
                  width_minus,
              32),
        FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, blend_image.get());
    bm_image_alloc_dev_mem(*blend_image, 1);
    {
      // blend
      bmcv_blending(handle, input_num, blend_img, *blend_image, blend_config);
    }
    blendframe->mSpData = blend_image;
    blendframe->mSpDataDwa = blend_image;
    blendframe->mWidth = blend_image->width;
    blendframe->mHeight = blend_image->height;

    if (need_convert) {
      bm_image_destroy(&blend_img[0]);
      bm_image_destroy(&blend_img[1]);
    }
  }

  blendframe->mChannelId = l_frame->mChannelId;
  blendframe->mFrameId = l_frame->mFrameId;
  blendframe->mChannelIdInternal = l_frame->mChannelIdInternal;
  blendframe->mHandle = l_frame->mHandle;

  return 0;
}
