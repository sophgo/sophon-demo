//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dwa.h"

#include <chrono>
#include <iostream>

#include "json.hpp"
#define DEBUG 0
//  const char* Dwa::CONFIG_INTERNAL_IS_GRAY_FILED = "is_gray";
//    const char* Dwa::CONFIG_INTERNAL_IS_RESIZE_FILED = "is_resize";
//    const char* Dwa::CONFIG_INTERNAL_IS_ROT_FILED = "is_rot";
//    const char* Dwa::CONFIG_INTERNAL_DIS_MODE_FILED = "dis_mode";
//    const char* Dwa::CONFIG_INTERNAL_GRID_NAME_FILED = "grid_name";
//    const char* Dwa::CONFIG_INTERNAL_USE_GRIDE_FILED = "use_grid";
//    const char* Dwa::CONFIG_INTERNAL_GRIDE_SIZE_FILED = "grid_size";

//    const char* Dwa::CONFIG_INTERNAL_SRC_H_FILED = "src_h";
//    const char* Dwa::CONFIG_INTERNAL_SRC_W_FILED = "src_w";
//    const char* Dwa::CONFIG_INTERNAL_DST_H_FILED = "dst_h";
//    const char* Dwa::CONFIG_INTERNAL_DST_W_FILED = "dst_w";
//    const char* Dwa::CONFIG_INTERNAL_TMP_H_FILED = "tmp_h";
//    const char* Dwa::CONFIG_INTERNAL_TMP_W_FILED = "tmp_w";
//    const char* Dwa::CONFIG_INTERNAL_DWA_MODE_FILED = "dwa_mode";
#define YUV_8BIT(y, u, v) \
  ((((y) & 0xff) << 16) | (((u) & 0xff) << 8) | ((v) & 0xff))

bm_status_t set_fish_default_param(bmcv_fisheye_attr_s* fisheye_attr) {
  fisheye_attr->bEnable = 1;
  fisheye_attr->bBgColor = 1;
  fisheye_attr->u32BgColor = YUV_8BIT(0, 128, 128);
  fisheye_attr->s32HorOffset = 512;
  fisheye_attr->s32VerOffset = 512;
  fisheye_attr->u32TrapezoidCoef = 0;
  fisheye_attr->s32FanStrength = 0;
  fisheye_attr->enMountMode = BMCV_FISHEYE_DESKTOP_MOUNT;
  fisheye_attr->enUseMode = BMCV_MODE_PANORAMA_360;
  fisheye_attr->enViewMode = BMCV_FISHEYE_VIEW_360_PANORAMA;
  fisheye_attr->u32RegionNum = 1;
  return BM_SUCCESS;
}

bm_status_t set_gdc_default_param(bmcv_gdc_attr* gdc_attr) {
  gdc_attr->bAspect = 0;
  gdc_attr->s32XRatio = 0;
  gdc_attr->s32YRatio = 0;
  gdc_attr->s32XYRatio = 0;
  gdc_attr->s32CenterXOffset = 0;
  gdc_attr->s32CenterYOffset = 0;
  gdc_attr->s32DistortionRatio = 0;
  gdc_attr->grid_info.u.system.system_addr = NULL;
  gdc_attr->grid_info.size = 0;
  return BM_SUCCESS;
}

Dwa::Dwa() {}
Dwa::~Dwa() {}

int Dwa::init(const std::string& json) {
  auto configure2 = nlohmann::json::parse(json, nullptr, false);
  if (!configure2.is_object()) {
    return -1;
  }
  // mFpsProfiler.config("dwa fps:", 100);
  if (!configure2.contains("configure")) {
    return -1;
  }
  auto configure = configure2["configure"];
  if (!configure.contains("is_resize")) {
    // Handle the case where "configure" key is missing
    return -1;
  }
  bool is_gray = configure.at("is_gray").get<bool>();

  bool is_resize2 = configure.at("is_resize").get<bool>();

  is_resize = is_resize2;

  src_fmt = is_gray ? FORMAT_GRAY : FORMAT_YUV420P;

  dst_h = configure["dst_h"];

  dst_w = configure["dst_w"];

  tmp_h = configure["resize_h"];
  tmp_w = configure["resize_w"];
  std::string dwa_mode_str = configure["dwa_mode"];
  dwa_mode = dwa_mode_map[dwa_mode_str];

  bool use_grid = configure["use_grid"];

  if (dwa_mode == DWA_GDC_MODE) {  // 用于04a10 dpu 2560x1440 需要resize
    is_rot = configure["is_rot"];
    if (is_rot == true) {
      rot_mode = BMCV_ROTATION_180;
    }

    if (use_grid) {
      grid_name = configure["grid_name"];
      int grid_size = configure["grid_size"];
      char* buffer = (char*)malloc(grid_size);
      memset(buffer, 0, grid_size);

      FILE* fp = fopen(grid_name.c_str(), "rb");

      fseek(fp, 0, SEEK_END);
      int fileSize = ftell(fp);

      if (grid_size != (unsigned int)fileSize) {
        // IVS_DEBUG("load grid_info file:{0} size is not match.",
        //           grid_name.c_str());
        fclose(fp);
        return -1;
      }

      rewind(fp);
      fread(buffer, grid_size, 1, fp);
      fclose(fp);
      ldc_attr.grid_info.u.system.system_addr = (void*)buffer;
      ldc_attr.grid_info.size = grid_size;
    }

  } else if (dwa_mode ==
             DWA_FISHEYE_MODE) {  // 用于04e10 blend 2240x2240 不需要resize
    fisheye_attr = {0};
    // set_fish_default_param(&fisheye_attr);
    rot_mode = BMCV_ROTATION_180;

    if (configure.contains("dis_mode")) {
      std::string dis_mode_str = configure["dis_mode"];
      // STREAM_CHECK(fisheye_mode_map.count(dis_mode_str) != 0,
      //              "Invalid dis_mode in Config File");
      dis_mode = fisheye_mode_map[dis_mode_str];
    }
    if (use_grid) {
      grid_name = configure["grid_name"];
      int grid_size = configure["grid_size"];
      char* buffer = (char*)malloc(grid_size);
      memset(buffer, 0, grid_size);

      FILE* fp = fopen(grid_name.c_str(), "rb");

      fseek(fp, 0, SEEK_END);
      int fileSize = ftell(fp);

      if (grid_size != (unsigned int)fileSize) {
        // IVS_DEBUG("load grid_info file:{0} size is not match.",
        //           grid_name.c_str());
        fclose(fp);
        return -1;
      }

      rewind(fp);
      fread(buffer, grid_size, 1, fp);
      fclose(fp);
      fisheye_attr.grid_info.u.system.system_addr = (void*)buffer;
      fisheye_attr.grid_info.size = grid_size;
      fisheye_attr.bEnable = true;
    }
  }

  return -1;
}

float Dwa::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h,
                                   bool* pIsAligWidth) {
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w) {
    *pIsAligWidth = true;
    ratio = r_w;
  } else {
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

int Dwa::fisheye_work(std::shared_ptr<Frame>& mFrame) {
  if (mFrame != nullptr) {
    std::shared_ptr<bm_image> fisheye_image = nullptr;
    fisheye_image.reset(new bm_image, [](bm_image* p) {
      bm_image_destroy(p);
      delete p;
      p = nullptr;
    });

    bm_status_t ret =
        bm_image_create(mFrame->mHandle, dst_h, dst_w, src_fmt,
                        DATA_TYPE_EXT_1N_BYTE, fisheye_image.get());
    bm_image_alloc_dev_mem(*fisheye_image, 1);

    std::shared_ptr<bm_image> resized_img = nullptr;
    resized_img.reset(new bm_image, [](bm_image* p) {
      bm_image_destroy(p);
      delete p;
      p = nullptr;
    });
    ret = bm_image_create(mFrame->mHandle, tmp_h, tmp_w, src_fmt,
                          mFrame->mSpData->data_type, resized_img.get(), NULL);
    bm_image_alloc_dev_mem(*resized_img, 1);

    bmcv_rect_t crop_rect{0, 0, (unsigned int)mFrame->mSpData->width,
                          (unsigned int)mFrame->mSpData->height};
    bmcv_padding_attr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty =
        int(tmp_h - (unsigned int)mFrame->mSpData->height) / 2;
    padding_attr.dst_crop_stx =
        int(tmp_w - (unsigned int)mFrame->mSpData->width) / 2;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    padding_attr.dst_crop_h = (unsigned int)mFrame->mSpData->height;
    padding_attr.dst_crop_w = (unsigned int)mFrame->mSpData->width;

    ret = bmcv_image_vpp_convert_padding(mFrame->mHandle, 1, *mFrame->mSpData,
                                         resized_img.get(), &padding_attr,
                                         &crop_rect);
    if (is_rot) {
      std::shared_ptr<bm_image> input_rot = nullptr;
      input_rot.reset(new bm_image, [](bm_image* p) {
        bm_image_destroy(p);
        delete p;
        p = nullptr;
      });
      ret = bm_image_create(mFrame->mHandle, tmp_h, tmp_w, src_fmt,
                            mFrame->mSpData->data_type, input_rot.get(), NULL);
      bm_image_alloc_dev_mem(*input_rot, 1);
      bmcv_dwa_rot(mFrame->mHandle, *resized_img, *input_rot, rot_mode);
      resized_img = input_rot;
    }

    bmcv_dwa_fisheye(mFrame->mHandle, *resized_img, *fisheye_image,
                     fisheye_attr);

    mFrame->mSpDataDwa = fisheye_image;  // 是否需要dwa
  }
  return 0;
}
int Dwa::dwa_gdc_work(std::shared_ptr<Frame>& mFrame) {
  if (mFrame != nullptr) {
    // resize
    if (is_resize == true) {
      // resize 2560x1440 -->1920x1090
      std::shared_ptr<bm_image> resized_img = nullptr;
      resized_img.reset(new bm_image, [](bm_image* p) {
        bm_image_destroy(p);
        delete p;
        p = nullptr;
      });

      bm_image image_aligned;
      bool need_copy = (unsigned int)mFrame->mSpData->width & (64 - 1);
      if (need_copy) {
        int stride1[3], stride2[3];
        bm_image_get_stride(*mFrame->mSpData, stride1);
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);
        bm_image_create(mFrame->mHandle, (unsigned int)mFrame->mSpData->height,
                        (unsigned int)mFrame->mSpData->width, src_fmt,
                        DATA_TYPE_EXT_1N_BYTE, &image_aligned, stride2);

        bm_image_alloc_dev_mem(image_aligned, 1);
        bmcv_copy_to_atrr_t copyToAttr;
        memset(&copyToAttr, 0, sizeof(copyToAttr));
        copyToAttr.start_x = 0;
        copyToAttr.start_y = 0;
        copyToAttr.if_padding = 1;
        bmcv_image_copy_to(mFrame->mHandle, copyToAttr, *mFrame->mSpData,
                           image_aligned);
      } else {
        image_aligned = *mFrame->mSpData;
      }

      bool isAlignWidth = false;
      float ratio = get_aspect_scaled_ratio(
          (unsigned int)mFrame->mSpData->width,
          (unsigned int)mFrame->mSpData->height, dst_w, dst_h, &isAlignWidth);

      bmcv_rect_t crop_rect{0, 0, (unsigned int)mFrame->mSpData->width,
                            (unsigned int)mFrame->mSpData->height};
      bmcv_padding_attr_t padding_attr;
      memset(&padding_attr, 0, sizeof(padding_attr));
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = 0;
      padding_attr.padding_b = 114;
      padding_attr.padding_g = 114;
      padding_attr.padding_r = 114;
      padding_attr.if_memset = 1;
      padding_attr.dst_crop_h = dst_h;
      padding_attr.dst_crop_w = ((unsigned int)mFrame->mSpData->width * ratio);

      int tx1 = (int)((dst_w - padding_attr.dst_crop_w) / 2);
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = tx1;

      int aligned_net_w = FFALIGN(dst_w, 64);
      int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
      bm_image_create(mFrame->mHandle, dst_h, dst_w, src_fmt,
                      DATA_TYPE_EXT_1N_BYTE, resized_img.get(), strides);

      bm_image_alloc_dev_mem(*resized_img, 2);

      bm_status_t ret = bmcv_image_vpp_convert_padding(
          mFrame->mHandle, 1, *mFrame->mSpData, resized_img.get(),
          &padding_attr, &crop_rect);
      assert(BM_SUCCESS == ret);

      if (is_rot == true) {
        std::shared_ptr<bm_image> input_rot = nullptr;
        input_rot.reset(new bm_image, [](bm_image* p) {
          bm_image_destroy(p);
          delete p;
          p = nullptr;
        });

        bm_image_create(mFrame->mHandle, resized_img->height,
                        resized_img->width, src_fmt, DATA_TYPE_EXT_1N_BYTE,
                        input_rot.get(), NULL);
        bm_image_alloc_dev_mem(*input_rot, 2);

        bmcv_dwa_rot(mFrame->mHandle, *resized_img, *input_rot, rot_mode);
        resized_img = input_rot;
      }

      // dwa doing
      std::shared_ptr<bm_image> dwa_image = nullptr;
      dwa_image.reset(new bm_image, [](bm_image* p) {
        bm_image_destroy(p);
        delete p;
        p = nullptr;
      });

      ret = bm_image_create(mFrame->mHandle, resized_img->height,
                            resized_img->width, src_fmt, DATA_TYPE_EXT_1N_BYTE,
                            dwa_image.get());
      bm_image_alloc_dev_mem(*dwa_image, 2);

      { bmcv_dwa_gdc(mFrame->mHandle, *resized_img, *dwa_image, ldc_attr); }

      mFrame->mSpData = resized_img;
      mFrame->mSpDataDwa = dwa_image;
      mFrame->mWidth = resized_img->width;
      mFrame->mHeight = resized_img->height;

      if (need_copy) bm_image_destroy(&image_aligned);

    } else {  // dont need resize
      std::shared_ptr<bm_image> dwa_image = nullptr;
      dwa_image.reset(new bm_image, [](bm_image* p) {
        bm_image_destroy(p);
        delete p;
        p = nullptr;
      });
      bm_status_t ret = bm_image_create(
          mFrame->mHandle, mFrame->mSpData->height, mFrame->mSpData->width,
          src_fmt, DATA_TYPE_EXT_1N_BYTE, dwa_image.get());
      bm_image_alloc_dev_mem(*dwa_image, 1);

      bm_image input;
      ret = bm_image_create(mFrame->mHandle, mFrame->mSpData->height,
                            mFrame->mSpData->width, src_fmt,
                            mFrame->mSpData->data_type, &input, NULL);
      bm_image_alloc_dev_mem(input, 1);
      ret = bmcv_image_storage_convert(mFrame->mHandle, 1,
                                       mFrame->mSpData.get(), &input);

      bmcv_dwa_gdc(mFrame->mHandle, input, *dwa_image, ldc_attr);

      mFrame->mSpDataDwa = dwa_image;  // dwa

      bm_image_destroy(&input);
    }
  }

  return 0;
}
int DWA_PIPE::init(std::vector<std::string>& dwa_path) {
  dwas.resize(dwa_path.size());
  output_frames = std::make_shared<DatePipe>();

  output_frames->frames.resize(dwa_path.size());

  int id = 0;
  for (int i = 0; i < dwa_path.size(); i++) {
    dwas[i].init(get_json(dwa_path[i]));
  }

  // std::vector<std::thread> threads;
  for (int i = 0; i < dwa_path.size(); i++) {
    threads.emplace_back(&DWA_PIPE::start, this, i);
  }
  return 1;
}

void DWA_PIPE::start(int dwa_id) {
  while (true) {
    std::shared_ptr<Frame> mframe;

    {
      while (input_frames->frames[dwa_id].empty());
      std::unique_lock<std::mutex> lock(*input_queue_lock);
      mframe = std::move(input_frames->frames[dwa_id].front());
      input_frames->frames[dwa_id].pop();
    }
    auto start = std::chrono::high_resolution_clock::now();
    if (dwas[dwa_id].dwa_mode != DWA_GDC_MODE)
      dwas[dwa_id].fisheye_work(mframe);
    else
      dwas[dwa_id].dwa_gdc_work(mframe);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "dwa程序执行时间：" << duration.count() << " ms" << std::endl;
    {
      while (output_frames->frames[dwa_id].size() == 5) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100));  // 可以添加短暂延迟以避免忙等待
      }

      std::unique_lock<std::mutex> lock(*output_queue_lock);
      if (mframe->mEndOfStream) {
        output_frames->frames[dwa_id].push(std::move(mframe));
        break;
      }
      output_frames->frames[dwa_id].push(std::move(mframe));
      std::cout << "dwa_size" << output_frames->frames[dwa_id].size()
                << std::endl;
    }
  }
  return;
}