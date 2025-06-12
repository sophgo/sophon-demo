//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef DWA_H
#define DWA_H

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <bmcv_api_ext.h>
#include <bmlib_runtime.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <sys/stat.h>

// 如果FFALIGN已经定义，先取消定义
#ifdef FFALIGN
#undef FFALIGN
#endif
#define FFALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))


class Dwa{
    public:
        Dwa();
        ~Dwa();
        int init(std::string grid_path, int input_width, int input_height, bm_image_format_ext input_dwa_fmt, bool debug);
        int process_image(bm_image input_image);
        int release();
    
    private:
        bm_handle_t handle;
        int ret;
        bool debug;
        // 读取grid文件
        bmcv_gdc_attr ldc_attr;
        bool init_gdc_attr();
        char* grid_buffer;
        size_t grid_size;

        bm_image input_image; // 输入图像
        bm_image converted_image; // 转换后的图像
        bm_image dwa_image; // DWA处理后的图像

};

#endif // DWA_H