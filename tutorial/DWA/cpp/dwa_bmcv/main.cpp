//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <bmcv_api_ext.h>
#include <bmlib_runtime.h>
#include <chrono>
#include "dwa.hpp"
bool file_exist(const std::string& path){
    struct stat info;
    return (stat(path.c_str(), &info) == 0);
}
int main(int argc, char* argv[]) {
    // 解析命令行参数
    std::cout.setf(std::ios::fixed);
    const char* keys =
        "{input_grid_path | /data/images/left/LL.dat  | input grid path}"
        "{input_image_path | /data/images/left/left.jpg  | input image path}"
        "{if_resize | true | if resize}"
        "{resize_h | 1080 | resize height}"
        "{resize_w | 1920 | resize width}"
        "{debug | true | debug}"
        ;
    cv::CommandLineParser parser(argc, argv, keys);
    std::string input_grid_path = parser.get<std::string>("input_grid_path");
    std::string input_image_path = parser.get<std::string>("input_image_path");
    bool if_resize = parser.get<bool>("if_resize");
    int resize_h = parser.get<int>("resize_h");
    int resize_w = parser.get<int>("resize_w");
    bool debug = parser.get<bool>("debug");
    // 检查输入路径
    if(!file_exist(input_grid_path)){
        std::cout << "Cannot find input grid path." << std::endl;
        exit(1);
    }
    if(!file_exist(input_image_path)){
        std::cout << "Cannot find input image path." << std::endl;
        exit(1);
    }

    // 读取图片
    cv::Mat image_mat = cv::imread(input_image_path, cv::IMREAD_COLOR, 0);
    if(image_mat.empty()){
        std::cout << "Decode error! Skipping current img." << std::endl;
    }
    std::cout << "read pics" << std::endl;
    // resize
    if(if_resize){
        cv::resize(image_mat, image_mat, cv::Size(resize_w, resize_h));
    }
    // 转换为bmcv格式
    bm_image input_image;
    cv::bmcv::toBMI(image_mat, &input_image);
    std::cout << "input_image.width: " << input_image.width << std::endl;

    // 初始化dwa
    Dwa dwa;
    dwa.init(input_grid_path, input_image.width, input_image.height, FORMAT_GRAY, debug); //默认设置灰度图
    
    dwa.process_image(input_image);

    // 释放资源
    int ret = bm_image_destroy(&input_image); // input_image 挂的是cvmat的指针，这里还需要destroy, 具体参照多媒体文档的说法需要
    if(ret != BM_SUCCESS){
        std::cout << "bm_image_destroy input_image failed" << std::endl;
        return ret;
    }
    return 0;
}
