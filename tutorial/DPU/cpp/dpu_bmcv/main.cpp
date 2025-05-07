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
#include "dpu.hpp"




int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::fixed);
    
    // 解析命令行参数
    const char* keys =
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{mode | 0 | 0: SGBM, 1: Online(SGBM+FGS)}"
        "{input | ../../KITTI12/kitti12_train194.txt | input path}"
        "{output | results/images | output path}"
        "{debug | 0 | debug mode}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    
    std::string input = parser.get<std::string>("input");
    int dev_id = parser.get<int>("dev_id");
    int mode = parser.get<int>("mode");

    // 检查输入路径
    struct stat info;
    if (stat(input.c_str(), &info) != 0) {
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    // 创建DPU实例
    bool debug = parser.get<bool>("debug");
    DPU dpu;
    if (dpu.init(dev_id, debug) != 0) {
        std::cerr << "Failed to initialize DPU" << std::endl;
        return -1;
    }

    // 创建保存目录
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    // 创建debug 目录
    if (debug) {
        if (access("debug", 0) != F_OK)
            mkdir("debug", S_IRWXU);
    }

    // 读取输入文件
    size_t index = input.rfind("/");
    std::string input_dir = input.substr(0, index);
    std::ifstream file(input);
    std::string line;

    // 时间戳
    TimeStamp dpu_ts;
    dpu.m_ts = &dpu_ts;
    
    if(file.is_open()) {
        while(std::getline(file, line)) {
            std::istringstream iss(line);
            std::string left_path, right_path;
            // 读取左右图像路径
            iss >> left_path >> right_path;
            
            // 构建完整路径 - 确保路径分隔符正确
            std::string full_left_path = left_path;
            std::string full_right_path = right_path;
            
            // 如果路径是相对的，则添加输入目录
            if (left_path[0] != '/') {
                full_left_path = input_dir + '/' + left_path;
            }
            if (right_path[0] != '/') {
                full_right_path = input_dir + '/' + right_path;
            }
            
            // 检查文件是否存在
            if (access(full_left_path.c_str(), F_OK) != 0) {
                std::cerr << "Left image not found: " << full_left_path << std::endl;
                continue;
            }
            if (access(full_right_path.c_str(), F_OK) != 0) {
                std::cerr << "Right image not found: " << full_right_path << std::endl;
                continue;
            }

            std::cout << "Processing, left: " << full_left_path << "; right: " << full_right_path << std::endl;
            
            // sophon-opencv解码图像
            bm_image left_img, right_img;
            cv::Mat left_mat = cv::imread(full_left_path, cv::IMREAD_COLOR, dev_id);
            cv::Mat right_mat = cv::imread(full_right_path, cv::IMREAD_COLOR, dev_id);
            if(left_mat.empty() || right_mat.empty()){
                std::cout << "Decode error! Skipping current img." << std::endl;
                continue;
            }
            
            // 转换为bm_image
            cv::bmcv::toBMI(left_mat, &left_img);
            cv::bmcv::toBMI(right_mat, &right_img);

            // 打印解码后的图像信息
            std::cout << "After decoding:" << std::endl;
            std::cout << "Left image size: " << left_img.width << "x" << left_img.height << std::endl;
            std::cout << "Right image size: " << right_img.width << "x" << right_img.height << std::endl;
    

            // 创建深度图
            bm_image depth_img;

            // 使用对齐后的图像进行处理
            if (dpu.process(left_img, right_img, depth_img, 
                          static_cast<DPUMode>(mode),
                          DPU_SGBM_MUX2,  
                          DPU_ONLINE_MUX0) != 0) {
                std::cerr << "Failed to process images" << std::endl;
                continue; // 跳过当前图片
            }

            // 生成输出文件名
            size_t name_index = left_path.rfind("/");
            std::string img_name = left_path.substr(name_index + 1);
            std::string output_path = "results/images/" + img_name;
            
            // 保存结果
            if (!dpu.save_image(depth_img, output_path)) {
                std::cerr << "Failed to save image" << std::endl;
                continue;
            }
            std::cout << "Saved result to: " << output_path << std::endl;
            
            bm_image_destroy(&left_img);
            bm_image_destroy(&right_img);
            bm_image_destroy(&depth_img);

            // print speed
            time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
            dpu_ts.calbr_basetime(base_time);
            dpu_ts.build_timeline("dpu test");
            dpu_ts.show_summary("dpu test");
            dpu_ts.clear();
        }
    }
    return 0;
} 