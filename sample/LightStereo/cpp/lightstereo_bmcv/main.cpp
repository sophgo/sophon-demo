//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "lightstereo.hpp"

// #define DEBUG 1

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::fixed);
    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684X/LightStereo-S-SceneFlow_fp16_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/KITTI12/kitti12_train194.txt | input path}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string input = parser.get<std::string>("input");
    int dev_id = parser.get<int>("dev_id");

    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }

    if (stat(input.c_str(), &info) != 0) {
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    // initialize net
    LightStereo lightstereo(bmodel_file, dev_id);

    // profiling
    TimeStamp ts;
    lightstereo.m_ts = &ts;

    // get batch_size
    int batch_size = lightstereo.batch_size;

    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    size_t index = input.rfind("/");
    std::string input_dir = input.substr(0, index);
    std::ifstream file(input);
    std::string line;
    if(file.is_open()){
        std::vector<std::string> left_img_paths;
        std::vector<std::string> right_img_paths;
        while(std::getline(file, line)){
            std::istringstream iss(line);
            std::string left_path, right_path;
            iss >> left_path >> right_path;
            left_path = input_dir + '/' + left_path;
            right_path = input_dir + '/' + right_path;
            left_img_paths.push_back(left_path);
            right_img_paths.push_back(right_path);
        }
        
        std::vector<cv::Mat> batch_left_mats, batch_right_mats;
        std::vector<bm_image> batch_left_imgs, batch_right_imgs;
        std::vector<std::string> batch_names;
        for(int i = 0; i < left_img_paths.size(); i++){

            std::cout<<"Processing, left: "<<left_img_paths[i]<<"; right:"<<right_img_paths[i]<<std::endl;

            size_t index = left_img_paths[i].rfind("/");
            std::string img_name = left_img_paths[i].substr(index + 1);
            ts.save("decode time");
            cv::Mat left_mat = cv::imread(left_img_paths[i], cv::IMREAD_COLOR, dev_id);
            cv::Mat right_mat = cv::imread(right_img_paths[i], cv::IMREAD_COLOR, dev_id);
            if(left_mat.empty() || right_mat.empty()){
                std::cout << "Decode error! Skipping current img." << std::endl;
                continue;
            }
            ts.save("decode time");
            bm_image left_bmimg, right_bmimg;
            cv::bmcv::toBMI(left_mat, &left_bmimg);
            cv::bmcv::toBMI(right_mat, &right_bmimg);
            batch_left_mats.push_back(left_mat);
            batch_right_mats.push_back(right_mat);
            batch_left_imgs.push_back(left_bmimg);
            batch_right_imgs.push_back(right_bmimg);
            batch_names.push_back(img_name);
            if ((batch_names.size() == batch_size || i == left_img_paths.size() - 1) && !batch_names.empty()){
                std::vector<cv::Mat> disp_images = lightstereo.process(batch_left_imgs, batch_right_imgs);
                for(int j = 0; j < disp_images.size(); j++){
                    cv::imwrite("results/images/"+batch_names[j], disp_images[j]);
                }
                batch_left_mats.clear();
                batch_right_mats.clear();
                batch_left_imgs.clear();
                batch_right_imgs.clear();
                batch_names.clear();
            }
            
        }
    }

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts.calbr_basetime(base_time);
    ts.build_timeline("superpoint test");
    ts.show_summary("superpoint test");
    ts.clear();
    return 0;
}