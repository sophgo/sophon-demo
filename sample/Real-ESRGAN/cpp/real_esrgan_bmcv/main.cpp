// ===----------------------------------------------------------------------===//

// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.

// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.

// ===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include<vector>
#include "real_esrgan.hpp"

using namespace std;
#define WITH_ENCODE 1
int main(int argc, char *argv[]){
    cout.setf(ios::fixed);
    // get params
    const char *keys="{bmodel | ../../models/BM1684X/real_esrgan_int8_1b.bmodel | bmodel file path}"
      "{dev_id | 0 | TPU device id}"
      "{help | 0 | print help information.}"
      "{input | ../../datasets/coco128 | input path, images direction or video file path}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
      parser.printMessage();
      return 0;
    }
    string bmodel_file = parser.get<string>("bmodel");
    string input = parser.get<string>("input");
    int dev_id = parser.get<int>("dev_id");

    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
      cout << "Cannot find valid model file." << endl;
      exit(1);
    }
  
    if (stat(input.c_str(), &info) != 0){
      cout << "Cannot find input path." << endl;
      exit(1);
    }

    // creat handle
    bm_handle_t handle;
    bm_dev_request(&handle, dev_id);

    // initialize net
    Real_ESRGAN real_esrgan(bmodel_file, dev_id);

    // profiling
    TimeStamp ts;
    real_esrgan.m_ts = &ts;

    // get batch_size
    int batch_size = real_esrgan.batch_size;

    // creat save path
    if (access("results", 0) != F_OK)
      mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
      mkdir("results/images", S_IRWXU);
    
    // test images
    if (info.st_mode & S_IFDIR){
      // get files
      vector<string> files_vector;
      DIR *pDir;
      struct dirent* ptr;
      pDir = opendir(input.c_str());
      while((ptr = readdir(pDir))!=0) {
          if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
              files_vector.push_back(input + "/" + ptr->d_name);
          }
      }
      closedir(pDir);
      std::sort(files_vector.begin(), files_vector.end());

      vector<cv::Mat> batch_mats;
      vector<bm_image> batch_imgs;
      vector<cv::Mat> output_mats;
      vector<string> batch_names;

      int cn = files_vector.size();
      int id = 0;
      for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++){
        string img_file = *iter; 
        id++;
        cout << id << "/" << cn << ", img_file: " << img_file << endl;
        ts.save("decode time");
        bm_image bmimg;
        cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
        if(mat.empty()){
          cout << "Decode error! Skipping current img." << endl;
          continue;
        }
        cv::bmcv::toBMI(mat, &bmimg);
        ts.save("decode time");
        size_t index = img_file.rfind("/");
        string img_name = img_file.substr(index + 1);
        batch_mats.push_back(mat);
        batch_imgs.push_back(bmimg);
        batch_names.push_back(img_name);
        iter++;
        bool end_flag = (iter == files_vector.end());
        iter--;
        if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
            // predict
            CV_Assert(0 == real_esrgan.process(batch_imgs, output_mats));

            for (int i = 0; i < output_mats.size(); i++) {
                // save image as JPEG
                string img_file = "results/images/" + batch_names[i] ;
                cv::imwrite(img_file, output_mats[i]);

                // destroy the original bm_image
                bm_image_destroy(batch_imgs[i]);
            }
            batch_mats.clear();
            batch_imgs.clear();
            output_mats.clear();
            batch_names.clear();
        }
      }
    }else {
        cv::VideoCapture cap(input, cv::CAP_ANY, dev_id);
        if(!cap.isOpened()) {
            std::cout << "open video stream failed!" << std::endl;
            exit(1);
        }
        int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
        int frameRate = cap.get(cv::CAP_PROP_FPS);
        std::cout << "Frame num: " << frame_num << std::endl;
        std::cout << "resolution of input stream: " << h << ", " << w << std::endl;
        cv::VideoWriter writer;
        std::string output_path = "results/output.mp4";
        auto output_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); //use "H","2","6","4" for h264, "H","V","C","1" for h265.
        bool end_flag = false;
        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
        vector<cv::Mat> output_mats;
        int cnt = 0;
        while (!end_flag) {
            cv::Mat mat;
            cap >> mat;
            if (mat.empty()) {
                end_flag = true;
            } else {
                batch_mats.push_back(mat);
                bm_image bmimg;
                cv::bmcv::toBMI(mat, &bmimg);
                batch_imgs.push_back(bmimg);
            }
            if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
                // predict
                CV_Assert(0 == real_esrgan.process(batch_imgs, output_mats));
                for (int i = 0; i < batch_size; i++) {
                    std::cout<<"write frame "<<cnt++<<std::endl;
                    string img_file = "results/images/" + std::to_string(cnt) + ".jpg" ;
                    cv::imwrite(img_file, output_mats[i]);
                    if(!writer.isOpened()){
                        writer.open(output_path, output_fourcc, frameRate, cv::Size(output_mats[i].cols, output_mats[i].rows));
                    }
                    if(output_mats[i].type() == CV_8UC1){
                        cv::Mat color;
                        cv::cvtColor(output_mats[i], color, cv::COLOR_GRAY2BGR);
                        writer.write(color);
                    }else{
                        writer.write(output_mats[i]);
                    }
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
                output_mats.clear();
            }
        }
        writer.release();
        cap.release();
    }
  
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("real_esrgan test");
  ts.show_summary("real_esrgan test");
  ts.clear();

  return 0;
}
