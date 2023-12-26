//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
//  main.cpp
//  NelivaSDK
//
//  Created by Bitmain on 2020/11/5.
//  Copyright © 2020年 AnBaolei. All rights reserved.
//

#include <iostream>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "face_detection.hpp"
#include <dirent.h>
using namespace std;
using namespace cv;

static void save_imgs(const std::vector<std::vector<stFaceRect> >& results,
                              const vector<cv::Mat>& batch_imgs,
                              const vector<string>& batch_names,
                              string save_foler, ofstream& ofs) {
  ofstream out;
  for (size_t i = 0; i < batch_imgs.size(); i++) {
    //each pic
    cv::Mat img = batch_imgs[i];
    vector<Rect> rcs;
    vector<float> scores;
    ofs << batch_names[i] << endl;
    ofs << results[i].size() << endl;
    for (size_t j = 0; j < results[i].size(); j++) {
      //each box
      Rect rc;
      rc.x = results[i][j].left;
      rc.y = results[i][j].top;
      rc.width = results[i][j].right - results[i][j].left + 1;
      rc.height = results[i][j].bottom - results[i][j].top + 1;
      ofs << rc.x << " " << rc.y << " " << rc.width << " " << rc.height << " " << results[i][j].score << endl;
      cv::rectangle(img, rc, cv::Scalar(0, 0, 255), 2, 1, 0);
      rcs.push_back(rc);
      scores.push_back(results[i][j].score);
      for (size_t k = 0; k < 5; k++) {
        cv::circle(img, Point(results[i][j].points_x[k],
                  results[i][j].points_y[k]), 1, cv::Scalar(255, 0, 0), 3);
      }
    }
    // write new pic
    string save_name = save_foler + "/" + batch_names[i];
    imwrite(save_name,img);
    // write txt
    // out.open("bmcv_cpp_result.txt", std::ios::out | std::ios::app); 
    // out << batch_names[i] <<"\n";
    // out << results[i].size()<<"\n";
    // for (size_t z = 0; z < results[i].size(); z++){
    //   out << rcs[z].x << " " << rcs[z].y << " " << rcs[z].width << " " << rcs[z].height << " " << scores[z] << "\n";
    // }
  }
}

int main(int argc, const char * argv[]) {
  if (argc < 3) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <input_mode> <image_list/video_list> <bmodel path> <nms threshold> <conf threshold> " << endl;
    exit(1);
  }

  int input_mode = atoi(argv[1]); // 0 image 1 video
  string input_url = argv[2];
  string bmodel_folder_path = argv[3];
  float nms_threshold = atof(argv[4]);
  float conf_threshold = atof(argv[5]);
  int device_id = 0;
  int split_indx = input_url.size() - 1;
  while (split_indx >= 0 && input_url.at(split_indx) == '/') {
    split_indx--;
  }
  if (split_indx < 0) {
    cout << "Param err: " << input_url << endl;
    exit(1);
  }
  input_url = input_url.substr(0, split_indx + 1);
  string input_filename = input_url.substr(input_url.find_last_of("/")+1);
  input_filename = input_filename.substr(0, input_filename.find_last_of("."));
  split_indx = bmodel_folder_path.size() - 1;
  while (split_indx >= 0 && bmodel_folder_path.at(split_indx) == '/') {
    split_indx--;
  }
  if (split_indx < 0) {
    cout << "Param err: " << bmodel_folder_path << endl;
    exit(1);
  }
  bmodel_folder_path = bmodel_folder_path.substr(0, split_indx + 1);
  string bmodel_filename = bmodel_folder_path.substr(bmodel_folder_path.find_last_of("/")+1);
  bmodel_filename = bmodel_filename.substr(0, bmodel_filename.find_last_of("."));

  string save_foler = "results";
  if (0 != access(save_foler.c_str(), 0)) {
    system("mkdir -p results");
  }

  shared_ptr<FaceDetection> face_detection_share_ptr(
                    new FaceDetection(bmodel_folder_path, device_id, nms_threshold));
  FaceDetection* face_detection_ptr = face_detection_share_ptr.get();
  face_detection_ptr->set_score_threshold(conf_threshold);

  struct timeval tpstart, tpend;
  float timeuse = 0.f;
  int batch_size = face_detection_ptr->batch_size();
  struct timeval infer_tpstart, infer_tpend;
  float infer_timeuse = 0.f;
  int img_num = 0;

  if (0 == input_mode) { // image mode
    ofstream ofs;
    ofs.open(save_foler + "/" + input_filename + "_image_data_" + bmodel_filename + "_result.txt", ios::out);
    vector<cv::Mat> batch_imgs;
    vector<string> batch_names;
    vector<string> files_vector;
    DIR *pDir;
    struct dirent* ptr;
    pDir = opendir(input_url.c_str());
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            files_vector.push_back(input_url + "/" + ptr->d_name);
        }
    }
    closedir(pDir);

    std::sort(files_vector.begin(),files_vector.end());
    vector<string>::iterator iter;
    
    for (iter = files_vector.begin(); iter != files_vector.end(); iter++){
      // string img_file = beg_iter->path().string();
      string img_file = *iter;
      Mat img = imread(img_file, cv::IMREAD_COLOR, 0);
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if (static_cast<int>(batch_imgs.size()) == batch_size) {
        std::vector<std::vector<stFaceRect> > results;
        face_detection_ptr->run(batch_imgs, results, infer_tpstart, infer_tpend);
        gettimeofday(&tpend, NULL);
        timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
        timeuse /= 1000;
        infer_timeuse += (1000000 * (infer_tpend.tv_sec - infer_tpstart.tv_sec) + infer_tpend.tv_usec - infer_tpstart.tv_usec) / 1000;
        img_num++;
        cout << "detect used time: " << timeuse << " ms" << endl;
        save_imgs(results, batch_imgs, batch_names, save_foler, ofs);
        batch_imgs.clear();
        batch_names.clear();
        gettimeofday(&tpstart, NULL);
      } else if((iter+1)==files_vector.end()){
        int rest = batch_size - static_cast<int>(batch_imgs.size());
        for(int i=0;i<rest;i++){
          batch_imgs.push_back(img);
          batch_names.push_back(img_name);
        }
        std::vector<std::vector<stFaceRect> > results;
        face_detection_ptr->run(batch_imgs, results, infer_tpstart, infer_tpend);
        gettimeofday(&tpend, NULL);
        infer_timeuse += (1000000 * (infer_tpend.tv_sec - infer_tpstart.tv_sec) + infer_tpend.tv_usec - infer_tpstart.tv_usec) / 1000;
        img_num++;
        timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
        timeuse /= 1000;
        cout << "detect used time: " << timeuse << " ms" << endl;
        save_imgs(results, batch_imgs, batch_names, save_foler, ofs);
        batch_imgs.clear();
        batch_names.clear();
        gettimeofday(&tpstart, NULL);
      }
    }
    ofs.close();
  } else { // video mode
    ofstream ofs;
    ofs.open(save_foler + "/" + input_filename + "_video_data_" + bmodel_filename + "_result.txt", ios::out);
    cv::VideoCapture cap(input_url);

    uint32_t batch_id = 0;
    uint32_t frame_id = 0;
    if (!cap.isOpened()) {
      cout << "VideoCapture " << input_url << " open failed!" << endl;
      exit(1);
    }
    vector<cv::Mat> batch_imgs;
    vector<string> batch_names;
    bool end_flag = false;
    while(!end_flag) {
      gettimeofday(&tpstart, NULL);
      cv::Mat img;
      cap >> img;
      if (img.empty()) {
        if (batch_id == 0)
          break;
        for (; batch_id < batch_size; batch_id++) {
          batch_imgs.push_back(batch_imgs[0]);
          batch_names.push_back(batch_names[0]);
          frame_id++;
          img_num++;
        }
        end_flag = true;
      }
      else {
        batch_imgs.push_back(img);
        batch_names.push_back(to_string(frame_id) + ".jpg");
        batch_id++;
        frame_id++;
        img_num++;
      }
      if (batch_id < batch_size) {
        continue;
      }
      std::vector<std::vector<stFaceRect> > results;
      face_detection_ptr->run(batch_imgs, results, infer_tpstart, infer_tpend);
      gettimeofday(&tpend, NULL);
      infer_timeuse += (1000000 * (infer_tpend.tv_sec - infer_tpstart.tv_sec) + infer_tpend.tv_usec - infer_tpstart.tv_usec) / 1000;
      timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
      timeuse /= 1000;
      cout << "detect used time: " << timeuse << " ms" << endl;
      save_imgs(results, batch_imgs, batch_names, save_foler, ofs);
      batch_imgs.clear();
      batch_names.clear();
      batch_id = 0;
    }
    ofs.close();

  }
  std::cout << "avg infer time(ms): " << infer_timeuse / img_num << std::endl;
  std::cout << "QPS: " << img_num * 1000 / infer_timeuse << std::endl;
  return 0;
}
