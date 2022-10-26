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
                              string save_foler) {
  ofstream out;
  for (size_t i = 0; i < batch_imgs.size(); i++) {
    //each pic
    cv::Mat img = batch_imgs[i];
    vector<Rect> rcs;
    vector<float> scores;
    for (size_t j = 0; j < results[i].size(); j++) {
      //each box
      Rect rc;
      rc.x = results[i][j].left;
      rc.y = results[i][j].top;
      rc.width = results[i][j].right - results[i][j].left + 1;
      rc.height = results[i][j].bottom - results[i][j].top + 1;
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
    cout << "  " << argv[0] << " <input_mode> <image_list/video_list> <bmodel path> " << endl;
    exit(1);
  }

  int input_mode = atoi(argv[1]); // 0 image 1 video
  string input_url = argv[2];
  string bmodel_folder_path = argv[3];
  int device_id = 0;

  string save_foler = "results";
  if (0 != access(save_foler.c_str(), 0)) {
    system("mkdir -p results");
  }

  shared_ptr<FaceDetection> face_detection_share_ptr(
                    new FaceDetection(bmodel_folder_path, device_id));
  FaceDetection* face_detection_ptr = face_detection_share_ptr.get();

  struct timeval tpstart, tpend;
  float timeuse = 0.f;
  int batch_size = face_detection_ptr->batch_size();

  if (0 == input_mode) { // image mode
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
        face_detection_ptr->run(batch_imgs, results);
        gettimeofday(&tpend, NULL);
        timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
        timeuse /= 1000;
        cout << "detect used time: " << timeuse << " ms" << endl;
        save_imgs(results, batch_imgs, batch_names, save_foler);
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
        face_detection_ptr->run(batch_imgs, results);
        gettimeofday(&tpend, NULL);
        timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
        timeuse /= 1000;
        cout << "detect used time: " << timeuse << " ms" << endl;
        save_imgs(results, batch_imgs, batch_names, save_foler);
        batch_imgs.clear();
        batch_names.clear();
        gettimeofday(&tpstart, NULL);
      }
    }
  } else { // video mode
    vector <cv::VideoCapture> caps;
    vector <string> cap_srcs;
    char image_path[1024] = {0};
    ifstream fp_img_list(input_url);
    while(fp_img_list.getline(image_path, 1024)) {
      cv::VideoCapture cap(image_path);
      caps.push_back(cap);
      cap_srcs.push_back(image_path);
    }

    if ((int)caps.size() != batch_size) {
      cout << "video num should equal model's batch size" << endl;
      exit(1);
    }

    uint32_t batch_id = 0;
    const uint32_t run_frame_no = 200; 
    uint32_t frame_id = 0;
    while(1) {
      if (frame_id == run_frame_no) {
        break;
      }
      vector<cv::Mat> batch_imgs;
      vector<string> batch_names;
      gettimeofday(&tpstart, NULL);
      for (size_t i = 0; i < caps.size(); i++) {
         if (caps[i].isOpened()) {
           int w = int(caps[i].get(cv::CAP_PROP_FRAME_WIDTH));
           int h = int(caps[i].get(cv::CAP_PROP_FRAME_HEIGHT));
           cv::Mat img;
           caps[i] >> img;
           if (img.rows != h || img.cols != w) {
             break;
           }
           batch_imgs.push_back(img);
           batch_names.push_back(to_string(batch_id) + "_" +
                            to_string(i) + "_video.jpg");
           batch_id++;
         }else{
           cout << "VideoCapture " << i << " "
                   << cap_srcs[i] << " open failed!" << endl;
         }
      }
      if ((int)batch_imgs.size() < batch_size) {
        break;
      }
      std::vector<std::vector<stFaceRect> > results;
      face_detection_ptr->run(batch_imgs, results);
      gettimeofday(&tpend, NULL);
      timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
      timeuse /= 1000;
      cout << "detect used time: " << timeuse << " ms" << endl;
      save_imgs(results, batch_imgs, batch_names, save_foler);
      batch_imgs.clear();
      batch_names.clear();
      frame_id += 1;
    }

  }
  return 0;
}
