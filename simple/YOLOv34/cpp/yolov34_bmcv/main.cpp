/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#include <boost/filesystem.hpp>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include "yolov3.hpp"
#include "utils.hpp"

namespace fs = boost::filesystem;
using namespace std;
using time_stamp_t = time_point<steady_clock, microseconds>;

static void detect(YOLO &net, vector<cv::Mat>& images,
                                      vector<string> names, TimeStamp *ts) {
  ts->save("detection overall");
  ts->save("stage 1: pre-process");
  net.preForward(images);
  ts->save("stage 1: pre-process");
  ts->save("stage 2: detection  ");
  net.forward();
  ts->save("stage 2: detection  ");
  ts->save("stage 3:post-process");
  vector<vector<yolov3_DetectRect>> dets = net.postForward();
  ts->save("stage 3:post-process");
  ts->save("detection overall");

  string save_folder = "result_imgs";
  if (!fs::exists(save_folder)) {
    fs::create_directory(save_folder);
  }

  for (size_t i = 0; i < images.size(); i++) {
    for (size_t j = 0; j < dets[i].size(); j++) {
      int x_min = dets[i][j].left;
      int x_max = dets[i][j].right;
      int y_min = dets[i][j].top;
      int y_max = dets[i][j].bot;

      std::cout << "Category: " << dets[i][j].category
        << " Score: " << dets[i][j].score << " : " << x_min <<
        "," << y_min << "," << x_max << "," << y_max << std::endl;

      cv::Rect rc;
      rc.x = x_min;
      rc.y = y_min;;
      rc.width = x_max - x_min;
      rc.height = y_max - y_min;
      cv::rectangle(images[i], rc, cv::Scalar(255, 0, 0), 2, 1, 0);
    }
    cv::imwrite(save_folder + "/" + names[i], images[i]);
  }
}

int main(int argc, char **argv) {
  cout.setf(ios::fixed);

  if (argc < 8) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image list> <cfg file> <bmodel file> <test count> <device id> <conf thresh> <nms thresh>" << endl;
    cout << "  " << argv[0] << " video <video list> <cfg file> <bmodel file> <test count> <device id> <conf thresh> <nms thresh>" << endl;
    exit(1);
  }

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0) {
    is_video = true;
  } else if (strcmp(argv[1], "image") == 0) {
    is_video = false;
  } else {
    cout << "Wrong input type, neither image nor video." << endl;
    exit(1);
  }

  string image_list = argv[2];
  if (!is_video && !fs::exists(image_list)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }

  string cfg_file = argv[3];
  if (!fs::exists(cfg_file)) {
    cout << "Cannot find config file." << endl;
    exit(1);
  }

  string bmodel_file = argv[4];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  uint32_t test_loop;
  test_loop = stoull(string(argv[5]), nullptr, 0);
  if (test_loop < 1 && is_video) {
    cout << "test loop must large 0." << endl;
    exit(1);
  }

  // set device id
  std::string dev_str = argv[6];
  std::stringstream checkdevid(dev_str);
  double t;
  if (!(checkdevid >> t)) {
    std::cout << "Is not a valid dev ID: " << dev_str << std::endl;
    exit(1);
  }
  int dev_id = std::stoi(dev_str);
  std::cout << "set device id: " << dev_id << std::endl;
 
  int max_dev_id = 0;
  bm_dev_getcount(&max_dev_id);
  if (dev_id >= max_dev_id) {
        std::cout << "ERROR: Input device id="<< dev_id
        << " exceeds the maximum number " << max_dev_id << std::endl;
        exit(-1);
  }

  float conf_thresh = std::stof(argv[7]);
  float nms_thresh = std::stof(argv[8]);
  std::cout << "confidence threshold:" <<  conf_thresh << ", nms threshold: " << nms_thresh << std::endl;

  YOLO net(cfg_file, bmodel_file, dev_id, conf_thresh, nms_thresh);
  int batch_size = net.getBatchSize();
  TimeStamp ts;
  net.enableProfile(&ts);
  char image_path[1024] = {0};
  ifstream fp_img_list(image_list);
  if (!is_video) {
    vector<cv::Mat> batch_imgs;
    vector<string> batch_names;
    while(fp_img_list.getline(image_path, 1024)) {
      ts.save("decode overall");
      ts.save("stage 0: decode");
      cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR, dev_id);
      ts.save("stage 0: decode");
      if (img.empty()) {
         cout << "read image error!" << endl;
         exit(1);
      }
      ts.save("decode overall");
      fs::path fs_path(image_path);
      string img_name = fs_path.filename().string();
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if (static_cast<int>(batch_imgs.size()) == batch_size) {
        detect(net, batch_imgs, batch_names, &ts);
        batch_imgs.clear();
        batch_names.clear();
      }
    }
  } else {
    vector <cv::VideoCapture> caps;
    vector <string> cap_srcs;
    while(fp_img_list.getline(image_path, 1024)) {
      cv::VideoCapture cap(image_path, cv::CAP_ANY, dev_id);
      cap.set(cv::CAP_PROP_OUTPUT_YUV, 1);
      caps.push_back(cap);
      cap_srcs.push_back(image_path);
    }

    if ((int)caps.size() != batch_size) {
      cout << "video num should equal model's batch size" << endl;
      exit(1);
    }

    uint32_t batch_id = 0;
    uint32_t run_frame_no = test_loop;
    uint32_t frame_id = 0;
    while(1) {
      if (frame_id == run_frame_no) {
        break;
      }
      vector<cv::Mat> batch_imgs;
      vector<string> batch_names;
      ts.save("decode overall");
      ts.save("stage 0: decode");
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
      ts.save("stage 0: decode");
      ts.save("decode overall");
      detect(net, batch_imgs, batch_names, &ts);
      batch_imgs.clear();
      batch_names.clear();
      frame_id += 1;
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("yolo detect");
  ts.show_summary("detect ");
  ts.clear();

  std::cout << std::endl;

  return 0;
}
