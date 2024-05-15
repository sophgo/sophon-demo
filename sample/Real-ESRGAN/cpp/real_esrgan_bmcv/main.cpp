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

// #include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "real_esrgan.hpp"

using namespace std;
#define WITH_ENCODE 1
int main(int argc, char *argv[]){
  cout.setf(ios::fixed);
  // get params
  const char *keys="{bmodel | ../../models/BM1684X/real_esrgan_int8_1b.bmodel | bmodel file path}"
    "{dev_id | 0 | TPU device id}"
    "{help | 0 | print help information.}"
    "{input | /home/yht/yht/demo/sophon-demo/sample/Real-ESRGAN/datasets/coco128 | input path, images direction or video file path}";
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
  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: "  << dev_id << endl;
  bm_handle_t h = handle->handle();

  // load bmodel
  shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

  // initialize net
  Real_ESRGAN real_esrgan(bm_ctx);
  CV_Assert(0 == real_esrgan.Init());

  // profiling
  TimeStamp real_esrgan_ts;
  TimeStamp *ts = &real_esrgan_ts;
  real_esrgan.enableProfile(&real_esrgan_ts);

  // get batch_size
  int batch_size = real_esrgan.batch_size();

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

    vector<bm_image> batch_imgs;
    vector<cv::Mat> output_images;
    vector<string> batch_names;

    int cn = files_vector.size();
    int id = 0;
    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter; 
      id++;
      cout << id << "/" << cn << ", img_file: " << img_file << endl;
      ts->save("decode time");
      bm_image bmimg;
      picDec(h, img_file.c_str(), bmimg);
      ts->save("decode time");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);
      iter++;
      bool end_flag = (iter == files_vector.end());
      iter--;
      if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
          // predict
          CV_Assert(0 == real_esrgan.Detect(batch_imgs, output_images));

          for (int i = 0; i < output_images.size(); i++) {
              // save image as JPEG
              string img_file = "results/images/" + batch_names[i] ;
              cv::imwrite(img_file, output_images[i]);

              // destroy the original bm_image
              bm_image_destroy(batch_imgs[i]);
          }
          batch_imgs.clear();
          output_images.clear();
          batch_names.clear();
      }
    }
  }
  
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  real_esrgan_ts.calbr_basetime(base_time);
  real_esrgan_ts.build_timeline("real_esrgan test");
  real_esrgan_ts.show_summary("real_esrgan test");
  real_esrgan_ts.clear();

  return 0;
}
