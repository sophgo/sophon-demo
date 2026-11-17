//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include "json.hpp"
#include "dinov2.hpp"
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
using json = nlohmann::json;
using namespace std;

int main(int argc, char *argv[]) {
  cout.setf(ios::fixed);
  // get params
  const char *keys="{input | ../../datasets/imagenet_val_1k/img | input path, image file path}"
    "{bmodel | ../../models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel | bmodel file path}"
    "{head | ../../datasets/dinov2_vitb14_reg4_linear_head.bin | linear head bin file path}"
    "{dev_id | 0 | TPU device id}"
    "{help | 0 | print help information.}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  string head_file = parser.get<string>("head");
  int dev_id = parser.get<int>("dev_id");

  // check params
  struct stat info;
  if (stat(bmodel_file.c_str(), &info) != 0) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0) {
    cout << "Cannot find input path." << endl;
    exit(1);
  }
  // head 校验使用独立 stat，避免覆盖 input 的 S_IFDIR/S_IFREG 信息
  {
    struct stat head_info;
    if (stat(head_file.c_str(), &head_info) != 0) {
      cout << "Cannot find valid linear head file." << endl;
      exit(1);
    }
  }

  // create net（构造里完成设备句柄申请、bmodel 加载与网络信息解析）
  DINOV2 dinov2(bmodel_file, head_file, dev_id);
  cout << "set device id: "  << dev_id << endl;

  // profiling
  TimeStamp dinov2_ts;
  TimeStamp *ts = &dinov2_ts;
  dinov2.enableProfile(ts);

  // get batch_size
  int batch_size = dinov2.batch_size();

  // test images
  vector<bm_image> batch_imgs;
  vector<cv::Mat> batch_mats;  // 与 batch_imgs 一一对应，保证 bm_image 引用的内存不被提前释放
  vector<string> batch_names;
  vector<pair<int, float>> results;
  vector<json> results_json;
  if (info.st_mode & S_IFREG) {
    if (batch_size != 1) {
      cout << "ERROR: batch_size of model is " << batch_size << endl;
      exit(-1);
    }
    ts->save("dinov2 overall");
    ts->save("decode time");
    // decode jpg: 用 sophon-opencv 在设备侧解码（JPU），再转成 bm_image
    cv::Mat mat = cv::imread(input, cv::IMREAD_COLOR, dev_id);
    if (mat.empty()) {
      cout << "Decode error!" << endl;
      exit(-1);
    }
    bm_image bmimg;
    cv::bmcv::toBMI(mat, &bmimg);
    ts->save("decode time");
    batch_imgs.push_back(bmimg);
    // do infer（mat 需在 Classify 期间保持有效，故不提前释放）
    CV_Assert(0 == dinov2.Classify(batch_imgs, results));
    ts->save("dinov2 overall");
    bm_image_destroy(batch_imgs[0]);
    // print the results
    cout << input << " pred: " << results[0].first << ", score:" << results[0].second << endl;
  }
  else if (info.st_mode & S_IFDIR) {
    // get files
    vector<string> files_vector;
    DIR *pDir;
    struct dirent* ptr;
    pDir = opendir(input.c_str());
    if (pDir == nullptr) {
        std::cerr << "错误: 无法打开目录: " << input << std::endl;
        return -1;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
    closedir(pDir);

    vector<string>::iterator iter;
    ts->save("dinov2 overall");
    for (iter = files_vector.begin(); iter != files_vector.end(); iter++) {
      string img_file = *iter;
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      ts->save("decode time");
      // decode jpg: sophon-opencv 设备侧解码（JPU），再转成 bm_image
      cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
      if (mat.empty()) {
        std::cerr << "WARNING: skip undecodable image " << img_name << std::endl;
        continue;
      }
      bm_image bmimg;
      cv::bmcv::toBMI(mat, &bmimg);
      ts->save("decode time");
      batch_mats.push_back(mat);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);
      if ((int)batch_imgs.size() == batch_size) {
        // predict
        CV_Assert(0 == dinov2.Classify(batch_imgs, results));
        // print and save the results
        for (int i = 0; i < (int)batch_imgs.size(); i++) {
          img_name = batch_names[i];
          cout << img_name << " pred: " << results[i].first << ", score:" << results[i].second << endl;
          json res_json;
          res_json["filename"] = img_name;
          res_json["prediction"] = results[i].first;
          res_json["score"] = results[i].second;
          results_json.push_back(res_json);
          bm_image_destroy(batch_imgs[i]);
        }
        batch_imgs.clear();
        batch_mats.clear();  // bm_image 已销毁，可以释放其引用的 Mat
        batch_names.clear();
        results.clear();
      }
    }
    if (!batch_imgs.empty()) {
      CV_Assert(0 == dinov2.Classify(batch_imgs, results));
      for (int i = 0; i < (int)batch_imgs.size(); i++) {
        string img_name = batch_names[i];
        cout << img_name << " pred: " << results[i].first << ", score:" << results[i].second << endl;
        json res_json;
        res_json["filename"] = img_name;
        res_json["prediction"] = results[i].first;
        res_json["score"] = results[i].second;
        results_json.push_back(res_json);
        bm_image_destroy(batch_imgs[i]);
      }
      batch_imgs.clear();
      batch_mats.clear();  // bm_image 已销毁，可以释放其引用的 Mat
      batch_names.clear();
      results.clear();
    }
    ts->save("dinov2 overall");

    // save the results to the txt file
    if (access("results", 0) != F_OK) {
      mkdir("results", S_IRWXU);
    }
    size_t index = input.rfind("/");
    if(index == input.length() - 1){
      input = input.substr(0, input.length() - 1);
      index = input.rfind("/");
    }
    string dataset_name = input.substr(index + 1);
    index = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(index + 1);
    string json_file = "results/" + model_name + "_" + dataset_name + "_bmcv_cpp" + "_result.json";
    cout << "================" << endl;
    cout << "result saved in " << json_file << endl;
    ofstream(json_file) << std::setw(4) << results_json;
  }
  else {
    cout << "Is not a valid path: " << input << endl;
    exit(1);
  }

  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  dinov2_ts.calbr_basetime(base_time);
  dinov2_ts.build_timeline("dinov2 infer");
  dinov2_ts.show_summary("dinov2 infer");
  dinov2_ts.clear();

  cout << endl;
  return 0;
}