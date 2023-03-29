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
#include "resnet.hpp"
#include "ff_decode.hpp"
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
    "{bmodel | ../../models/BM1684X/resnet50_fp32_1b.bmodel | bmodel file path}"
    "{dev_id | 0 | TPU device id}"
    "{help | 0 | print help information.}";
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
  if (stat(input.c_str(), &info) != 0) {
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
  RESNET resnet(bm_ctx);
  CV_Assert(0 == resnet.Init());

  // profiling
  TimeStamp resnet_ts;
  TimeStamp *ts = &resnet_ts;
  resnet.enableProfile(ts);

  // get batch_size
  int batch_size = resnet.batch_size();

  // test images 
  vector<bm_image> batch_imgs;
  vector<string> batch_names;
  vector<pair<int, float>> results;
  vector<json> results_json;
  if (info.st_mode & S_IFREG) {
    if (batch_size != 1) {
      cout << "ERROR: batch_size of model is " << batch_size << endl;
      exit(-1);
    }
    ts->save("resnet overall");
    ts->save("read image");
    // decode jpg
    bm_image bmimg;
    picDec(h, input.c_str(), bmimg);
    ts->save("read image");
    batch_imgs.push_back(bmimg);
    // do infer
    CV_Assert(0 == resnet.Classify(batch_imgs, results));
    ts->save("resnet overall");
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
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
    closedir(pDir);

    vector<string>::iterator iter;
    ts->save("resnet overall");
    for (iter = files_vector.begin(); iter != files_vector.end(); iter++) {
      string img_file = *iter;
      ts->save("read image");
      // cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
      bm_image bmimg;
      picDec(h, img_file.c_str(), bmimg);
      ts->save("read image");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);
      if ((int)batch_imgs.size() == batch_size) {
        // predict
        CV_Assert(0 == resnet.Classify(batch_imgs, results));
        // print and save the results
        for (int i = 0; i < batch_size; i++) {
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
        batch_names.clear();
        results.clear();
      }
    }
    if (!batch_imgs.empty()) {
      CV_Assert(0 == resnet.Classify(batch_imgs, results));
      // print and save the results
      for (int i = 0; i < batch_imgs.size(); i++) {
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
      batch_names.clear();
      results.clear();     
    }
    ts->save("resnet overall");

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
  resnet_ts.calbr_basetime(base_time);
  resnet_ts.build_timeline("resnet infer");
  resnet_ts.show_summary("resnet infer");
  resnet_ts.clear();

  cout << endl;
  return 0;
}
