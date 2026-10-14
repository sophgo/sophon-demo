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
#include "ram.hpp"
#include "opencv2/opencv.hpp"
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
  const char *keys="{input | ../../datasets/test | input path, image directory}"
    "{bmodel | ../../models/BM1684X/ram_fp32_1b.bmodel | bmodel file path}"
    "{dev_id | 0 | TPU device id}"
    "{tag_list | ../../datasets/ram_tag_list.txt | path of tag_list}"
    "{tag_list_chinese | ../../datasets/ram_tag_list_chinese.txt | path of tag_list_chinese}"
    "{tag_list_threshold | ../../datasets/ram_tag_list_threshold.txt | path of tag_list_threshold}"
    "{help | 0 | print help information.}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  int dev_id = parser.get<int>("dev_id");
  string tag_list_file = parser.get<string>("tag_list");
  string tag_list_chinese_file = parser.get<string>("tag_list_chinese");
  string tag_list_threshold_file = parser.get<string>("tag_list_threshold");

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

  // load bmodel
  shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

  // initialize net
  RAM ram(bm_ctx, dev_id);
  CV_Assert(0 == ram.Init(tag_list_file, tag_list_chinese_file,
                          tag_list_threshold_file));

  // profiling
  TimeStamp ram_ts;
  TimeStamp *ts = &ram_ts;
  ram.enableProfile(ts);

  // get batch_size
  int batch_size = ram.batch_size();

  // test images
  vector<cv::Mat> batch_imgs;
  vector<string> batch_names;
  vector<pair<string, string>> results;
  vector<json> results_json;
  if (info.st_mode & S_IFREG) {
    if (batch_size != 1) {
      cout << "ERROR: batch_size of model is " << batch_size << endl;
      exit(-1);
    }
    ts->save("ram overall");
    ts->save("decode time");
    // decode jpg
    cv::Mat img = cv::imread(input);
    ts->save("decode time");
    batch_imgs.push_back(img);
    size_t index = input.rfind("/");
    string img_name = input.substr(index + 1);
    batch_names.push_back(img_name);
    // do infer
    CV_Assert(0 == ram.Tag(batch_imgs, results));
    ts->save("ram overall");
    // print the results
    cout << input << endl;
    cout << results[0].first << endl;
    cout << results[0].second << endl;
  }
  else if (info.st_mode & S_IFDIR) {
    // get files
    vector<string> files_vector;
    DIR *pDir;
    struct dirent* ptr;
    pDir = opendir(input.c_str());
    if (pDir == nullptr) {
        std::cerr << "Error: cannot open directory: " << input << std::endl;
        return -1;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
    closedir(pDir);

    // sort files for deterministic order
    sort(files_vector.begin(), files_vector.end());

    vector<string>::iterator iter;
    ts->save("ram overall");
    for (iter = files_vector.begin(); iter != files_vector.end(); iter++) {
      string img_file = *iter;
      // skip non-image files
      string ext = img_file.substr(img_file.find_last_of('.') + 1);
      transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp")
        continue;
      ts->save("decode time");
      cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
      ts->save("decode time");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if ((int)batch_imgs.size() == batch_size) {
        // predict
        CV_Assert(0 == ram.Tag(batch_imgs, results));
        // print and save the results
        for (int i = 0; i < batch_size; i++) {
          img_name = batch_names[i];
          cout << img_name << endl;
          cout << results[i].first << endl;
          cout << results[i].second << endl;
          json res_json;
          res_json["filename"] = img_name;
          res_json["tags"] = results[i].first;
          res_json["tags_chinese"] = results[i].second;
          results_json.push_back(res_json);
        }
        batch_imgs.clear();
        batch_names.clear();
        results.clear();
      }
    }
    if (!batch_imgs.empty()) {
      CV_Assert(0 == ram.Tag(batch_imgs, results));
      // print and save the results
      for (int i = 0; i < (int)batch_imgs.size(); i++) {
        string img_name = batch_names[i];
        cout << img_name << endl;
        cout << results[i].first << endl;
        cout << results[i].second << endl;
        json res_json;
        res_json["filename"] = img_name;
        res_json["tags"] = results[i].first;
        res_json["tags_chinese"] = results[i].second;
        results_json.push_back(res_json);
      }
      batch_imgs.clear();
      batch_names.clear();
      results.clear();
    }
    ts->save("ram overall");

    // save the results to the json file
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
    string json_file = "results/" + model_name + "_" + dataset_name + "_opencv_cpp" + "_result.json";
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
  ram_ts.calbr_basetime(base_time);
  ram_ts.build_timeline("ram infer");
  ram_ts.show_summary("ram infer");
  ram_ts.clear();

  cout << endl;
  return 0;
}
