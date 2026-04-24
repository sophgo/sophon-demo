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
#include "json.hpp"
#include "yolov8_cls.hpp"
using json = nlohmann::json;
using namespace std;

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684X/yolov8s_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/imagenet_val_1k/img | input path, images direction.}";
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

    // initialize net
    YoloV8_cls yolov8(bmodel_file,dev_id);

    // profiling
    TimeStamp yolov8_ts;
    yolov8.m_ts = &yolov8_ts;

    // get batch_size
    int batch_size = yolov8.batch_size;

    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);

    // test images
    if (info.st_mode & S_IFDIR) {
        // get files
        vector<string> files_vector;
        DIR* pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        if (pDir == nullptr) {
            std::cerr << "错误: 无法打开目录: " << input << std::endl;
            return -1;
        }
        while ((ptr = readdir(pDir)) != 0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);

        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        vector<pair<int, float>> results;
        vector<json> results_json;
        int cn = files_vector.size();
        int id = 0;
        std::sort(files_vector.begin(), files_vector.end());
        for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            string img_file = *iter;
            id++;
            cout << id << "/" << cn << ", img_file: " << img_file << endl;
            yolov8_ts.save("decode time");
            bm_image bmimg;
            cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
            if(mat.empty()){
                cout << "Decode error! Skipping current img." << endl;
                continue;
            }
            cv::bmcv::toBMI(mat, &bmimg);
            yolov8_ts.save("decode time");
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
                auto ret = yolov8.Classfy(batch_imgs, results);
                assert(0 == ret);

                for (int i = 0; i < batch_imgs.size(); i++) {
                    img_name = batch_names[i];
                    cout << img_name << " pred: " << results[i].first << ", score:" << results[i].second << endl;
                    json res_json;
                    res_json["filename"] = img_name;
                    res_json["prediction"] = results[i].first;
                    res_json["score"] = results[i].second;
                    results_json.push_back(res_json);
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
                batch_names.clear();
                results.clear();
            }
        }

        // save results
        size_t index = input.rfind("/");
        if (index == input.length() - 1) {
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
    } else {
        std::cout << "Open input failed! Only support image dir now." << std::endl;
    }

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    yolov8_ts.calbr_basetime(base_time);
    yolov8_ts.build_timeline("yolov8 test");
    yolov8_ts.show_summary("yolov8 test");
    yolov8_ts.clear();
    return 0;
}