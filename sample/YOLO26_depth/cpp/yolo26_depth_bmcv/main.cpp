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
#include "yolo26_depth.hpp"
using json = nlohmann::json;
using namespace std;

static bool is_img(const string& name) {
    size_t dot = name.find_last_of(".");
    if (dot == string::npos) return false;
    string ext = name.substr(dot);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

// 递归列出目录下所有图片（NYU Depth V2 评测集按目录存放，test 下也可能存在二级子目录）。
static void list_imgs_recursive(const string& dir, vector<string>& files) {
    DIR* pDir = opendir(dir.c_str());
    if (pDir == nullptr) return;
    struct dirent* ptr;
    while ((ptr = readdir(pDir)) != 0) {
        string name = ptr->d_name;
        if (name == "." || name == "..") continue;
        string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            list_imgs_recursive(path, files);
        } else if (is_img(name)) {
            files.push_back(path);
        }
    }
    closedir(pDir);
}

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684X/yolo26s_depth_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/test | input path, images direction.}";
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
    Yolo26_depth yolov8(bmodel_file, dev_id);

    // profiling
    TimeStamp depth_ts;
    yolov8.m_ts = &depth_ts;

    // get batch_size
    int batch_size = yolov8.batch_size;

    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);
    if (access("results/depth_maps", 0) != F_OK)
        mkdir("results/depth_maps", S_IRWXU);

    // test images
    if (info.st_mode & S_IFDIR) {
        // get files（递归收集目录下所有图片）
        vector<string> files_vector;
        list_imgs_recursive(input, files_vector);

        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        vector<cv::Mat> depth_maps;
        vector<json> results_json;
        int cn = files_vector.size();
        int id = 0;
        std::sort(files_vector.begin(), files_vector.end());
        for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            string img_file = *iter;
            id++;
            cout << id << "/" << cn << ", img_file: " << img_file << endl;
            depth_ts.save("decode time");
            bm_image bmimg;
            cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
            if (mat.empty()) {
                cout << "Decode error! Skipping current img." << endl;
                continue;
            }
            cv::bmcv::toBMI(mat, &bmimg);
            depth_ts.save("decode time");
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
                auto ret = yolov8.Detect(batch_imgs, depth_maps);
                assert(0 == ret);

                for (int i = 0; i < batch_imgs.size(); i++) {
                    string base_name = batch_names[i].substr(0, batch_names[i].rfind("."));
                    // 保存 raw 深度图（net 分辨率 float32），供精度评测
                    Yolo26_depth::save_depth_npy(depth_maps[i], "results/depth_maps/" + base_name + ".npy");
                    // 保存伪彩色可视化（缩回原图尺寸）
                    cv::Mat vis;
                    cv::resize(depth_maps[i], vis, cv::Size(batch_mats[i].cols, batch_mats[i].rows), 0, 0, cv::INTER_LINEAR);
                    vis = Yolo26_depth::depth_to_vis(vis);
                    cv::imwrite("results/images/" + base_name + ".png", vis);

                    json res_json;
                    res_json["image_name"] = batch_names[i];
                    res_json["depth_map"] = "results/depth_maps/" + base_name + ".npy";
                    results_json.push_back(res_json);

                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
                batch_names.clear();
                depth_maps.clear();
            }
        }

        // save results
        size_t index = bmodel_file.rfind("/");
        string model_name = bmodel_file.substr(index + 1);
        string json_file = "results/" + model_name + "_bmcv_cpp_result.json";
        cout << "================" << endl;
        cout << "result saved in " << json_file << endl;
        ofstream(json_file) << std::setw(4) << results_json;
    } else {
        cv::VideoCapture cap(input, cv::CAP_FFMPEG, dev_id);
        if (!cap.isOpened()) {
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
        auto output_fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        writer.open(output_path, output_fourcc, frameRate, cv::Size(w, h));
        bool end_flag = false;
        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
        vector<cv::Mat> depth_maps;
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
                CV_Assert(0 == yolov8.Detect(batch_imgs, depth_maps));
                for (int i = 0; i < batch_imgs.size(); i++) {
                    static int id = 0;
                    id++;
                    cout << "frame " << id << endl;
                    cv::Mat vis;
                    cv::resize(depth_maps[i], vis, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
                    vis = Yolo26_depth::depth_to_vis(vis);
                    writer.write(vis);
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
                depth_maps.clear();
            }
        }
        writer.release();
        cap.release();
    }

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    depth_ts.calbr_basetime(base_time);
    depth_ts.build_timeline("yolo26_depth test");
    depth_ts.show_summary("yolo26_depth test");
    depth_ts.clear();
    return 0;
}