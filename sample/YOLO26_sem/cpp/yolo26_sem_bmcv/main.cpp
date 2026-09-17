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
#include "opencv2/opencv.hpp"
#include "bm_wrapper.hpp"
#include "yolo26_sem.hpp"

using namespace std;

static bool is_img(const string& name) {
    size_t dot = name.find_last_of(".");
    if (dot == string::npos) return false;
    string ext = name.substr(dot);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

// 递归列出目录下所有图片（Cityscapes 按 city 分二级子目录存放）。
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
    const char* keys =
        "{bmodel | ../../models/BM1684X/yolo26s_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/test | input path, images direction or video.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    string bmodel_file = parser.get<string>("bmodel");
    string input = parser.get<string>("input");
    int dev_id = parser.get<int>("dev_id");

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
    Yolo26_sem yolo26(bmodel_file, dev_id);

    // profiling
    TimeStamp yolo26_ts;
    yolo26.m_ts = &yolo26_ts;
    int batch_size = yolo26.batch_size;

    // create save path
    if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK) mkdir("results/images", S_IRWXU);
    if (access("results/segmaps", 0) != F_OK) mkdir("results/segmaps", S_IRWXU);

    if (info.st_mode & S_IFDIR) {
        vector<string> files_vector;
        list_imgs_recursive(input, files_vector);
        std::sort(files_vector.begin(), files_vector.end());

        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        int cn = files_vector.size();
        int id = 0;
        for (auto iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            string img_file = *iter;
            id++;
            cout << id << "/" << cn << ", img_file: " << img_file << endl;
            yolo26_ts.save("decode time");
            cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
            if (mat.empty()) {
                cout << "Decode error! Skipping current img." << endl;
                continue;
            }
            bm_image bmimg;
            cv::bmcv::toBMI(mat, &bmimg);
            yolo26_ts.save("decode time");

            size_t index = img_file.rfind("/");
            string img_name = img_file.substr(index + 1);
            batch_mats.push_back(mat);
            batch_imgs.push_back(bmimg);
            batch_names.push_back(img_name);

            iter++;
            bool end_flag = (iter == files_vector.end());
            iter--;
            if ((batch_imgs.size() == (size_t)batch_size || end_flag) && !batch_imgs.empty()) {
                vector<cv::Mat> segmaps;
                CV_Assert(0 == yolo26.Detect(batch_imgs, segmaps));
                for (int i = 0; i < (int)batch_imgs.size(); i++) {
                    cv::Mat blend;
                    yolo26.draw_result(batch_mats[i], segmaps[i], blend);
                    string save_base = "results/images/" + batch_names[i];
                    cv::imwrite(save_base, blend);
                    size_t dot = batch_names[i].rfind(".");
                    string stem = batch_names[i].substr(0, dot);
                    cv::imwrite("results/segmaps/" + stem + ".png", segmaps[i]);
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
                batch_names.clear();
            }
        }
    } else {
        cv::VideoCapture cap(input, cv::CAP_ANY, dev_id);
        if (!cap.isOpened()) {
            cout << "open video stream failed!" << endl;
            exit(1);
        }
        int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int frameRate = cap.get(cv::CAP_PROP_FPS);
        cout << "resolution of input stream: " << h << ", " << w << endl;
        cv::VideoWriter writer;
        std::string output_path = "results/output.mp4";
        auto output_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(output_path, output_fourcc, frameRate, cv::Size(w, h));
        bool end_flag = false;
        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
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
            if ((batch_imgs.size() == (size_t)batch_size || end_flag) && !batch_imgs.empty()) {
                vector<cv::Mat> segmaps;
                CV_Assert(0 == yolo26.Detect(batch_imgs, segmaps));
                for (int i = 0; i < (int)batch_imgs.size(); i++) {
                    static int fid = 0;
                    fid++;
                    cout << fid << ", write frame" << endl;
                    cv::Mat blend;
                    yolo26.draw_result(batch_mats[i], segmaps[i], blend);
                    writer.write(blend);
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
            }
        }
        writer.release();
        cap.release();
    }

    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    yolo26_ts.calbr_basetime(base_time);
    yolo26_ts.build_timeline("yolo26_sem test");
    yolo26_ts.show_summary("yolo26_sem test");
    yolo26_ts.clear();
    return 0;
}