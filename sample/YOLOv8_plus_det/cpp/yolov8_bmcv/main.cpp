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
#include "yolov8_det.hpp"
using json = nlohmann::json;
using namespace std;

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684X/yolov8s_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{conf_thresh | 0.25 | confidence threshold for filter boxes}"
        "{nms_thresh | 0.7 | iou threshold for nms}"
        "{help | 0 | print help information.}"
        "{classnames | ../../datasets/coco.names | class names file path}"
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

    string coco_names = parser.get<string>("classnames");
    if (stat(coco_names.c_str(), &info) != 0) {
        cout << "Cannot find classnames file." << endl;
        exit(1);
    }
    
    if (stat(input.c_str(), &info) != 0) {
        cout << "Cannot find input path." << endl;
        exit(1);
    }

    // initialize net
    YoloV8_det yolov8(bmodel_file, coco_names, dev_id, parser.get<float>("conf_thresh"), parser.get<float>("nms_thresh"));

    // profiling
    TimeStamp yolov8_ts;
    yolov8.m_ts = &yolov8_ts;

    // get batch_size
    int batch_size = yolov8.batch_size;

    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    // test images
    if (info.st_mode & S_IFDIR) {
        // get files
        vector<string> files_vector;
        DIR* pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        while ((ptr = readdir(pDir)) != 0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);

        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        vector<YoloV8BoxVec> boxes;
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
                auto ret = yolov8.Detect(batch_imgs, boxes);
                assert(0 == ret);

                for (int i = 0; i < batch_imgs.size(); i++) {
                    yolov8.draw_result(batch_mats[i], boxes[i]);
                    string img_file = "results/images/" + batch_names[i];
                    cv::imwrite(img_file, batch_mats[i]);
                    vector<json> bboxes_json;
                    if (boxes[i].size() != 0) {
                        for (auto bbox : boxes[i]) {
                            float bboxwidth = bbox.x2 - bbox.x1;
                            float bboxheight = bbox.y2 - bbox.y1;
                            json bbox_json;
                            bbox_json["category_id"] = bbox.class_id;
                            bbox_json["score"] = bbox.score;
                            bbox_json["bbox"] = {bbox.x1, bbox.y1, bboxwidth, bboxheight};
                            bboxes_json.push_back(bbox_json);
                        }
                    }

                    json res_json;
                    res_json["bboxes"] = bboxes_json;
                    res_json["image_name"] = batch_names[i];
                    results_json.push_back(res_json);

                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
                batch_names.clear();
                boxes.clear();
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
        cv::VideoCapture cap(input, cv::CAP_FFMPEG, dev_id);
        if(!cap.isOpened()) {
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
        auto output_fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4'); //use "H","2","6","4" for h264, "H","V","C","1" for h265.
        writer.open(output_path, output_fourcc, frameRate, cv::Size(w, h));
        bool end_flag = false;
        vector<cv::Mat> batch_mats;
        vector<bm_image> batch_imgs;
        vector<YoloV8BoxVec> boxes;
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
                CV_Assert(0 == yolov8.Detect(batch_imgs, boxes));
                for (int i = 0; i < batch_imgs.size(); i++) {
                    static int id = 0;
                    id++;
                    cout << id << ", det_nums: " << boxes[i].size() << endl;
                    yolov8.draw_result(batch_mats[i], boxes[i]);
                    writer.write(batch_mats[i]);
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_mats.clear();
                batch_imgs.clear();
                boxes.clear();
            }
        }
        writer.release();
        cap.release();
    }

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    yolov8_ts.calbr_basetime(base_time);
    yolov8_ts.build_timeline("yolov8 test");
    yolov8_ts.show_summary("yolov8 test");
    yolov8_ts.clear();
    return 0;
}