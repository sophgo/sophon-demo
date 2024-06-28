#include <iostream>
#include <cstdio>
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <thread>
#include "yolov5.hpp"

int main(int argc, char* argv[]) {
    const char* keys =
        "{bmodel | ../../models/BM1688/yolov5s_v6.1_3output_int8_4b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{conf_thresh | 0.5 | confidence threshold for filter boxes}"
        "{nms_thresh | 0.5 | iou threshold for nms}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/test_car_person_1080P.mp4 | input video file path}"
        "{chan_num | 2 | copy the input video into chan_num copies}"
        "{classnames | ../../datasets/coco.names | class names file path}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string input = parser.get<std::string>("input");
    std::string coco_names = parser.get<std::string>("classnames");

    int dev_id = parser.get<int>("dev_id");
    float conf_thresh = parser.get<float>("conf_thresh");
    float nms_thresh = parser.get<float>("nms_thresh");
    int chan_num = parser.get<int>("chan_num");
    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }
    if (stat(coco_names.c_str(), &info) != 0) {
        std::cout << "Cannot find classnames file." << std::endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0) {
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    // create save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    std::string save_paths[chan_num];
    for (int i = 0; i < chan_num; i++) {
        save_paths[i] = "results/images_chan_" + std::to_string(i);
        if (access(save_paths[i].c_str(), 0) != F_OK)
            mkdir(save_paths[i].c_str(), S_IRWXU);
    }

    // check board type
    char board_name[50];
    bm_handle_t handle;
    assert(BM_SUCCESS == bm_dev_request(&handle, dev_id));
    assert(BM_SUCCESS == bm_get_board_name(handle, board_name));
    printf("########################\n");
    printf("board_name: %s\n", board_name);
    printf("########################\n");
    if(strncmp(board_name, "BM1688-SOC", 8) != 0){
        std::cout << "Only support BM1688-SOC board." << std::endl;
        exit(1);
    }

    std::vector<std::thread*> m_threads;
    for (int i = 0; i < chan_num; i++) {
        std::thread* pth = new std::thread([&, i] {
            auto yolov5 = YOLOv5(dev_id, bmodel_file, conf_thresh, nms_thresh, coco_names);
            yolov5.set_core_id(i % 2);
            int batch_size = yolov5.get_batch_size();
            cv::VideoCapture cap(input, cv::CAP_ANY, dev_id);
            if (!cap.isOpened()) {
                printf("open video stream %d failed!\n", i);
                return;
            }
            int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
            printf("Chan:%d; Frame_num:%d, Resolution:<%dx%d>;\n", i, frame_num, h, w);
            int frame_count = 0;
            bool end_flag;
            std::vector<cv::Mat> batch_imgs;
            std::vector<std::string> batch_names;
            std::vector<YoloV5BoxVec> boxes;
            while (!end_flag) {
                std::string frame_name = "chan" + std::to_string(i) + "_frame" + std::to_string(frame_count) + ".jpg";
                printf("Chan:%d; Frame_id:%d\n", i, frame_count++);
                cv::Mat img;
                cap >> img;
                if (img.empty()) {
                    end_flag = true;
                }else{
                    batch_imgs.push_back(img);
                    batch_names.push_back(frame_name);
                }
                if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()){
                    assert(0 == yolov5.yolov5_detect(batch_imgs, boxes));
                    for (int j = 0; j < batch_imgs.size(); j++) {
                        for (auto bbox : boxes[j]) {
                            yolov5.yolov5_draw(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.width + bbox.x,
                                            bbox.height + bbox.y, batch_imgs[j]);
                        }
                        cv::imwrite(save_paths[i]+"/"+batch_names[j], batch_imgs[j]);
                    }
                    batch_imgs.clear();
                    batch_names.clear();
                    boxes.clear();  
                }
            }
        });
        m_threads.push_back(pth);
    }
    for(int i = 0; i < m_threads.size(); i++){
        m_threads[i]->join();
        delete m_threads[i];
        m_threads[i] = NULL;
    }
    return 0;
}