// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "DecoderConsole.h"
#include "VideoConsole.h"
#include"opencv2/opencv.hpp"
#include<string>
#include<fstream>
#include"common.h"
#include"yolov5.h"
#include"json.h"
#include"profiler.h"

//1: 多路解码+yolov5_pipeline+多路QT显示  0: 多路解码+多路QT显示
#define OPEN_YOLOV5 1


int main(int argc, char *argv[]){

    std::string keys = "{config | ./config/yolov5_app.json | path to config.json}";
    cv::CommandLineParser parser(argc, argv, keys);
    std::string config_file = parser.get<std::string>("config");

    std::ifstream file(config_file.c_str());
    if (!file.is_open()) {
        std::cerr << "Failed to open json file." << std::endl;
        return 1;
    }
    nlohmann::json config;
    file >> config;

    //此处需要按顺序填写需要处理的rtsp流地址。
    std::vector<std::string> url_vec_= config["decoder"]["urls"];
    int channel_num = url_vec_.size();
    //QT显示界面的大小，一路视频为（1，1），四路视频应为（2，2），以此类推
    int display_channel_rows = config["display"]["rows"];
    int display_channel_cols = config["display"]["cols"];

    int dev_id = config["yolov5"]["dev_id"];
    std::string bmodel_path = config["yolov5"]["bmodel_path"];
    std::string tpu_kernel_module_path = config["yolov5"]["tpu_kernel_module_path"];
    int que_size = config["yolov5"]["que_size"];
    int skip_num = config["yolov5"]["skip_num"];
    float nmsThreshold = config["yolov5"]["nmsThreshold"];
    float confThreshold = config["yolov5"]["confThreshold"];

    QApplication app(argc, argv);
    
#if OPEN_YOLOV5
    VideoConsole<FrameInfoDetect> video_console(display_channel_rows,display_channel_cols,channel_num,channel_num);
    Yolov5 yolo(dev_id, bmodel_path, tpu_kernel_module_path, que_size ,skip_num, nmsThreshold, confThreshold);
#else
    VideoConsole<bm_image> video_console(display_channel_rows,display_channel_cols,channel_num,channel_num);
#endif

    DecoderConsole multi_dec;
    for(int i=0;i<url_vec_.size();i++){
        multi_dec.addChannel(url_vec_[i]);
    }
    std::this_thread::sleep_for(std::chrono::seconds(3));

#if OPEN_YOLOV5

    std::thread display_thread([&](){
        FpsProfiler fpsApp("yolov5",100);
        while(true){
                auto ptr = yolo.get_img();
                video_console.push_img(ptr->channel_id,ptr);
                fpsApp.add(1);
        }
    });
    std::thread decode_thread([&](){
        while(true){
            for(int channel_id=0; channel_id<channel_num; channel_id++){
                std::shared_ptr<bm_image> image;
                if(multi_dec.read(channel_id, image)==0){
                    yolo.push_img(channel_id, image);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    
    display_thread.detach();
    decode_thread.detach();

#else
    std::thread only_display([&](){
        FpsProfiler fpsApp("only_display",100);
        while(true){
            for(int channel_id=0; channel_id < channel_num ;channel_id++){
                std::shared_ptr<bm_image> image;
                if(multi_dec.read(channel_id, image)==0){
                    
                    video_console.push_img(channel_id, image);
                    fpsApp.add(1);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    only_display.detach();
#endif

    return app.exec();
}