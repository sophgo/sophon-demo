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
#include "yolov8_det.hpp"
#include "ff_decode.hpp"
#include "ff_video_encode.h"
bool is_video(const std::string& input) {
    const std::string videoExtensions[] = {
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".mpeg", ".3gp"
    };
    const std::string streamProtocols[] = {
        "rtsp://", "http://", "https://", "ftp://"
    };
    for (const auto& ext : videoExtensions) {
        if (input.size() >= ext.size() &&
            input.compare(input.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    for (const auto& protocol : streamProtocols) {
        if (input.find(protocol) == 0) {
            return true;
        }
    }
                                                                                                                 return false;
}
int main(int argc, char* argv[]) {
    // get params
    const char* keys =
        "{bmodel | yolov8s_int8_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{conf_thresh | 0.25 | confidence threshold for filter boxes}"
        "{nms_thresh | 0.7 | iou threshold for nms}"
        "{help | 0 | print help information.}"
        "{classnames | coco.names | class names file path}"
        "{output | rtsp://172.21.80.56:8554/test | output path}"
        "{input | test_car_person_1080P.mp4 | input path.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string input = parser.get<std::string>("input");
    std::string output = parser.get<std::string>("output");
    int dev_id = parser.get<int>("dev_id");
    bm_handle_t handle;
    bm_dev_request(&handle, dev_id);
    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }

    std::string coco_names = parser.get<std::string>("classnames");
    if (stat(coco_names.c_str(), &info) != 0) {
        std::cout << "Cannot find classnames file." << std::endl;
        exit(1);
    }
    
    if (stat(input.c_str(), &info) != 0) {
        std::cout << "Cannot find input path." << std::endl;
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
    if (is_video(input)){
        VideoDecFFM* decoder = new VideoDecFFM;
        int ret = decoder->openDec(&handle, input.c_str());
        if (ret != 0){
            throw std::runtime_error("Open decoder failed.");
        }
        VideoEnc_FFMPEG encoder;
        bool end_flag = false;
        std::vector<bm_image> batch_imgs;
        std::vector<YoloV8BoxVec> boxes;
        while (!end_flag) {
           bm_image *img = decoder->grab();

            if (!img) {
                // end_flag = true;
                delete decoder;
                decoder = new VideoDecFFM;
                ret = decoder->openDec(&handle, input.c_str());
                if (ret != 0){
                    throw std::runtime_error("Open decoder failed.");
                }
                continue;
            } else {
                batch_imgs.push_back(*img);
                delete img;
                img = nullptr;
            }
            if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
                // predict
                CV_Assert(0 == yolov8.Detect(batch_imgs, boxes));
                for (int i = 0; i < batch_size; i++) {
                    static int id = 0;
                    id++;
                    std::cout << id << ", det_nums: " << boxes[i].size() << std::endl;
                    yolov8.draw_result_bmcv(batch_imgs[i], boxes[i]);
                    AVFrame* frame_yuv420p = av_frame_alloc(); //for encoder
                    bm_image* bm_image_yuv420p = (bm_image*) malloc(sizeof(bm_image)); // for bmBufferDeviceMemFree
                    memcpy(bm_image_yuv420p, &batch_imgs[i], sizeof(bm_image));
                    assert(0 == bm_image_to_avframe(handle, bm_image_yuv420p, frame_yuv420p)); //on 1684x, this bmimg must on vpu heap.

                    if (encoder.isClosed()){
                        ret = encoder.openEnc(output.c_str(), 
                                              "h264_bm", 
                                              0, 
                                              25, 
                                              frame_yuv420p->width,
                                              frame_yuv420p->height, 
                                              frame_yuv420p->format, 
                                              25 * frame_yuv420p->width * frame_yuv420p->height / 8);
                        if (ret != 0){
                            throw std::runtime_error("Open encoder failed.");
                        }
                        if (0 != encoder.writeFrame(frame_yuv420p)){
                            std::cout << "encode failed once." << std::endl;
                        }
                    }else if (0 != encoder.writeFrame(frame_yuv420p)){
                        std::cout << "encode failed once." << std::endl;
                    }
                    av_frame_unref(frame_yuv420p);
                    av_frame_free(&frame_yuv420p);
                    // bm_image_destroy(batch_imgs[i]);
                }
                batch_imgs.clear();
                boxes.clear();
            }
        }
        encoder.closeEnc();
    }

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    yolov8_ts.calbr_basetime(base_time);
    yolov8_ts.build_timeline("yolov8 test");
    yolov8_ts.show_summary("yolov8 test");
    yolov8_ts.clear();
    return 0;
}