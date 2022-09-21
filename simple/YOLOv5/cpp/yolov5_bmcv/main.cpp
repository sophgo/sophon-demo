/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/

#include "opencv2/opencv.hpp"
#include "yolov5.hpp"
#include <sys/types.h>
#include <unistd.h>
#include<sys/wait.h>
#include <errno.h>
#include <string.h>

int vsystem(const char *cmdStr)
{
  int status;
  int pid;

  if(NULL == cmdStr){
    printf("invalid param\n");
    return -1;
  }	

  if((pid = vfork()) < 0){
    printf("vsystem %s failure, %s\n", cmdStr, strerror(errno));
    return -1;		
  }
  else if(0 == pid){
    execl("/bin/sh", "sh", "-c", cmdStr, NULL);
    _exit(127);
  }
  else{
    while(waitpid(pid, &status, 0) < 0){
      if(EINTR != errno)
      {
        status = -1;
        break;
      }
    }
	}
	
	return status;
	
}

static std::string trim(std::string &s) {  
  if (s.empty()){  
    return s;  
  }  
  s.erase(0,s.find_first_not_of(" "));  
  s.erase(s.find_last_not_of(" ") + 1);  
  return s;  
}

static std::vector<std::string> split(const std::string& s){
  std::vector<std::string> result(1);
  for(auto c: s){
    if(c==','){
      result.back() = trim(result.back());
      result.push_back("");
    } else {
      result.back() += c;
    }
  }
  result.back() = trim(result.back());
  return result;
}

int main(int argc, char *argv[])
{
  const char *keys="{bmodel | ../../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel | bmodel file path}"
    "{tpuid | 0 | TPU device id}"
    "{conf | 0.5 | confidence threshold for filter boxes}"
    "{obj | 0.5 | object score threshold for filter boxes}"
    "{iou | 0.5 | iou threshold for nms}"
    "{help | 0 | Print help information.}"
    "{is_video | 0 | input video file path}"
    "{frame_num | 0 | number of frames in video to process, 0 means processing all frames}"
    "{input | ../../data/images/zidane.jpg | input stream file path}"
    "{classnames |../../data/images/coco.names | class names' file path}";

  // profiling
  TimeStamp ts;

  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  std::string bmodel_file = parser.get<std::string>("bmodel");
  int dev_id = parser.get<int>("tpuid");
  BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
  // Load bmodel
  std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
  YoloV5 yolo(bm_ctx);

  yolo.enableProfile(&ts);
  std::string coco_names = parser.get<std::string>("classnames");

  CV_Assert(0 == yolo.Init(
        parser.get<float>("conf"),
        parser.get<float>("obj"),
        parser.get<float>("iou"),
        coco_names));

  if (!parser.get<bool>("is_video")) {
    auto image_files = split(parser.get<std::string>("input"));
    std::vector<cv::Mat> images;
    for(auto image_file: image_files){
      std::cout<<"input image #"<<images.size()<<": "<<image_file<<std::endl;
      std::cout<<"dev_id： "<< dev_id <<std::endl;
      cv::Mat img = cv::imread(image_file, cv::IMREAD_AVFRAME);
      if (img.empty()){
        std::cout << "Warning! decoded image is empty" << std::endl;
        return -1;
      }
      images.push_back(img);
    }

    std::vector<YoloV5BoxVec> boxes;
    CV_Assert(0 == yolo.Detect(images, boxes));

    std::cout<<std::endl;
    for (int i = 0; i < (int) images.size(); ++i) {

      std::cout<<"image #"<<i<<" "<<image_files[i]<<" detection num: "<< boxes[i].size()<<std::endl;
      cv::Mat frame = images[i];
      auto frame_boxes = boxes[i];
      for (auto bbox : boxes[i]) {
        std::cout << "  class id=" << bbox.class_id << ", score = " << bbox.score
          << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",y=" << bbox.height << ")"
          << std::endl;
        yolo.drawPred(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.x + bbox.width,
            bbox.y + bbox.height, frame);
      }

      {
        // 保存输出
        std::string cmd("mkdir -p results");
        int ret = vsystem(cmd.c_str());
        if (ret)
        {
          std::cout << "create dir error: " << ret << ", :" << strerror(errno) << std::endl;
          return -1;
        }

        size_t index = bmodel_file.rfind("/");
        std::string model_name = bmodel_file.substr(index + 1);
        std::string output_file = "results/" + model_name + cv::format("_bmcv_cpp_result_%d.jpg", i);
        cv::imwrite(output_file, frame);
      }
      std::cout<<std::endl;
    }

  }else {
    std::string input_url = parser.get<std::string>("input");
    // open stream
    cv::VideoCapture cap(input_url, cv::CAP_ANY, dev_id);
    if (!cap.isOpened()) {
      std::cout << "open stream " << input_url << " failed!" << std::endl;
      exit(1);
    }

    // get resolution
    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "resolution of input stream: " << h << "," << w << std::endl;

    int frame_count=0;
    int frame_num = parser.get<int>("frame_num");
    while(1) {
      cv::Mat img;
      if (!cap.read(img)) {
        std::cout << "Read frame failed or end of file!" << std::endl;
        exit(1);
      }

      std::vector<cv::Mat> images;
      images.push_back(img);

      std::vector<YoloV5BoxVec> boxes;

      CV_Assert(0 == yolo.Detect(images, boxes));

      for (int i = 0; i < (int) images.size(); ++i) {

        cv::Mat frame = images[i];
        std::cout << "frame #"<<frame_count<<": detect boxes: " << boxes[i].size() << std::endl;
        if(frame_num<10 || frame_count%32 == 0) {
          for (auto bbox : boxes[i]) {
            //std::cout << "class id =" << bbox.class_id << ",score = " << bbox.score
            //          << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",y=" << bbox.height << ")"
            //          << std::endl;
            yolo.drawPred(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.x + bbox.width,
                bbox.y + bbox.height, frame);
          }

            // 保存结果
            std::string cmd("mkdir -p results");
            int ret = vsystem(cmd.c_str());
            if (ret)
            {
              std::cout << "create dir error: " << ret << ", :" << strerror(errno) << std::endl;
              return -1;
            }

            size_t index = bmodel_file.rfind("/");
            std::string model_name = bmodel_file.substr(index + 1);
            std::string output_file = "results/" + model_name + cv::format("_bmcv_cpp_result_%d.jpg", frame_count);
            cv::imwrite(output_file, frame);
        }
        frame_count ++;
      }
			if(frame_num>0 && frame_count>=frame_num) break;
    }

  }


  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("YoloV5");
  ts.show_summary("YoloV5 Demo");
  ts.clear();

}
