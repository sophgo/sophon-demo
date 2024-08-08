//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"

#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "../dependencies/include/yolov5.hpp"
#include "hrnet_pose.hpp"

using json = nlohmann::json;
using namespace std;

#define DRAW_OPENCV 1

json saveCocoKeypoints(vector<cv::Point2f>& keypoints, vector<float>& maxvals, string& image_name, float& box_score){

  int image_id = stoi(image_name.substr(0, image_name.find('.')), nullptr, 10);

  vector<float> keypoints_flatten;
  keypoints_flatten.reserve(keypoints.size() * 3);

  for (int i = 0; i < keypoints.size(); ++i) {
      float maxval = maxvals[i];
      keypoints_flatten.push_back(keypoints[i].x);
      keypoints_flatten.push_back(keypoints[i].y);
      keypoints_flatten.push_back(maxval);
  }

  float k_score = 0.0f;
  int num_valid = 0;
  float sum_valid = 0.0f;

  for (float maxval : maxvals) {
      if (maxval >= 0.2f) {
          sum_valid += maxval;
          num_valid++;
      }
  }

  if (num_valid > 0) {
      k_score = sum_valid / num_valid;
  }
 
  json result;
  result["image_id"] = image_id;
  result["category_id"] = 1;
  result["keypoints"] = keypoints_flatten;
  result["score"] = box_score * k_score;
 

  return result;

}

int main(int argc, char *argv[]){
  
  cout.setf(ios::fixed);
  
  const char *keys = "{input | ../../datasets/coco/val2017 | input path, images direction or video file path}"
    "{pose_bmodel | ../../models/BM1684X/hrnet_w32_256x192_int8.bmodel | path of pose estimation bmodel}"
    "{dev_id | 0 | TPU device id}"
    "{flip | true | whether using flipped imagese}"
    "{person_thresh | 0.5 | threshold for person detection}"
    "{detection_bmodel | ../../models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel | path of detection bmodel}"
    "{conf_thresh | 0.01 | confidence threshold for fiMlter boxes}"
    "{nms_thresh | 0.6 | iou threshold for nms}"
    "{help | 0 | print help information.}"
    "{classnames | ../../datasets/coco.names | class names file path}"
    "{use_cpu_opt | false | accelerate cpu postprocess}";

  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
      parser.printMessage();
      return 0;
  }

  string input = parser.get<string>("input");
  string pose_bmodel = parser.get<string>("pose_bmodel");
  int dev_id = parser.get<int>("dev_id");  
  bool flip = parser.get<bool>("flip");
  float person_thresh = parser.get<float>("person_thresh");

  string detection_bmodel = parser.get<string>("detection_bmodel");
  float conf_thresh = parser.get<float>("conf_thresh");
  float nms_thresh = parser.get<float>("nms_thresh");
  string coco_names = parser.get<string>("classnames");
  bool use_cpu_opt = parser.get<bool>("use_cpu_opt"); 

  struct stat info;

  if (stat(pose_bmodel.c_str(), &info) != 0){
    cout << "Cannot find pose estimation bmodel." << endl;
    exit(1);
  }

  if (stat(detection_bmodel.c_str(), &info) != 0){
    cout << "Cannot find detection bmodel." << endl;
    exit(1);
  }

  if (stat(coco_names.c_str(), &info) != 0) {
    cout << "Cannot find classnames file." << endl;
    exit(1);
  }

  if (stat(input.c_str(), &info) != 0){
    cout << "Cannot find input path." << endl;
    exit(1);
  }

  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: "  << dev_id << endl;
  bm_handle_t h = handle->handle();

  shared_ptr<BMNNContext> bm_detect_context = make_shared<BMNNContext>(handle, detection_bmodel.c_str());
  shared_ptr<BMNNContext> bm_pose_context = make_shared<BMNNContext>(handle, pose_bmodel.c_str());

  YoloV5 yolov5(bm_detect_context, use_cpu_opt);
  yolov5.Init(conf_thresh, nms_thresh, coco_names);

  HRNetPose hrnet_pose(bm_pose_context);
  hrnet_pose.Init(flip, coco_names);
  if (hrnet_pose.get_batch_size() != 1){
    cout << "Only supports batch size of 1 for HRNet." << endl;
    exit(1);
  }

  TimeStamp pose_ts;
  TimeStamp* ts = &pose_ts;
  yolov5.enableProfile(&pose_ts);
  hrnet_pose.enableProfile(&pose_ts);

  int batch_size = yolov5.batch_size();

  if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
  if (access("results/images", 0) != F_OK)
      mkdir("results/images", S_IRWXU);
  if (access("results/video", 0) != F_OK)
      mkdir("results/video", S_IRWXU);

  if (info.st_mode & S_IFDIR)
  {
    vector<string> files_vector;

    DIR *pDir;
    struct dirent* ptr;  
    pDir = opendir(input.c_str()); 

    while((ptr = readdir(pDir)) != 0) {
        
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }

    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end()); 

    vector<bm_image> batch_decode_images;  
    vector<string> batch_images_names;   
    vector<YoloV5BoxVec> yolov5_boxes;  
    vector<json> results_json;   

    int image_nums = files_vector.size(); 
    int count_det = 0;

    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++)
    {

      string image_file = *iter;
      count_det++;
      cout << count_det << "/" << image_nums << ", image_file: " << image_file << endl;

      ts->save("decode time");
      bm_image decode_image;
      picDec(h, image_file.c_str(), decode_image);
      ts->save("decode time");

      size_t index = image_file.rfind("/");  
      string image_name = image_file.substr(index + 1);  

      batch_decode_images.push_back(decode_image);  
      batch_images_names.push_back(image_name);

      iter++;
      bool end_flag = (iter == files_vector.end()); 
      iter--;

      if ((batch_decode_images.size() == batch_size || end_flag) && !batch_decode_images.empty())
      {
        CV_Assert(0 == yolov5.Detect(batch_decode_images, yolov5_boxes));

        ts->save("get person boxes time");
        vector<vector<YoloV5Box>> person_boxes = hrnet_pose.get_person_detection_boxes(yolov5_boxes, person_thresh);
        ts->save("get person boxes time");

        for (int i = 0; i < batch_decode_images.size(); i++)
        {
          
          if (person_boxes[i].size() != 0)
          {

          #if DRAW_OPENCV
            cv::Mat cv_mat_image;
            bm_image bm_image_to_mat = batch_decode_images[i];
            int ret = cv::bmcv::toMAT(&bm_image_to_mat, cv_mat_image);
            string image_file = "results/images/" + batch_images_names[i];
          #endif
            
            for(YoloV5Box& person_box : person_boxes[i])
            {
              vector<cv::Point2f> keypoints; 
              vector<float> maxvals;
              vector<cv::Mat> heatMaps;
              hrnet_pose.poseEstimate(batch_decode_images[i], person_box, keypoints, maxvals, heatMaps);

            #if DRAW_OPENCV
              hrnet_pose.drawPose(keypoints, cv_mat_image);
              cv::imwrite(image_file, cv_mat_image);
            #endif
      
              json result = saveCocoKeypoints(keypoints, maxvals, batch_images_names[i], person_box.score);
              results_json.emplace_back(result);

            }
          }

          bm_image_destroy(batch_decode_images[i]);
        }
        batch_decode_images.clear();
        batch_images_names.clear();
        yolov5_boxes.clear();
      }
    }

    string keypoints_results = "results/keypoints_results_cpp.json";
    ofstream(keypoints_results) << setw(4) << results_json;
  
    cout << "result saved in " << keypoints_results << endl;
  }


  else
  {
    VideoDecFFM decoder;
    decoder.openDec(&h, input.c_str());

    int id = 0;
    vector<bm_image> batch_decode_images; 
    vector<YoloV5BoxVec> yolov5_boxes;   

    bool end_flag = false;

    while(!end_flag){
      
      ts->save("decode time");
      bm_image *image = decoder.grab();
      ts->save("decode time");

      if (!image){
        end_flag = true;
      }

      else{
        batch_decode_images.push_back(*image);
      }

      if ((batch_decode_images.size() == batch_size || end_flag) && !batch_decode_images.empty()){
        
        CV_Assert(0 == yolov5.Detect(batch_decode_images, yolov5_boxes));

        ts->save("get person boxes time");
        vector<vector<YoloV5Box>> person_boxes = hrnet_pose.get_person_detection_boxes(yolov5_boxes, person_thresh);
        ts->save("get person boxes time");

        for (int i = 0; i < batch_decode_images.size(); i++)
        {

          id++;
          cout << id << ", det_persons_num: " << person_boxes[i].size() << endl;
          if (person_boxes[i].size() != 0)
          {

          #if DRAW_OPENCV
            cv::Mat cv_mat_image;
            bm_image bm_image_to_mat = batch_decode_images[i];
            int ret = cv::bmcv::toMAT(&bm_image_to_mat, cv_mat_image);
            string image_file = "results/video/" + to_string(id) + ".jpg";
          #endif

            for(YoloV5Box& person_box : person_boxes[i])
            {
              vector<cv::Point2f> keypoints; 
              vector<float> maxvals;
              vector<cv::Mat> heatMaps;

              hrnet_pose.poseEstimate(batch_decode_images[i], person_box, keypoints, maxvals, heatMaps);

            #if DRAW_OPENCV
              hrnet_pose.drawPose(keypoints, cv_mat_image);
              cv::imwrite(image_file, cv_mat_image);
            #endif

            }
          }
          bm_image_destroy(batch_decode_images[i]);
        }

        batch_decode_images.clear();
        yolov5_boxes.clear();
      }
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
 
  pose_ts.calbr_basetime(base_time);
 
  pose_ts.build_timeline("HRNetPose test");
 
  pose_ts.show_summary("HRNetPose test");

  pose_ts.clear();

  return 0;

}