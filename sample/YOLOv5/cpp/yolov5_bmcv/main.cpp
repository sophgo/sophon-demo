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
#include "ff_decode.hpp"
#include "yolov5.hpp"
using json = nlohmann::json;
using namespace std;
#define WITH_ENCODE 1
int main(int argc, char *argv[]){
  cout.setf(ios::fixed);
  // get params
  const char *keys="{bmodel | ../../models/BM1690/yolov5s_v6.1_3output_fp32_1b.bmodel | bmodel file path}"
    "{dev_id | 0 | TPU device id}"
    "{conf_thresh | 0.001 | confidence threshold for filter boxes}"
    "{nms_thresh | 0.6 | iou threshold for nms}"
    "{help | 0 | print help information.}"
    "{input | ../../datasets/test | input path, images direction or video file path}"
    "{classnames | ../../datasets/coco.names | class names file path}"
    "{use_cpu_opt | false | accelerate cpu postprocess}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  int dev_id = parser.get<int>("dev_id");
  bool use_cpu_opt = parser.get<bool>("use_cpu_opt");

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
  if (stat(input.c_str(), &info) != 0){
    cout << "Cannot find input path." << endl;
    exit(1);
  }
  
  tpuRtInit();
  tpuRtSetDevice(dev_id);
  
  // create stream
  tpuRtStream_t stream;
  auto tpuret = tpuRtStreamCreate(&stream);
  assert(tpuret == tpuRtSuccess);

  // create handle
  bm_handle_t handle = new bm_context();
  handle->dev_id = dev_id;
  handle->stream = stream;

  // initialize net
  YoloV5 yolov5(handle, bmodel_file, dev_id, use_cpu_opt);
  CV_Assert(0 == yolov5.Init(
        parser.get<float>("conf_thresh"),
        parser.get<float>("nms_thresh"),
        coco_names));

  // profiling
  TimeStamp yolov5_ts;
  TimeStamp *ts = &yolov5_ts;
  yolov5.enableProfile(&yolov5_ts);

  // get batch_size
  int batch_size = yolov5.batch_size();

  // creat save path
  if (access("results", 0) != F_OK)
    mkdir("results", S_IRWXU);
  if (access("results/images", 0) != F_OK)
    mkdir("results/images", S_IRWXU);
  
  // test images
  if (info.st_mode & S_IFDIR){
    // get files
    vector<string> files_vector;
    DIR *pDir;
    struct dirent* ptr;
    pDir = opendir(input.c_str());
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());

    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    vector<YoloV5BoxVec> boxes;
    vector<json> results_json;
    bmcv_status_t ret = bmcvErrFailure;
    int cn = files_vector.size();
    int id = 0;
    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter; 
      id++;
      cout << id << "/" << cn << ", img_file: " << img_file << endl;
      ts->save("decode time");
      bm_image bmimg;
      picDec(handle, img_file.c_str(), bmimg);
      ts->save("decode time");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);
      
      iter++;
      bool end_flag = (iter == files_vector.end());
      iter--;
      if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
        // predict
        CV_Assert(0 == yolov5.Detect(batch_imgs, boxes));

        for(int i = 0; i < batch_imgs.size(); i++){
          vector<json> bboxes_json;
        #if WITH_ENCODE
          if (batch_imgs[i].image_format != 0){
            bm_image frame;
            ret = bm_image_create(handle, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            assert(ret == bmcvSuccess);
            ret = bm_image_alloc_dev_mem(frame, BMCV_HEAP1_ID);
            assert(ret == bmcvSuccess);
            ret = bmcv_image_storage_convert(handle, 1, &batch_imgs[i], &frame);
            int stride[3];
            bm_image_get_stride(frame, stride);
            assert(ret == bmcvSuccess);
            ret = bm_image_destroy(&batch_imgs[i]);
            assert(ret == bmcvSuccess);
            batch_imgs[i] = frame;
          }
        #endif
        
          // cv::Mat mat_img;
          // cv::bmcv::toMAT(&batch_imgs[i], mat_img);
          for (auto bbox : boxes[i]) {
#if DEBUG
            cout << "  class id=" << bbox.class_id << ", score = " << bbox.score << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",h=" << bbox.height << ")" << endl;
#endif
            // draw image with bm_image
              yolov5.draw_bmcv(handle, bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.width, bbox.height, batch_imgs[i]);
            // draw image with cv::Mat
              // yolov5.drawPred(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height, mat_img);
            
            // save result
            json bbox_json;
            bbox_json["category_id"] = bbox.class_id;
            bbox_json["score"] = bbox.score;
            bbox_json["bbox"] = {bbox.x, bbox.y, bbox.width, bbox.height};
            bboxes_json.push_back(bbox_json);
          }
          json res_json;
          res_json["image_name"] = batch_names[i];
          res_json["bboxes"] = bboxes_json;
          results_json.push_back(res_json);

          // save image
        #if WITH_ENCODE
          void* jpeg_data = NULL;
          size_t out_size = 0;
          ret = bmcv_image_jpeg_enc(handle, 1, &batch_imgs[i], &jpeg_data, &out_size, 85);
          if (ret == bmcvSuccess) {
            string img_file = "results/images/" + batch_names[i];
            std::cout << "fwriting " << img_file << std::endl;
            FILE *fp = fopen(img_file.c_str(), "wb");
            fwrite(jpeg_data, out_size, 1, fp);
            fclose(fp);
          } else {
            std::cout << "bmcv_image_jpeg_enc failed, ret: " << ret << std::endl; 
          }
          free(jpeg_data);
          // string filename = "results/images/" + batch_names[i];
          // cv::imwrite(filename, mat_img);
        #endif
          ret = bm_image_destroy(&batch_imgs[i]);
          assert(ret == bmcvSuccess);
        }
        batch_imgs.clear();
        batch_names.clear();
        boxes.clear();
      }
    }
    
    // save results
    size_t index = input.rfind("/");
    if(index == input.length() - 1){
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
  }
  
  // test video
  else {
    VideoDecFFM decoder;
    bmcv_status_t ret = bmcvErrFailure;
    decoder.openDec(handle, input.c_str());
    int id = 0;
    vector<bm_image> batch_imgs;
    vector<YoloV5BoxVec> boxes;
    bool end_flag = false;
    while(!end_flag){
      bm_image *img = decoder.grab();
      if (!img){
        end_flag=true;
      }else {
        batch_imgs.push_back(*img);
        delete img;
        img = nullptr;
      }
      if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
        CV_Assert(0 == yolov5.Detect(batch_imgs, boxes));
        for(int i = 0; i < batch_imgs.size(); i++){
          id++;
          cout << id << ", det_nums: " << boxes[i].size() << endl;
          if (batch_imgs[i].image_format != 0){
            bm_image frame;
            ret = bm_image_create(handle, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            assert(ret == bmcvSuccess);
            ret = bm_image_alloc_dev_mem(frame, BMCV_HEAP1_ID);
            assert(ret == bmcvSuccess);
            ret = bmcv_image_storage_convert(handle, 1, &batch_imgs[i], &frame);
            assert(ret == bmcvSuccess);
            ret = bm_image_destroy(&batch_imgs[i]);
            assert(ret == bmcvSuccess);
            batch_imgs[i] = frame;
          }
          for (auto bbox : boxes[i]) {
#if DEBUG
            cout << "  class id=" << bbox.class_id << ", score = " << bbox.score << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",h=" << bbox.height << ")" << endl;
#endif
            yolov5.draw_bmcv(handle, bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.width, bbox.height, batch_imgs[i], false);
          }
          string img_file = "results/images/" + to_string(id) + ".jpg";
          void* jpeg_data = NULL;
          size_t out_size = 0;
          ret = bmcv_image_jpeg_enc(handle, 1, &batch_imgs[i], &jpeg_data, &out_size, 85);
          if (ret == bmcvSuccess) {
            FILE *fp = fopen(img_file.c_str(), "wb");
            fwrite(jpeg_data, out_size, 1, fp);
            fclose(fp);
          } else {
            std::cout << "bmcv_image_jpeg_enc failed, ret: " << ret << std::endl;
          }
          free(jpeg_data);
          ret = bm_image_destroy(&batch_imgs[i]);
          assert(ret == bmcvSuccess);
        }
        batch_imgs.clear();
        boxes.clear();
      }
    }
  }
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  yolov5_ts.calbr_basetime(base_time);
  yolov5_ts.build_timeline("yolov5 test");
  yolov5_ts.show_summary("yolov5 test");
  yolov5_ts.clear();
  tpuRtStreamDestroy(stream);
  delete handle;
  return 0;
}
