//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "json.hpp"
#include "openpose.hpp"
#include "pose_postprocess.hpp"
// #define DEBUG
using json = nlohmann::json;
using namespace std;

int main(int argc, char** argv) {
  cout.setf(ios::fixed);
  // get params
  const char *keys = "{help | 0 | print help information.}"
    "{dev_id | 0 | TPU device id}"
    "{bmodel | ../../models/BM1684/pose_coco_fp32_1b.bmodel | bmodel file path}"
    "{input | ../../datasets/test | input path, images direction or video file path}"
    "{performance_opt | no_opt | performance optimization type, supporting [tpu_kernel_opt, tpu_kernel_half_img_size_opt, cpu_opt, no_opt]}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  int dev_id = parser.get<int>("dev_id");
  std::string performance_opt = parser.get<std::string>("performance_opt");
  assert(performance_opt == "no_opt" || performance_opt == "tpu_kernel_opt" | performance_opt == "tpu_kernel_half_img_size_opt" || performance_opt == "cpu_opt");
  bool use_tpu_kernel_post = performance_opt == "tpu_kernel_opt" || performance_opt == "tpu_kernel_half_img_size_opt";
  bool restore_half_img_size = performance_opt == "tpu_kernel_half_img_size_opt";
  bool use_cpu_opt = performance_opt == "cpu_opt";
  if (use_tpu_kernel_post)
    cout << "Using kernel postprocess." << endl;
  if (restore_half_img_size)
    cout << "Restore half size of image in kernel postprocess." << endl;
  if (use_cpu_opt)
    cout << "Using cpu optimization." << endl;

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

  // creat handle
  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: "  << dev_id << endl;
  bm_handle_t h = handle->handle();

  // load bmodel
  shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

  // initialize net
  OpenPose openpose(bm_ctx);
  CV_Assert(0 == openpose.Init(use_tpu_kernel_post));

  // profiling
  TimeStamp openpose_ts;
  TimeStamp *ts = &openpose_ts;
  openpose.enableProfile(&openpose_ts);

  // get batch_size
  int batch_size = openpose.batch_size();
  PoseKeyPoints::EModelType model_type = openpose.get_model_type();

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

    // sort files
    // sort(files_vector.begin(),files_vector.end());
    
    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    vector<PoseKeyPoints> vct_keypoints;
    vector<json> results_json;
    int cn = files_vector.size();
    int id = 0;
    // ts->save("openpose overall");
    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter;
      id++;
      cout << id << "/" << cn << ", filename: " << img_file << endl;
      ts->save("decode time");
      // cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
      bm_image bmimg;
      picDec(h, img_file.c_str(), bmimg);
      ts->save("decode time");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);
      if (batch_imgs.size() == batch_size) {
        CV_Assert(0 == openpose.Detect(batch_imgs, vct_keypoints, performance_opt));
        for(int i = 0; i < batch_size; i++){
          cout << "keypoints.size: " << vct_keypoints[i].keypoints.size() << endl;
          string img_file = "results/images/" + batch_names[i];
          // save image
          // cv::Mat mat;
          // cv::bmcv::toMAT(&batch_imgs[i], mat);
          // OpenPosePostProcess::renderPoseKeypointsCpu(mat, vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
          // cv::imwrite(img_file, mat);
          // bm_image_destroy(batch_imgs[i]);

          // bmcv save image
          bm_image res_bmimg = OpenPosePostProcess::renderPoseKeypointsBmcv(h, batch_imgs[i], vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
          void* jpeg_data = NULL;
          size_t out_size = 0;
          int ret = bmcv_image_jpeg_enc(h, 1, &res_bmimg, &jpeg_data, &out_size);
          if (ret == BM_SUCCESS) {
            FILE *fp = fopen(img_file.c_str(), "wb");
            fwrite(jpeg_data, out_size, 1, fp);
            fclose(fp);
          }
          free(jpeg_data);
          bm_image_destroy(res_bmimg);

          // save result
          json res_json;
          res_json["image_name"] = batch_names[i];
          res_json["keypoints"] = vct_keypoints[i].keypoints;
          results_json.push_back(res_json);
        }
        batch_imgs.clear();
        batch_names.clear();
        vct_keypoints.clear();
      }
    }
    if (!batch_imgs.empty()){
      // predict
      CV_Assert(0 == openpose.Detect(batch_imgs, vct_keypoints, performance_opt));
      for(int i = 0; i < batch_imgs.size(); i++){
        cout << "keypoints.size: " << vct_keypoints[i].keypoints.size() << endl;
        string img_file = "results/images/" + batch_names[i];
        // save image 
        // cv::Mat mat;
        // cv::bmcv::toMAT(&batch_imgs[i], mat);
        // OpenPosePostProcess::renderPoseKeypointsCpu(mat, vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
        // cv::imwrite(img_file, mat);
        // bm_image_destroy(batch_imgs[i]);

        // bmcv save image
        bm_image res_bmimg = OpenPosePostProcess::renderPoseKeypointsBmcv(h, batch_imgs[i], vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
        void* jpeg_data = NULL;
        size_t out_size = 0;
        int ret = bmcv_image_jpeg_enc(h, 1, &res_bmimg, &jpeg_data, &out_size);
        if (ret == BM_SUCCESS) {
          FILE *fp = fopen(img_file.c_str(), "wb");
          fwrite(jpeg_data, out_size, 1, fp);
          fclose(fp);
        }
        free(jpeg_data);
        bm_image_destroy(res_bmimg);


        // save result
        json res_json;
        res_json["image_name"] = batch_names[i];
        res_json["keypoints"] = vct_keypoints[i].keypoints;
        results_json.push_back(res_json);
      }
      batch_imgs.clear();
      batch_names.clear();
      vct_keypoints.clear();
    }
    // ts->save("openpose overall");
    
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
    decoder.openDec(&h, input.c_str());
    int id = 0;
    vector<bm_image> batch_imgs;
    vector<PoseKeyPoints> vct_keypoints;
    while(true){
      bm_image *img = decoder.grab();
      if (!img)
        break;
      batch_imgs.push_back(*img);
      if (batch_imgs.size() == batch_size) {
        CV_Assert(0 == openpose.Detect(batch_imgs, vct_keypoints, performance_opt));
        for(int i = 0; i < batch_size; i++){
          id++;
          cout << id << ", keypoints.size: " << vct_keypoints[i].keypoints.size() << endl;
          string img_file = "results/images/" + to_string(id) + ".jpg";
          // cv::Mat mat;
          // cv::bmcv::toMAT(&batch_imgs[i], mat);
          // OpenPosePostProcess::renderPoseKeypointsCpu(mat, vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
          // cv::imwrite(img_file, mat);
          // bm_image_destroy(batch_imgs[i]);
          bm_image res_bmimg = OpenPosePostProcess::renderPoseKeypointsBmcv(h, batch_imgs[i], vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
          void* jpeg_data = NULL;
          size_t out_size = 0;
          int ret = bmcv_image_jpeg_enc(h, 1, &res_bmimg, &jpeg_data, &out_size);
          if (ret == BM_SUCCESS) {
            FILE *fp = fopen(img_file.c_str(), "wb");
            fwrite(jpeg_data, out_size, 1, fp);
            fclose(fp);
          }
          free(jpeg_data);
          bm_image_destroy(res_bmimg);
        }
        batch_imgs.clear();
        vct_keypoints.clear();
      }
    }
    if (!batch_imgs.empty()){
      CV_Assert(0 == openpose.Detect(batch_imgs, vct_keypoints, performance_opt));
      for(int i = 0; i < batch_imgs.size(); i++){
        id++;
        cout << id << ", keypoints.size: " << vct_keypoints[i].keypoints.size() << endl;
        string img_file = "results/images/" + to_string(id) + ".jpg";
        // cv::Mat mat;
        // cv::bmcv::toMAT(&batch_imgs[i], mat);
        // OpenPosePostProcess::renderPoseKeypointsCpu(mat, vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
        // cv::imwrite(img_file, mat);
        // bm_image_destroy(batch_imgs[i]);
        bm_image res_bmimg = OpenPosePostProcess::renderPoseKeypointsBmcv(h, batch_imgs[i], vct_keypoints[i].keypoints,  vct_keypoints[i].shape, 0.05, 1.0, model_type);
        void* jpeg_data = NULL;
        size_t out_size = 0;
        int ret = bmcv_image_jpeg_enc(h, 1, &res_bmimg, &jpeg_data, &out_size);
        if (ret == BM_SUCCESS) {
          FILE *fp = fopen(img_file.c_str(), "wb");
          fwrite(jpeg_data, out_size, 1, fp);
          fclose(fp);
        }
        free(jpeg_data);
        bm_image_destroy(res_bmimg);
      }
      batch_imgs.clear();
      vct_keypoints.clear();
    }
  }
  // print speed
    // cout << "================" << endl;
    // vector<time_stamp_t> t_infer = *openpose_ts.records_["openpose inference"];
    // microseconds sum_infer(0);
    // for (size_t j = 0; j < t_infer.size(); j += 2) {
    //   microseconds duration = duration_cast<microseconds>(t_infer[j + 1] - t_infer[j]);
    //   sum_infer += duration;
    // }
    // cout << "infer_time = " << float((sum_infer / (t_infer.size() / 2)).count())/1000/batch_size << "ms" << endl;

    // vector<time_stamp_t> t_overall = *openpose_ts.records_["openpose overall"];
    // microseconds sum_overall(0);
    // for (size_t j = 0; j < t_overall.size(); j += 2) {
    //   microseconds duration = duration_cast<microseconds>(t_overall[j + 1] - t_overall[j]);
    //   sum_overall += duration;
    // }
    // cout << "QPS = " << cn * 1000000 / int((sum_overall / (t_overall.size() / 2)).count()) << endl;
    // cout << "cn = " << cn << ", time = " << int((sum_overall / (t_overall.size() / 2)).count()) << endl;
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  openpose_ts.calbr_basetime(base_time);
  openpose_ts.build_timeline("openpose test");
  openpose_ts.show_summary("openpose test");
  openpose_ts.clear();

  return 0;
}
