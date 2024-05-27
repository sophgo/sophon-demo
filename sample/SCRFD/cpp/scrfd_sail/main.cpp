//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>

#include "ff_decode.hpp"
#include "opencv2/opencv.hpp"
#include "scrfd.hpp"
using namespace std;
#define USE_OPENCV_DECODE 0

#define IMG_RESIZE_DIMS 640, 640
#define BGR_MEAN 127.5, 127.5, 127.5
#define INPUT_SCALE 0.00781  // 1/128.0
#define FACE_POINTS_SIZE 5

int main(int argc, char* argv[]) {
  cout.setf(ios::fixed);
  // get params
  const char* keys =
      "{bmodel | ../../models/BM1684/scrfd_10g_kps_fp32_1b.bmodel | bmodel "
      "file "
      "path}"
      "{dev_id | 0 | TPU device id}"
      "{conf_thresh | 0.5 | confidence threshold for filter boxes}"
      "{nms_thresh | 0.4 | iou threshold for nms}"
      "{help | 0 | print help information.}"
      "{eval | False | if true then gen result_txt}"
      "{input | ../../datasets/test/ | input path, images direction or video "
      "file path}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  int dev_id = parser.get<int>("dev_id");
  bool eval = parser.get<bool>("eval");

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

  // create handle
  auto handle = sail::Handle(dev_id);
  sail::Bmcv bmcv(handle);  // for imwrite
  cout << "set device id: " << dev_id << endl;

  // initialize net
  Scrfd scrfd(dev_id, bmodel_file);
  CV_Assert(0 == scrfd.Init(parser.get<float>("conf_thresh"),
                            parser.get<float>("nms_thresh")));

  // profiling
  TimeStamp scrfd_ts;
  TimeStamp* ts = &scrfd_ts;
  scrfd.enableProfile(&scrfd_ts);

  // get batch_size
  int batch_size = scrfd.batch_size();

  // creat save path
  if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
  if (access("results/images", 0) != F_OK) mkdir("results/images", S_IRWXU);
  const string prediction_dir = "../../tools/prediction_dir";
  const string result_dir = "./results/txt_results";
  if (access("./results/txt_results", 0) != F_OK)
    mkdir("./results/txt_results", S_IRWXU);
  if (eval) {
    if (access("../../tools/prediction_dir", 0) != F_OK)
      mkdir("../../tools/prediction_dir", S_IRWXU);
  }
  // test images
  if (info.st_mode & S_IFDIR) {
    // get files
    vector<string> files_vector;
    scrfd.readDirectory(input, files_vector, true);
    std::sort(files_vector.begin(), files_vector.end());

    vector<string> batch_names;
    vector<ScrfdBoxVec> batch_boxes;
    vector<cvai_face_info_t> boxes;
    int cn = files_vector.size();
    int id = 0;
    for (vector<string>::iterator iter = files_vector.begin();
         iter != files_vector.end(); iter++) {
      vector<cv::Mat> cvmats;
      cvmats.resize(batch_size);  // bmimage attach to cvmats, so we must keep
                                  // cvmat present.
      vector<sail::BMImage> batch_imgs;  // re declare for every batch, to free
                                         // sail::BMImage inside
      batch_imgs.resize(batch_size);     // push_back forbidden, sail use
                                         // BMImageArray to manage BMImage
      string img_file_name = *iter;
      size_t last_slash_index = img_file_name.find_last_of("/\\");
      size_t second_last_slash_index =
          img_file_name.find_last_of("/\\", last_slash_index - 1);
      std::string directory_name =
          img_file_name.substr(second_last_slash_index + 1,
                               last_slash_index - second_last_slash_index - 1);
      for (int i = 0; i < batch_size; i++) {
        if (iter == files_vector.end()) {
          iter--;
          cout << "using last img file to complete img batch" << endl;
        }
        string img_file = *iter;
        string img_name = img_file.substr(img_file.rfind("/") + 1);
        batch_names.push_back(img_name);  // batch_names has real batch_size
        id++;
        cout << id << "/" << cn << ", img_file: " << img_file << endl;
        ts->save("decode time");
#if USE_OPENCV_DECODE
        cvmats[i] = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
        bmcv.mat_to_bm_image(cvmats[i], batch_imgs[i]);
#else
        sail::Decoder decoder((const string)img_file, true, dev_id);
        int ret = decoder.read(handle, batch_imgs[i]);
        if (ret != 0) {
          cout << "read failed"
               << "\n";
        }
#endif

#if DEBUG
        cout << "batch img:" << batch_imgs[i].format() << " "
             << batch_imgs[i].dtype() << endl;
#endif
        ts->save("decode time");

        iter++;
      }
      iter--;
      CV_Assert(0 == scrfd.Detect(batch_imgs, batch_boxes));

      for (int i = 0; i < batch_size; i++) {  // use real batch size
        if (i > 0 && batch_names[i] == batch_names[i - 1]) {
          break;  // last batch may have same conponents.
        }
        if (eval) {
          string base_name =
              batch_names[i].substr(0, batch_names[i].rfind('.'));
          string txt_name = base_name + ".txt";
          string sub_result_directory = prediction_dir + "/" + directory_name;
          if (access(sub_result_directory.c_str(), 0) != F_OK)
            mkdir(sub_result_directory.c_str(), S_IRWXU);
          string file_path =
              prediction_dir + "/" + directory_name + "/" + txt_name;
          ofstream result_file(file_path);
          if (result_file.is_open()) {
            result_file << batch_names[i] << std::endl;
            result_file << batch_boxes[i].size() << std::endl;
            for (auto boxes : batch_boxes[i]) {
              cvai_bbox_t b_box = boxes.bbox;
              float x1 = (float)b_box.x1;
              float y1 = (float)b_box.y1;
              float x2 = (float)b_box.x2;
              float y2 = (float)b_box.y2;
              cvai_pts_t pti = boxes.pts;
              float bbox_width = std::abs(x2 - x1);
              float bbox_height = std::abs(y2 - y1);
              result_file << x1 << " " << y1 << " " << bbox_width << " "
                          << bbox_height << " " << b_box.score << endl;
            }
            result_file.close();
          } else {
          }
        } else {
          string base_name =
              batch_names[i].substr(0, batch_names[i].rfind('.'));
          string txt_name = base_name + ".txt";
          string sub_result_directory = result_dir + "/" + directory_name;
          if (access(sub_result_directory.c_str(), 0) != F_OK)
            mkdir(sub_result_directory.c_str(), S_IRWXU);
          string file_path = result_dir + "/" + directory_name + "/" + txt_name;
          ofstream result_file(file_path);
          if (result_file.is_open()) {
            result_file << batch_names[i] << std::endl;
            result_file << batch_boxes[i].size() << std::endl;
            for (auto boxes : batch_boxes[i]) {
              cvai_bbox_t b_box = boxes.bbox;
              float x1 = (float)b_box.x1;
              float y1 = (float)b_box.y1;
              float x2 = (float)b_box.x2;
              float y2 = (float)b_box.y2;
              cvai_pts_t pti = boxes.pts;
              float bbox_width = std::abs(x2 - x1);
              float bbox_height = std::abs(y2 - y1);
              result_file << x1 << " " << y1 << " " << bbox_width << " "
                          << bbox_height << " " << b_box.score << endl;
#if USE_OPENCV_DECODE
              scrfd.draw_opencv(boxes, cvmats[i]);
#else
              scrfd.draw_bmcv(pti, b_box.score, int(x1), int(y1),
                              int(bbox_width), int(bbox_height), batch_imgs[i],
                              false, false);
#endif
            }
            result_file.close();
          } else {
          }
        }
        if (!eval) {
#if USE_OPENCV_DECODE
          cv::imwrite("./results/images/" + batch_names[i], cvmats[i]);
#else
          bmcv.imwrite("./results/images/" + batch_names[i], batch_imgs[i]);
#endif
        } else {
        }
      }
      // batch_imgs.clear(); //severe bug here, do not free batch_imgs!
      batch_names.clear();
      boxes.clear();
      batch_boxes.clear();
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
    string result_output_dir;
    if (eval) {
      result_output_dir = prediction_dir;
    } else {
      result_output_dir = result_dir;
    }
    cout << "================" << endl;
    cout << "result saved in " << result_output_dir << endl;
  }
  // test video
  else {
    vector<ScrfdBoxVec> batch_boxes;
    vector<cvai_face_info_t> boxes;
    sail::Decoder decoder(input, false, dev_id);
    int id = 0;
    bool flag = true;
    while (1) {
      vector<sail::BMImage> batch_imgs;
      batch_imgs.resize(batch_size);
      for (int i = 0; i < batch_size; i++) {
        int ret = decoder.read(handle, batch_imgs[i]);
        if (ret != 0) {
          flag = false;
          break;  // discard last batch.
        }
      }
      if (flag == false) {
        break;
      }
      CV_Assert(0 == scrfd.Detect(batch_imgs, batch_boxes));
      for (int i = 0; i < batch_size; i++) {  // use real batch size
        cout << ++id << ", det_nums: " << batch_boxes[i].size() << endl;
        for (auto boxes : batch_boxes[i]) {
          cvai_bbox_t b_box = boxes.bbox;
          int x1 = (int)b_box.x1;
          int y1 = (int)b_box.y1;
          int x2 = (int)b_box.x2;
          int y2 = (int)b_box.y2;
          cvai_pts_t pti = boxes.pts;
          int bbox_width = std::abs(x2 - x1);
          int bbox_height = std::abs(y2 - y1);
#if DEBUG
          cout << ", score = " << b_box.score << " (x=" << x1 << ",y=" << y1
               << ",w=" << bbox_width << ",h=" << bbox_height << ")" << endl;
#endif
          // draw image
          scrfd.draw_bmcv(pti, b_box.score, x1, y1, bbox_width, bbox_height,
                          batch_imgs[i], false, false);
        }
#if USE_OPENCV_DECODE
        scrfd.draw_opencv(boxes, cvmats[i]);
#else
        bmcv.imwrite("./results/images/" + to_string(id) + ".jpg",
                     batch_imgs[i]);
#endif
      }
      boxes.clear();
      batch_boxes.clear();
    }
  }

  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  scrfd_ts.calbr_basetime(base_time);
  scrfd_ts.build_timeline("scrfd test");
  scrfd_ts.show_summary("scrfd test");
  scrfd_ts.clear();

  return 0;
}
