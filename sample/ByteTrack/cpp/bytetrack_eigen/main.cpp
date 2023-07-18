//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.    All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <regex>

#include "bytetrack.h"
#include "draw_utils.hpp"
#include "ff_decode.hpp"
#include "json.hpp"
#include "yolov5.hpp"
#define USE_OPENCV_DRAW_BOX 1
using json = nlohmann::json;
using namespace std;

void bytetrack_yaml_parse(const std::string& config_path,
                          bytetrack_params& params) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open config file: " << config_path << std::endl;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream ss(line);
    std::string key, value;
    std::getline(ss, key, ':');
    std::getline(ss, value);
    key = key.substr(key.find_first_not_of(" \t\r\n"));
    value.erase(value.find_last_not_of(" \t\r\n") + 1);
    if (key == "CONF_THRE") {
      std::istringstream iss(value);
      iss >> params.conf_thresh;
    } else if (key == "NMS_THRE") {
      std::istringstream iss(value);
      iss >> params.nms_thresh;
    } else if (key == "TRACK_THRESH") {
      std::istringstream iss(value);
      iss >> params.track_thresh;
    } else if (key == "MATCH_THRESH") {
      std::istringstream iss(value);
      iss >> params.match_thresh;
    } else if (key == "TRACK_BUFFER") {
      std::istringstream iss(value);
      iss >> params.track_buffer;
    } else if (key == "FRAME_RATE") {
      std::istringstream iss(value);
      iss >> params.frame_rate;
    } else if (key == "MIN_BOX_AREA") {
      std::istringstream iss(value);
      iss >> params.min_box_area;
    }
  }
}

vector<string> stringSplit(const string& str, char delim) {
  string s;
  s.append(1, delim);
  regex reg(s);
  vector<string> elems(sregex_token_iterator(str.begin(), str.end(), reg, -1),
                       sregex_token_iterator());
  return elems;
}
bool check_path(string file_path, vector<string> correct_postfixes) {
  auto index = file_path.rfind('.');
  string postfix = file_path.substr(index + 1);
  if (find(correct_postfixes.begin(), correct_postfixes.end(), postfix) !=
      correct_postfixes.end()) {
    return true;
  } else {
    cout << "skipping path: " << file_path << ", please check your dataset!"
         << endl;
    return false;
  }
};
void getAllFiles(string path, vector<string>& files,
                 vector<string> correct_postfixes) {
  DIR* dir;
  struct dirent* ptr;
  if ((dir = opendir(path.c_str())) == NULL) {
    perror("Open dri error...");
    exit(1);
  }
  while ((ptr = readdir(dir)) != NULL) {
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
      continue;
    else if (ptr->d_type == 8 &&
             check_path(path + "/" + ptr->d_name, correct_postfixes))  // file
      files.push_back(path + "/" + ptr->d_name);
    else if (ptr->d_type == 10)  // link file
      continue;
    else if (ptr->d_type == 4) {
      // files.push_back(ptr->d_name);//dir
      getAllFiles(path + "/" + ptr->d_name, files, correct_postfixes);
    }
  }
  closedir(dir);
}

int main(int argc, char* argv[]) {
  cout.setf(ios::fixed);
  // get params
  const char* keys =
      "{bmodel_detector | "
      "../../models/BM1684X/yolov5s_v6.1_3output_int8_4b.bmodel | detector "
      "bmodel file path}"
      "{dev_id | 0 | TPU device id}"
      "{help | 0 | print help information.}"
      "{input | ../../datasets/test_car_person_1080P.mp4 | input path, video "
      "file path or image folder}"
      "{classnames | ./coco.names | class names file path}"
      "{config | ./bytetrack.yaml | config params}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  std::string bmodel_detector = parser.get<std::string>("bmodel_detector");
  std::string input = parser.get<std::string>("input");
  std::string classnames = parser.get<std::string>("classnames");
  std::string config = parser.get<std::string>("config");
  int dev_id = parser.get<int>("dev_id");
  bytetrack_params params;
  bytetrack_yaml_parse(config, params);

  // check params
  struct stat info;
  if (stat(bmodel_detector.c_str(), &info) != 0) {
    cout << "Cannot find valid detector model file." << endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0) {
    cout << "Cannot find input path." << endl;
    exit(1);
  }

  // creat handle
  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: " << dev_id << endl;
  bm_handle_t h = handle->handle();

  // load bmodel
  shared_ptr<BMNNContext> bm_ctx_detector =
      make_shared<BMNNContext>(handle, bmodel_detector.c_str());

  // initialize net
  YoloV5 yolov5(bm_ctx_detector);
  yolov5.Init(params.conf_thresh, params.nms_thresh, classnames);
  BYTETracker bytetrack(params);

  // profiling
  TimeStamp bytetrack_ts;
  TimeStamp* ts = &bytetrack_ts;
  yolov5.enableProfile(&bytetrack_ts);
  bytetrack.enableProfile(&bytetrack_ts);

  // get batch_size
  int batch_size = yolov5.batch_size();

  // creat save path
  if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
  if (access("results/images", 0) != F_OK) mkdir("results/images", S_IRWXU);
  if (access("results/video", 0) != F_OK) mkdir("results/video", S_IRWXU);
  if (access("results/mot_eval", 0) != F_OK) mkdir("results/mot_eval", S_IRWXU);

  // initialize data buffer.
  vector<bm_image> batch_imgs;
  vector<YoloV5BoxVec> yolov5_boxes;
  int id = 0;

  // TODO: merge image folder and video code
  //  test images
  vector<string> image_paths;
  VideoDecFFM decoder;
  string save_image_path;
  string save_result_path = "results/mot_eval/";
  ofstream mot_saver;
  auto stringBuffer = stringSplit(input, '/');
  auto bmodel_detector_name_buffer = stringSplit(bmodel_detector, '/');
  auto bmodel_detector_name =
      bmodel_detector_name_buffer[bmodel_detector_name_buffer.size() - 1];
  if (info.st_mode & S_IFDIR) {
    vector<string> correct_postfixes = {"jpg", "png"};
    getAllFiles(input, image_paths, correct_postfixes);
    sort(image_paths.begin(), image_paths.end());
    save_image_path = "results/images/";
    if (stringBuffer[stringBuffer.size() - 1] == "img1")
      save_result_path += stringBuffer[stringBuffer.size() - 2] + "_" +
                          bmodel_detector_name + ".txt";
    else
      save_result_path += stringBuffer[stringBuffer.size() - 1] + "_" +
                          bmodel_detector_name + ".txt";
  } else {
    decoder.openDec(&h, input.c_str());
    save_image_path = "results/video/";
    save_result_path += stringBuffer[stringBuffer.size() - 1] + "_" +
                        bmodel_detector_name + ".txt";
  }

  mot_saver.open(save_result_path, ios::out);
  bool end_flag = false;
  int ind = 0;
  while (!end_flag) {
    ts->save("decode time");
    if (info.st_mode & S_IFDIR) {
      if (ind >= image_paths.size()) {
        end_flag = true;
        break;
      }
      bm_image img;
      picDec(h, image_paths[ind].c_str(), img);
      end_flag = (ind == (image_paths.size() - 1));
      batch_imgs.push_back(img);
    } else {
      bm_image* img;
      img = decoder.grab();
      if (!img) {
        end_flag = true;
      } else {
        batch_imgs.push_back(*img);
      }
    }
    ind++;
    ts->save("decode time");
    if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
      ts->save("yolov5 time");
      CV_Assert(0 == yolov5.Detect(batch_imgs, yolov5_boxes));
      ts->save("yolov5 time");
      for (int i = 0; i < batch_imgs.size(); i++) {
        id++;
        // tracker, directly output tracked boxes.
        ts->save("bytetrack time");
        vector<STrack> output_stracks = bytetrack.update(yolov5_boxes[i]);
        ts->save("bytetrack time");

        ts->save("encode time");
        for (auto track_box : output_stracks) {
          string save_str = cv::format("%d,%d,%f,%f,%f,%f,1,-1,-1,-1\n", id,
                                       track_box.track_id, track_box.tlwh[0],
                                       track_box.tlwh[1], track_box.tlwh[2],
                                       track_box.tlwh[3]);
          mot_saver << save_str;
        }
        cout << id << ", detect_nums: " << yolov5_boxes[i].size()
             << "; track_nums: " << output_stracks.size() << endl;

#if USE_OPENCV_DRAW_BOX
        cv::Mat frame_to_draw;
        cv::bmcv::toMAT(&batch_imgs[i], frame_to_draw);
        for (auto bbox : output_stracks) {
          draw_opencv(bbox.track_id, bbox.class_id, bbox.score, bbox.tlwh[0],
                      bbox.tlwh[1], bbox.tlwh[0] + bbox.tlwh[2],
                      bbox.tlwh[1] + bbox.tlwh[3], frame_to_draw);
        }
        bm_image frame_drawed;
        cv::bmcv::toBMI(frame_to_draw, &frame_drawed);
        if (frame_drawed.image_format != FORMAT_YUV420P) {
          bm_image frame;
          bm_image_create(h, frame_drawed.height, frame_drawed.width,
                          FORMAT_YUV420P, frame_drawed.data_type, &frame);
          bmcv_image_storage_convert(h, 1, &frame_drawed, &frame);
          bm_image_destroy(frame_drawed);
          frame_drawed = frame;
        }
        string img_file = save_image_path + to_string(id) + ".jpg";
        void* jpeg_data = NULL;
        size_t out_size = 0;
        int ret =
            bmcv_image_jpeg_enc(h, 1, &frame_drawed, &jpeg_data, &out_size);
        bm_image_destroy(frame_drawed);
#else
        for (auto bbox : track_boxes[i]) {
          draw_bmcv(h, bbox.track_id, bbox.class_id, bbox.score, bbox.x, bbox.y,
                    bbox.width, bbox.height, batch_imgs[i], true);
        }
        if (batch_imgs[i].image_format != 0) {
          bm_image frame;
          bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width,
                          FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
          bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
          bm_image_destroy(batch_imgs[i]);
          batch_imgs[i] = frame;
        }
        string img_file = save_image_path + to_string(id) + ".jpg";
        void* jpeg_data = NULL;
        size_t out_size = 0;
        int ret =
            bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
#endif
        if (ret == BM_SUCCESS) {
          FILE* fp = fopen(img_file.c_str(), "wb");
          fwrite(jpeg_data, out_size, 1, fp);
          fclose(fp);
        }
        free(jpeg_data);
        ts->save("encode time");
        bm_image_destroy(batch_imgs[i]);
      }
      batch_imgs.clear();
      yolov5_boxes.clear();
    }
  }
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  bytetrack_ts.calbr_basetime(base_time);
  bytetrack_ts.build_timeline("ByteTtrack test");
  bytetrack_ts.show_summary("ByteTrack test");
  bytetrack_ts.clear();
  mot_saver.close();
  return 0;
}
