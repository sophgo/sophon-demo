//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include "ssd.hpp"

using namespace std;

static void detect(bm_handle_t         &bm_handle,
                   SSD                 &net,
                   vector<cv::Mat>     &images,
                   vector<string>      &batch_names,
                   TimeStamp           *ts) {

  vector<vector<ObjRect>> detections;

  vector<bm_image> input_img_bmcv;
  ts->save("attach input");
  bm_image_from_mat(bm_handle, images, input_img_bmcv);
  ts->save("attach input");

  ts->save("detection");
  net.preForward (input_img_bmcv);

  // do inference
  net.forward();

  net.postForward (input_img_bmcv , detections);
  ts->save("detection");
  // destory bm_image
  for (size_t i = 0; i < input_img_bmcv.size();i++) {
    bm_image_destroy (input_img_bmcv[i]);
  }

  // save results to jpg file
  for (size_t i = 0; i < detections.size(); i++) {
    for (size_t j = 0; j < detections[i].size(); j++) {
      ObjRect rect = detections[i][j];
      cv::rectangle(images[i], cv::Rect(rect.x1, rect.y1, rect.x2 - rect.x1 + 1,
                                    rect.y2 - rect.y1 + 1), cv::Scalar(255, 0, 0), 2);
    }

    // check result directory
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    // jpg encode
    if (net.getPrecision()) {
      cv::imwrite("results/out-batch-int8-" + batch_names[i], images[i]);
    } else {
      cv::imwrite("results/out-batch-fp32-" + batch_names[i], images[i]);
    }
  }
}

int main(int argc, char **argv) {

  cout.setf(ios::fixed);

  // sanity check
  if (argc != 6) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image file> <bmodel path> <test count> <device id>" << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel path> <test count> <device id>" << endl;
    exit(1);
  }
  struct stat info;
  string bmodel_file = argv[3];
  if (stat(bmodel_file.c_str(), &info) != 0) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  if (strcmp(argv[1], "video") != 0 && strcmp(argv[1], "image") != 0){
    cout << "mode must be image or video" << endl;
    exit(1);
  }

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0)
    is_video = true;

  string input_url = argv[2];
  if (stat(input_url.c_str(), &info) != 0) {
    cout << "Cannot find input image path." << endl;
    exit(1);
  }

  unsigned long test_loop = stoul(string(argv[4]), nullptr, 0);
  if (test_loop < 1) {
    std::cout << "test_loop must large 0" << std::endl;
    exit(1);
  }

  // set device id
  std::string dev_str = argv[5];
  std::stringstream checkdevid(dev_str);
  double t;
  if (!(checkdevid >> t)) {
    std::cout << "Is not a valid dev ID: " << dev_str << std::endl;
    exit(1);
  }
  int dev_id = std::stoi(dev_str);
  std::cout << "set device id:"  << dev_id << std::endl;

  // profiling
  TimeStamp ssd_ts;
  TimeStamp *ts = &ssd_ts;

  // initialize handle of low level device
    int max_dev_id = 0;
    bm_dev_getcount(&max_dev_id);
    if (dev_id >= max_dev_id) {
        std::cout << "ERROR: Input device id=" << dev_id
                  << " exceeds the maximum number " << max_dev_id << std::endl;
        exit(-1);
    }
    bm_handle_t  bm_handle;
  bm_status_t ret = bm_dev_request (&bm_handle, dev_id);
  if (ret != BM_SUCCESS) {
    cout << "Initialize bm handle failed, ret = " << ret << endl;
    exit(-1);
  }

  // initialize SSD class
  SSD net(bm_handle , bmodel_file);

  // for profiling
  net.enableProfile(ts);
  int batch_size = net.batch_size();
  vector<cv::Mat> batch_imgs;
  vector<string> batch_names;
  // decode and detect
  if (!is_video) {

    for (uint32_t i = 0; i < test_loop; i++) {
      ts->save("ssd overall");
      ts->save("read image");

      // decode jpg file to Mat object
      cv::Mat img = cv::imread(input_url, cv::IMREAD_COLOR, dev_id);
      ts->save("read image");

      // do detect
      string img_out = "t_" + to_string(i) + "_dev_" + to_string(dev_id) + "_image.jpg";
      batch_imgs.push_back(img);
      batch_names.push_back(img_out);
      if ((int)batch_imgs.size() == batch_size) {
        detect(bm_handle, net, batch_imgs, batch_names, ts);
        batch_imgs.clear();
        batch_names.clear();
      }
      ts->save("ssd overall");
    }

  } else {

    // open stream
    cv::VideoCapture cap(input_url, cv::CAP_ANY, dev_id);
    if (!cap.isOpened()) {
      cout << "open stream " << input_url << " failed!" << endl;
      exit(1);
    }

    // get resolution
    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cout << "resolution of input stream: " << h << "," << w << endl;
    // set output format to YUVi420
    cap.set(cv::CAP_PROP_OUTPUT_YUV, 1.0);

    for (uint32_t c = 0; c < test_loop; c++) {

      // get one frame from decoder
      cv::Mat *p_img = new cv::Mat;
      cap.read(*p_img);

      // sanity check
      if (p_img->avRows() != h || p_img->avCols() != w) {
        if (p_img != nullptr) delete p_img;
        continue;
      }
      //cv::Mat *input;

      //cv::resize(*p_img,*input,cv::Size(300,300));
      // do detct
      string img_out = "t_" + to_string(c) + "_dev_" + to_string(dev_id)  + "_video.jpg";
      batch_imgs.push_back(*p_img);
      batch_names.push_back(img_out);
      if ((int)batch_imgs.size() == batch_size) {
        detect(bm_handle, net, batch_imgs, batch_names, ts);
        batch_imgs.clear();
        batch_names.clear();
      }
      // release Mat object
      if (p_img != nullptr) delete p_img;
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ssd_ts.calbr_basetime(base_time);
  ssd_ts.build_timeline("ssd detect");
  ssd_ts.show_summary("detect ");
  ssd_ts.clear();

  bm_dev_free(bm_handle);

  cout << endl;

  return 0;
}
