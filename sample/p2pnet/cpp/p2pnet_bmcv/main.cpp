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
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "p2pnet.hpp"
using namespace std;

static void save_imgs(bm_handle_t h, P2Pnet *p2p, 
                      const std::vector<std::vector<PPoint>> &points,
                      const vector<bm_image> &batch_imgs,
                      const vector<string> &batch_names,
                      string save_folder)
{
    cout<<"batch_imgs.size(): "<<batch_imgs.size()<<std::endl;
    for (size_t i = 0; i < batch_imgs.size(); i++)
    {
        std::cout<<batch_names[i]<<std::endl;
        // each pic
        bm_image img = batch_imgs[i];
        cout<<points[i].size()<<std::endl;
        for (size_t j = 0; j < points[i].size(); j++)
        {
            // cv::circle(img, Point(points[i][j].x, points[i][j].y), 1, 
            //     cv::Scalar(255, 0, 0), 3);
            p2p->draw_bmcv(h, int(points[i][j].x), int(points[i][j].y), img);
        }

        // save image
        string save_name = save_folder + "/" + batch_names[i];
        std::cout<<save_name<<std::endl;
        // imwrite(save_name, img);
        void* jpeg_data = NULL;
        size_t out_size = 0;
        int ret = bmcv_image_jpeg_enc(h, 1, (bm_image*)&batch_imgs[i], &jpeg_data, &out_size);
        if (ret == BM_SUCCESS) {
          FILE *fp = fopen(save_name.c_str(), "wb");
          fwrite(jpeg_data, out_size, 1, fp);
          fclose(fp);
        }
        free(jpeg_data);
        bm_image_destroy(batch_imgs[i]);

        //write txt
        string tmp = ".txt";
        string txt_name = batch_names[i];
        txt_name.replace(batch_names[i].find(".jpg"), 4, tmp);
        string txt_file = save_folder + "/" + txt_name;
        ofstream out;
        out.open(txt_file, std::ios::out | std::ios::trunc);
            cout<<points[i].size()<<std::endl;
        for (size_t j = 0; j < points[i].size(); j++)
        {
            out << points[i][j].x << " " << points[i][j].y << "\n";
        }
        out.close();
    }
}

int main(int argc, char *argv[]){
  cout.setf(ios::fixed);
  // get params
  const char *keys="{bmodel | ../../models/BM1684X/p2pnet_1684x_int8_1b.bmodel | bmodel file path}"
    "{dev_id | 0 | TPU device id}"
    "{help | 0 | print help information.}"
    "{input | ../../video/video.avi | input path, images direction or video file path}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  int dev_id = parser.get<int>("dev_id");

  // check params
  struct stat info;
  if (stat(bmodel_file.c_str(), &info) != 0) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0){
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
  P2Pnet p2p(bm_ctx);
  CV_Assert(0 == p2p.Init());

  // profiling
  TimeStamp p2p_ts;
  TimeStamp *ts = &p2p_ts;
  p2p.enableProfile(&p2p_ts);

  // get batch_size
  int batch_size = p2p.batch_size();

  // creat save path
  string save_foler = "results";
  if (access(save_foler.c_str(), 0) != F_OK)
    mkdir(save_foler.c_str(), S_IRWXU);
  
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

    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    vector<PPointVec> points;
    int cn = files_vector.size();
    int id = 0;
    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter; 
      id++;
      cout << id << "/" << cn << ", img_file: " << img_file << endl;
      ts->save("read image");
      // cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
      bm_image bmimg;
      picDec(h, img_file.c_str(), bmimg);
      ts->save("read image");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);
      if ((int)batch_imgs.size() == batch_size){
        // predict
        CV_Assert(0 == p2p.Detect(batch_imgs, points));

        for(int i = 0; i < batch_size; i++){
          if (batch_imgs[i].image_format != 0){
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }
        }
        save_imgs(h, &p2p, points, batch_imgs, batch_names, save_foler);
        batch_imgs.clear();
        batch_names.clear();
        points.clear();
      }
    }
  }
  
  // test video
  else {
    VideoDecFFM decoder;
    decoder.openDec(&h, input.c_str());
    int id = 0;
    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    vector<PPointVec> points;
    while(true){
      bm_image *img = decoder.grab();
      if (!img)
        break;
      batch_imgs.push_back(*img);
      if ((int)batch_imgs.size() == batch_size) {
        CV_Assert(0 == p2p.Detect(batch_imgs, points));
        for(int i = 0; i < batch_size; i++){
          id++;
          batch_names.push_back(to_string(id) + ".jpg");
          cout << id << ", det_nums: " << points[i].size() << endl;
          if (batch_imgs[i].image_format != 0){
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }
          save_imgs(h, &p2p, points, batch_imgs, batch_names, save_foler);
        }
        batch_imgs.clear();
        batch_names.clear();
        points.clear();
      }
    }
  }
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  p2p_ts.calbr_basetime(base_time);
  p2p_ts.build_timeline("p2p test");
  p2p_ts.show_summary("p2p test");
  p2p_ts.clear();

  return 0;
}
