//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
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

#include "ff_decode.hpp"
#include "p2pnet.hpp"

void draw_imgs(bm_handle_t h, P2Pnet* p2p,
                      const std::vector<std::vector<PPoint>>& points,
                      const std::vector<bm_image>& batch_imgs,
                      const std::vector<std::string>& batch_names,
                      std::string save_folder) {
  for (size_t i = 0; i < batch_imgs.size(); i++) {
    bm_image img = batch_imgs[i];
    for (size_t j = 0; j < points[i].size(); j++) {
      p2p->draw_bmcv(h, int(points[i][j].x), int(points[i][j].y), img);
    }
    // save image
    std::string save_name = save_folder + "/" + batch_names[i];
    void* jpeg_data = NULL;
    size_t out_size = 0;
    int ret = bmcv_image_jpeg_enc(h, 1, (bm_image*)&batch_imgs[i], &jpeg_data,
                                  &out_size);
    if (ret == BM_SUCCESS) {
      FILE* fp = fopen(save_name.c_str(), "wb");
      fwrite(jpeg_data, out_size, 1, fp);
      fclose(fp);
    }
    free(jpeg_data);
    bm_image_destroy(batch_imgs[i]);

    // write txt
    std::string txt_name = batch_names[i];
    txt_name.replace(batch_names[i].find(".jpg"), 4, ".txt");
    std::string txt_file = save_folder + "/" + txt_name;
    std::ofstream out;
    out.open(txt_file, std::ios::out | std::ios::trunc);
    for (size_t j = 0; j < points[i].size(); j++) {
      out << points[i][j].x << " " << points[i][j].y << "\n";
    }
    out.close();
  }
}

int main(int argc, char* argv[]) {
  std::cout.setf(std::ios::fixed);
  // get params
  const char* keys =
      "{bmodel | ../../models/BM1684X/p2pnet_1684x_int8_1b.bmodel | bmodel "
      "file path}"
      "{dev_id | 0 | TPU device id}"
      "{help | 0 | print help information.}"
      "{input | ../../video/video.avi | input path, images direction or video "
      "file path}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  std::string bmodel_file = parser.get<std::string>("bmodel");
  std::string input = parser.get<std::string>("input");
  int dev_id = parser.get<int>("dev_id");

  // check params
  struct stat info;
  if (stat(bmodel_file.c_str(), &info) != 0) {
    std::cout << "Cannot find valid model file." << std::endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0) {
    std::cout << "Cannot find input path." << std::endl;
    exit(1);
  }

  // creat handle
  BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
  std::cout << "set device id: " << dev_id << std::endl;
  bm_handle_t h = handle->handle();

  // load bmodel
  std::shared_ptr<BMNNContext> bm_ctx =
      std::make_shared<BMNNContext>(handle, bmodel_file.c_str());

  // initialize net
  P2Pnet p2pnet(bm_ctx);
  CV_Assert(0 == p2pnet.Init());

  // profiling
  TimeStamp p2pnet_ts;
  TimeStamp* ts = &p2pnet_ts;
  p2pnet.enableProfile(&p2pnet_ts);

  // get batch_size
  int batch_size = p2pnet.batch_size();

  // creat save path
  std::string save_foler = "results";
  if (access(save_foler.c_str(), 0) != F_OK) mkdir(save_foler.c_str(), S_IRWXU);

  // test images
  if (info.st_mode & S_IFDIR) {
    // get files
    std::vector<std::string> files_vector;
    DIR* pDir;
    struct dirent* ptr;
    pDir = opendir(input.c_str());
    while ((ptr = readdir(pDir)) != 0) {
      if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
        files_vector.push_back(input + "/" + ptr->d_name);
      }
    }
    closedir(pDir);

    std::vector<bm_image> batch_imgs;
    std::vector<std::string> batch_names;
    std::vector<PPointVec> points;
    int id = 0;
    for (std::vector<std::string>::iterator iter = files_vector.begin();
         iter != files_vector.end(); iter++) {
      std::string img_file = *iter;
      ts->save("decode time");
      bm_image bmimg;
      picDec(h, img_file.c_str(), bmimg);
      ts->save("decode time");
      size_t index = img_file.rfind("/");
      std::string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);

      iter++;
      bool end_flag = (iter == files_vector.end());
      iter--;
      if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
        CV_Assert(0 == p2pnet.Detect(batch_imgs, points));
        for (int i = 0; i < batch_size; i++) {
          if (batch_imgs[i].image_format != 0) {
            id++;
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width,
                            FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
            std::cout << id << ", point_nums: " << points[i].size()
                      << std::endl;
          }
        }
        draw_imgs(h, &p2pnet, points, batch_imgs, batch_names, save_foler);
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
    std::vector<bm_image> batch_imgs;
    std::vector<std::string> batch_names;
    std::vector<PPointVec> points;
    int id = 0;
    bool endFlag = false;
    while (!endFlag) {
      ts->save("decode time");
      bm_image* img = decoder.grab();
      ts->save("decode time");
      if (!img) {
        endFlag = true;
      } else {
        batch_imgs.push_back(*img);
      }
      if (batch_imgs.size() == batch_size || (endFlag && batch_imgs.size())) {
        CV_Assert(0 == p2pnet.Detect(batch_imgs, points));
        for (int i = 0; i < batch_imgs.size(); i++) {
          id++;
          batch_names.push_back(std::to_string(id) + ".jpg");
          std::cout << id << ", point_nums: " << points[i].size() << std::endl;
          if (batch_imgs[i].image_format != 0) {
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width,
                            FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }
        }
        draw_imgs(h, &p2pnet, points, batch_imgs, batch_names, save_foler);
        batch_imgs.clear();
        batch_names.clear();
        points.clear();
      }
    }
  }
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  p2pnet_ts.calbr_basetime(base_time);
  p2pnet_ts.build_timeline("P2PNet test");
  p2pnet_ts.show_summary("P2PNet test");
  p2pnet_ts.clear();

  return 0;
}
