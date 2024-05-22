#include <iostream>

#include "opencv2/opencv.hpp"
#include "ff_video_decode.h"

int avframe_to_cvmat1(std::string filename, int dev_id, cv::Mat &bgr_mat){
  int ret;
  VideoDec_FFMPEG reader;
  ret = reader.openDec(filename.c_str(), 1, NULL, 100, dev_id);
  if (ret < 0){
    std::cerr << "Avframe: Open input media failed" << std::endl;
    return -1;
  }

  AVFrame *picture = av_frame_alloc();
  reader.grabFrame(picture);

  // 生成yuv mat，此时mat释放的同时会释放内部的avframe
  cv::Mat ocv_frame(picture, dev_id);

  // 生成bgr mat，update参数决定是否将设备内存数据同步到系统内存，
  // 如果需要其他硬件操作的时候可以不同步来节省时间，等需要系统内存操作再同步
  cv::bmcv::toMAT(ocv_frame, bgr_mat, true);

  // 此时yuv mat释放会自动调用av_frame_free()来释放内部的avframe
  ocv_frame.release();
  picture = nullptr;

  return 0;
}

int avframe_to_cvmat2(std::string filename, int dev_id, cv::Mat &bgr_mat){
  int ret;
  VideoDec_FFMPEG reader;
  ret = reader.openDec(filename.c_str(), 1, NULL, 100, dev_id);
  if (ret < 0){
    std::cerr << "Avframe: Open input media failed" << std::endl;
    return -1;
  }

  AVFrame *picture = av_frame_alloc();
  reader.grabFrame(picture);

  // 生成yuv mat，此时mat的释放和avframe是相互独立的，avframe需要单独释放
  cv::Mat ocv_frame(picture, dev_id | BM_MAKEFLAG(cv::UMatData::AVFRAME_ATTACHED, 0, 0));

  cv::bmcv::toMAT(ocv_frame, bgr_mat, true);

  // yuv mat获取其中的avframe
  AVFrame * avframe = ocv_frame.u->frame;

  // 需要单独调用av_frame_free释放yuv mat内部的avframe
  av_frame_free(&avframe);
  ocv_frame.release();
  picture = nullptr;

  return 0;
}

int main(int argc, char *argv[]){
  std::string filename;
  int dev_id = 0;

  if (argc < 2) {
    std::cout << "usage:" << std::endl;
    std::cout << "\t" << argv[0] << " <input_video> [dev_id] " << std::endl;
    std::cout << "params:" << std::endl;
    std::cout << "\t" << "input_video: video for decode to get avframe" << std::endl;
    std::cout << "\t" << "dev_id:  using device id, default 0" << std::endl;
    exit(1);
  }

  filename = argv[1];
  if (argc == 3)
    dev_id = atoi(argv[2]);

  cv::Mat bgr_mat1, bgr_mat2;
  
  avframe_to_cvmat1(filename, dev_id, bgr_mat1);
  cv::imwrite("dump1.jpg", bgr_mat1);
  avframe_to_cvmat2(filename, dev_id, bgr_mat2);
  cv::imwrite("dump2.jpg", bgr_mat2);



}