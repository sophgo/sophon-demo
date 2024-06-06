#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"


void dump_yuv(std::string filename, AVFrame* frame, int card_id){
 
  bm_device_mem_t mems[3];

  int size0 = frame->height * frame->linesize[0];
  mems[0] = bm_mem_from_device((unsigned long long)frame->data[4], size0);
  int size1 = size0/4;
  mems[1] = bm_mem_from_device((unsigned long long)frame->data[5], size1);
  int size2 = size0/4;
  mems[2] = bm_mem_from_device((unsigned long long)frame->data[6], size2);
    
  char* buffer0 = new char[size0];
  char* buffer1 = new char[size1];
  char* buffer2 = new char[size2];

  bm_handle_t handle;
  bm_dev_request(&handle, card_id);
  bm_memcpy_d2s(handle, buffer0, mems[0]);
  bm_memcpy_d2s(handle, buffer1, mems[1]);
  bm_memcpy_d2s(handle, buffer2, mems[2]);

  std::ofstream outFile(filename);
  outFile.write(buffer0, size0);
  outFile.write(buffer1, size1);
  outFile.write(buffer2, size2);
  
  std::cout << "height: " << frame->height << ", stride: " << frame->linesize[0] << std::endl;

  delete[] buffer0;
  delete[] buffer1;
  delete[] buffer2;
}


int main(int argc, char *argv[]){

  std::string filename;
  int dev_id = 0;
  auto output_file = "output.yuv";

#if BMCV_VERSION_MAJOR > 1
  int heap_mask = 2;
#else
  int heap_mask = 4;
#endif

  if (argc < 2) {
    std::cout << "usage:" << std::endl;
    std::cout << "\t" << argv[0] << " <input_pic> [dev_id]" << std::endl;
    std::cout << "params:" << std::endl;
    std::cout << "\t" << "input_pic: picture for decode to get bgr mat" << std::endl;
    std::cout << "\t" << "dev_id:  using device id" << std::endl;
    exit(1);
  }

  filename = argv[1];
  if (argc == 3)
    dev_id = atoi(argv[2]);


  auto bgr_mat = cv::imread(filename, cv::IMREAD_COLOR, dev_id);
  int height = bgr_mat.rows;
  int width = bgr_mat.cols;

  // 创建的用于构造yuv mat的avframe在vpu上，以进行后续编码操作
  AVFrame* f = cv::av::create(height, width, BM_MAKEFLAG(0, heap_mask, dev_id));

  cv::Mat new_mat(f, BM_MAKEFLAG(cv::UMatData::AVFRAME_ATTACHED, heap_mask, dev_id));
  bm_status_t ret = cv::bmcv::convert(bgr_mat, new_mat, true);
  if (ret != BM_SUCCESS){
    std::cerr << "bgrmat to yuvmat failed" << std::endl;
    exit(1);
  }

  AVFrame * frame = new_mat.u->frame;


  // 使用avframe的其他操作
  dump_yuv(output_file, frame, dev_id);

  av_frame_free(&frame);
  frame = nullptr;
  
}