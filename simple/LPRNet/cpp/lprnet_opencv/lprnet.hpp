#ifndef LPRNET_HPP
#define LPRNET_HPP

#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "bmruntime_interface.h"
#include "utils.hpp"

#define MAX_BATCH 4
#define INPUT_WIDTH 94
#define INPUT_HEIGHT 24
#define BUFFER_SIZE (1024 * 500)

//char * get_res(int pred_num[], int len_char, int clas_char);

class LPRNET {
public:
  LPRNET(const std::string bmodel, int dev_id);
  ~LPRNET();
  void preForward(const std::vector<cv::Mat> &images);
  void forward();
  void postForward(std::vector<std::string> &detections);
  void enableProfile(TimeStamp *ts);
  int batch_size();
private:
  void setMean(std::vector<float> &values);
  void wrapInputLayer(std::vector<cv::Mat>* input_channels,const int batch_id);
  void preprocess(const cv::Mat &img, std::vector<cv::Mat>* input_channels);
  
  // handle of low level device 
  bm_handle_t bm_handle_;
  int dev_id_;

  // runtime helper
  const char **net_names_;
  void *p_bmrt_;

  // network input shape
  int batch_size_;
  int num_channels_;
  int net_h_;
  int net_w_;

  // network related parameters
  cv::Mat mean_;

  // input & output buffers
  bm_tensor_t input_tensor_;
  bm_tensor_t output_tensor_;
  float input_scale;
  float output_scale;
  float *input_f32;
  int8_t *input_int8;
  float *output_f32;
  int8_t *output_int8;
  bool int8_flag_;
  bool int8_output_flag;
  int count_per_img;
  int len_char;
  int clas_char;
  // for profiling
  TimeStamp *ts_;
};

#endif /* LPRNET_HPP */
