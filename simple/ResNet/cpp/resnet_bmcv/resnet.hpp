#ifndef RESNET_HPP
#define RESNET_HPP

#include <string>
#include <iostream>
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#include "utils.hpp"

#define MAX_BATCH 4


class RESNET {
public:
  RESNET(bm_handle_t bm_handle, const std::string bmodel);
  ~RESNET();
  void preForward(std::vector<bm_image> &input);
  void forward();
  void postForward(std::vector<bm_image> &input, std::vector<std::pair<int, float>> &results);
  void enableProfile(TimeStamp *ts);
  int batch_size();
private:
  void preprocess_bmcv (std::vector<bm_image> &input);

  // handle of runtime contxt
  void *p_bmrt_;

  // handle of low level device 
  bm_handle_t bm_handle_;

  // model info 
  const bm_net_info_t *net_info_;
  const char **net_names_;
  int net_h_;
  int net_w_;
  int batch_size_;
  int num_channels_;
  int class_num_;

  // indicate current bmodel type INT8 or FP32
  bool input_is_int8_;
  bool output_is_int8_;

  // buffer of inference results
  float *output_fp32;
  int8_t *output_int8;

  // input image shape used for inference call
  bm_shape_t input_shape_;

  float input_scale_;
  float output_scale_;
  int count_per_img;

  // bm image objects for storing intermediate results
  bm_image resize_bmcv_[MAX_BATCH];
  bm_image linear_trans_bmcv_[MAX_BATCH];

  // linear transformation arguments of BMCV
  bmcv_convert_to_attr linear_trans_param_;

  // for profiling
  TimeStamp *ts_ = NULL;
};

#endif /* RESNET_HPP */
