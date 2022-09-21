#ifndef SSD_HPP
#define SSD_HPP

#include <string>

// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"

#include "utils.hpp"

#define MAX_BATCH 4

struct ObjRect {
  unsigned int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
};

class SSD {
public:
  SSD(bm_handle_t& bm_handle, const std::string bmodel);
  ~SSD();
  void preForward(std::vector<bm_image> &input);
  void forward();
  void postForward(std::vector<bm_image> &input, std::vector<std::vector<ObjRect>> &detections);
  void enableProfile(TimeStamp *ts);
  bool getPrecision();

private:
  void preprocess_bmcv (std::vector<bm_image> &input);

  // handle of runtime contxt
  void *p_bmrt_;

  // handle of low level device 
  bm_handle_t bm_handle_;

  // model info 
  const bm_net_info_t *net_info_;

  // indicate current bmodel type INT8 or FP32
  bool is_int8_;

  // confidence 
  float threshold_;

  // buffer of inference results
  float *output_;

  // input image shape used for inference call
  bm_shape_t input_shape_;

  // bm image objects for storing intermediate results
  bm_image resize_bmcv_[MAX_BATCH];
  bm_image linear_trans_bmcv_[MAX_BATCH];

  // crop arguments of BMCV
  bmcv_rect_t crop_rect_;

  // linear transformation arguments of BMCV
  bmcv_convert_to_attr linear_trans_param_;

  // for profiling
  TimeStamp *ts_;
};

#endif /* SSD_HPP */
