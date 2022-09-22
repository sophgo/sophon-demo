//
//  bmodel_base.hpp
//  NelivaSDK
//
//  Created by Bitmain on 2020/11/5.
//  Copyright © 2020年 AnBaolei. All rights reserved.
//

#ifndef bmodel_base_hpp
#define bmodel_base_hpp

#include <cstring>
#include "bmruntime_interface.h"
#define USE_OPENCV
#include "bm_wrapper.hpp"

class BmodelBase {

public:
  BmodelBase(){};
  BmodelBase(const std::string bmodel, int device_id){};
  ~BmodelBase();
  bm_handle_t get_bm_handle();
  int batch_size();

protected:
  void forward();
  void load_model();

protected:
  std::vector<void*> outputs_;
  std::vector<int> output_sizes_;
  const char **net_names_;
  void *p_bmrt_;
  bm_net_info_t* net_info_;
  bm_handle_t bm_handle_;
  bmcv_convert_to_attr convert_attr_;
  bm_shape_t input_shape_;
  bm_image* resize_bmcv_;
  bm_image* scaled_inputs_;
  int net_h_;
  int net_w_;
  int output_num_;
  int batch_size_;
  int num_channels_;
  std::string bmodel_path_;
  int device_id_ = 0;
  bm_image_data_format_ext data_type_ = DATA_TYPE_EXT_1N_BYTE_SIGNED;
};

#endif /* bmodel_base_hpp */
