//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef BMNN_H
#define BMNN_H

#include <iostream>
#include <string>
#include <memory>
#include <cassert>

#include "bmruntime_interface.h"
#include "bmruntime_cpp.h"
// #include "bm_wrapper.hpp"

/*
 * Any class that inherits this class cannot be assigned.
 */
class NoCopyable {
  protected:
    NoCopyable() =default;
    ~NoCopyable() = default;
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable& rhs)= delete;
};

/*
 * Help user managing network to do inference.
 * Feat 1. Create and free device memory of output tensors automatically.
 *      2. Device memory of input tensors must be provided outside, and will not be freed here.
 *      3. Print Network information.
 */
class BModelNetwork : public NoCopyable {
  bm_tensor_t* m_inputTensors;
  bm_tensor_t* m_outputTensors;
  bm_handle_t  m_handle;
  void *m_bmrt;
  bool is_soc;
  int m_batch;

  public:
  const bm_net_info_t *m_netinfo;

  BModelNetwork(void *bmrt, const std::string& name);
  ~BModelNetwork(){
    delete[] m_inputTensors;
    delete[] m_outputTensors;
  };
  
  bool get_is_soc(){
    return is_soc;
  }
  bool batch(){
    return m_batch;
  }

  bm_tensor_t* get_input_tensor(int idx);
  bm_tensor_t* get_output_tensor(int idx);

  int forward(bm_device_mem_t* input_mem, bm_device_mem_t* output_mem);

};

/*
 * Help user managing handles and networks of a bmodel, using class instances above.
 */
class BModelContext : public NoCopyable {
  bm_handle_t m_handle;
  void *m_bmrt;
  std::vector<std::string> m_network_names;

  public:
  BModelContext(int dev_id, std::string bmodel_file);
  ~BModelContext() ;

  bm_handle_t handle() {
    return m_handle;
  }

  void load_network_names();

  std::string network_name(int index){
    if (index >= (int)m_network_names.size()) {
      return "Invalid index";
    }
    return m_network_names[index];
  }

  std::shared_ptr<BModelNetwork> network(const std::string& net_name)
  {
    return std::make_shared<BModelNetwork>(m_bmrt, net_name);
  }

  std::shared_ptr<BModelNetwork> network(int net_index) {
    assert(net_index < (int)m_network_names.size());
    return std::make_shared<BModelNetwork>(m_bmrt, m_network_names[net_index]);
  }

};

#endif 
