/*==========================================================================
 * Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
============================================================================*/
#ifndef YOLOV5_DEMO_BMNN_UTILS_H
#define YOLOV5_DEMO_BMNN_UTILS_H

#include <iostream>
#include <string>
#include <memory>

#include "bmruntime_interface.h"
#include "bm_wrapper.hpp"

class NoCopyable {
  protected:
    NoCopyable() =default;
    ~NoCopyable() = default;
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable& rhs)= delete;
};

class BMNNTensor{
  /**
   *  members from bm_tensor {
   *  bm_data_type_t dtype;
   bm_shape_t shape;
   bm_device_mem_t device_mem;
   bm_store_mode_t st_mode;
   }
   */
  bm_handle_t  m_handle;

  std::string m_name;
  float *m_cpu_data;
  float m_scale;
  bm_tensor_t *m_tensor;

  bool can_mmap;

  public:
  BMNNTensor(bm_handle_t handle, const char *name, float scale,
      bm_tensor_t* tensor, bool can_mmap):m_handle(handle), m_name(name),
  m_cpu_data(nullptr),m_scale(scale), m_tensor(tensor), can_mmap(can_mmap) {
  }

  virtual ~BMNNTensor() {
    if (m_cpu_data == NULL) return;
    if(can_mmap && BM_FLOAT32 == m_tensor->dtype) {
      int tensor_size = bm_mem_get_device_size(m_tensor->device_mem);
      bm_status_t ret = bm_mem_unmap_device_mem(m_handle, m_cpu_data, tensor_size);
      assert(BM_SUCCESS == ret);
    } else {
      delete [] m_cpu_data;
    }
  }

  int set_device_mem(bm_device_mem_t *mem){
    this->m_tensor->device_mem = *mem;
    return 0;
  }

  const bm_device_mem_t* get_device_mem() {
    return &this->m_tensor->device_mem;
  }

  float *get_cpu_data() {
    if(m_cpu_data) return m_cpu_data;
    bm_status_t ret;
    float *pFP32 = nullptr;
    int count = bmrt_shape_count(&m_tensor->shape);
    // in SOC mode, device mem can be mapped to host memory, faster then using d2s
    if(can_mmap){
      if (m_tensor->dtype == BM_FLOAT32) {
        unsigned long long  addr;
        ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
        assert(BM_SUCCESS == ret);
        ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
        assert(BM_SUCCESS == ret);
        pFP32 = (float*)addr;
      } else if (BM_INT8 == m_tensor->dtype) {
        int8_t * pI8 = nullptr;
        unsigned long long  addr;
        ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
        assert(BM_SUCCESS == ret);
        ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
        assert(BM_SUCCESS == ret);
        pI8 = (int8_t*)addr;

        // dtype convert
        pFP32 = new float[count];
        assert(pFP32 != nullptr);
        for(int i = 0;i < count; ++ i) {
          pFP32[i] = pI8[i] * m_scale;
        }
        ret = bm_mem_unmap_device_mem(m_handle, pI8, bm_mem_get_device_size(m_tensor->device_mem));
        assert(BM_SUCCESS == ret);
      } else{
        std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
      }
    } else {
      // the common method using d2s
      if (m_tensor->dtype == BM_FLOAT32) {
        pFP32 = new float[count];
        assert(pFP32 != nullptr);
        ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem, count * sizeof(float));
        assert(BM_SUCCESS ==ret);
      } else if (BM_INT8 == m_tensor->dtype) {
        int8_t * pI8 = nullptr;
        int tensor_size = bmrt_tensor_bytesize(m_tensor);
        pI8 = new int8_t[tensor_size];
        assert(pI8 != nullptr);

        // dtype convert
        pFP32 = new float[count];
        assert(pFP32 != nullptr);
        ret = bm_memcpy_d2s_partial(m_handle, pI8, m_tensor->device_mem, tensor_size);
        assert(BM_SUCCESS ==ret);
        for(int i = 0;i < count; ++ i) {
          pFP32[i] = pI8[i] * m_scale;
        }
        delete [] pI8;
      } else{
        std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
      }
    }

    m_cpu_data = pFP32;
    return m_cpu_data;
  }

  const bm_shape_t* get_shape() {
    return &m_tensor->shape;
  }

  bm_data_type_t get_dtype() {
    return m_tensor->dtype;
  }

  float get_scale() {
    return m_scale;
  }

  void set_shape(const int* shape, int dims) {
    m_tensor->shape.num_dims = dims;
    for(int i=0; i<dims; i++){
      m_tensor->shape.dims[i] = shape[i];
    }
  }
  void set_shape_by_dim(int dim, int size){
    assert(m_tensor->shape.num_dims>dim);
    m_tensor->shape.dims[dim] = size;
  }

};

class BMNNNetwork : public NoCopyable {
  const bm_net_info_t *m_netinfo;
  bm_tensor_t* m_inputTensors;
  bm_tensor_t* m_outputTensors;
  bm_handle_t  m_handle;
  void *m_bmrt;
  bool is_soc;
  std::set<int> m_batches;
  int m_max_batch;

  std::unordered_map<std::string, bm_tensor_t*> m_mapInputs;
  std::unordered_map<std::string, bm_tensor_t*> m_mapOutputs;

  public:
  BMNNNetwork(void *bmrt, const std::string& name):m_bmrt(bmrt) {
    m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
    m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
    m_max_batch = -1;
    std::vector<int> batches;
    for(int i=0; i<m_netinfo->stage_num; i++){
      batches.push_back(m_netinfo->stages[i].input_shapes[0].dims[0]);
      if(m_max_batch<batches.back()){
        m_max_batch = batches.back();
      }
    }
    m_batches.insert(batches.begin(), batches.end());
    m_inputTensors = new bm_tensor_t[m_netinfo->input_num];
    m_outputTensors = new bm_tensor_t[m_netinfo->output_num];
    for(int i = 0; i < m_netinfo->input_num; ++i) {
      m_inputTensors[i].dtype = m_netinfo->input_dtypes[i];
      m_inputTensors[i].shape = m_netinfo->stages[0].input_shapes[i];
      m_inputTensors[i].st_mode = BM_STORE_1N;
      // input device mem should be provided outside, such as from image's contiguous mem
      m_inputTensors[i].device_mem = bm_mem_null();
    }

    for(int i = 0; i < m_netinfo->output_num; ++i) {
      m_outputTensors[i].dtype = m_netinfo->output_dtypes[i];
      m_outputTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
      m_outputTensors[i].st_mode = BM_STORE_1N;
      
      // alloc as max size to reuse device mem, avoid to alloc and free everytime
      size_t max_size=0;
			for(int s=0; s<m_netinfo->stage_num; s++){
         size_t out_size = bmrt_shape_count(&m_netinfo->stages[s].output_shapes[i]);
         if(max_size<out_size){
            max_size = out_size;
         }
      }
      if(BM_FLOAT32 == m_netinfo->output_dtypes[i]) max_size *= 4;
      auto ret =  bm_malloc_device_byte(m_handle, &m_outputTensors[i].device_mem, max_size);
			assert(BM_SUCCESS == ret);
    }
    struct bm_misc_info misc_info;
    bm_status_t ret = bm_get_misc_info(m_handle, &misc_info);
    assert(BM_SUCCESS == ret);
    is_soc = misc_info.pcie_soc_mode == 1;

    printf("*** Run in %s mode ***\n", is_soc?"SOC": "PCIE");

    //assert(m_netinfo->stage_num == 1);
    showInfo();
  }

  ~BMNNNetwork() {
    //Free input tensors
    delete [] m_inputTensors;
    //Free output tensors
    for(int i = 0; i < m_netinfo->output_num; ++i) {
      if (m_outputTensors[i].device_mem.size != 0) {
        bm_free_device(m_handle, m_outputTensors[i].device_mem);
      }
    }
    delete []m_outputTensors;
  }

  int maxBatch() const {
    return m_max_batch;
  }
  int get_nearest_batch(int real_batch){
      for(auto batch: m_batches){
          if(batch>=real_batch){
             return batch;
					}
      }
      assert(0);
      return m_max_batch;
  }

  std::shared_ptr<BMNNTensor> inputTensor(int index){
    assert(index < m_netinfo->input_num);
    return std::make_shared<BMNNTensor>(m_handle, m_netinfo->input_names[index],
        m_netinfo->input_scales[index], &m_inputTensors[index], is_soc);
  }

  int outputTensorNum() {
    return m_netinfo->output_num;
  }

  std::shared_ptr<BMNNTensor> outputTensor(int index){
    assert(index < m_netinfo->output_num);
    return std::make_shared<BMNNTensor>(m_handle, m_netinfo->output_names[index],
        m_netinfo->output_scales[index], &m_outputTensors[index], is_soc);
  }

  int forward() {

    bool user_mem = false; // if false, bmrt will alloc mem every time.
    if (m_outputTensors->device_mem.size != 0) {
      // if true, bmrt don't alloc mem again.
      user_mem = true;
    }

    bool ok=bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,
        m_outputTensors, m_netinfo->output_num, user_mem, false);
    if (!ok) {
      std::cout << "bm_launch_tensor() failed=" << std::endl;
      return -1;
    }

#if 0
    for(int i = 0;i < m_netinfo->output_num; ++i) {
      auto tensor = m_outputTensors[i];
      // dump
      std::cout << "output_tensor [" << i << "] size=" << bmrt_tensor_device_size(&tensor) << std::endl;
    }
#endif

    return 0;
  }

  static std::string shape_to_str(const bm_shape_t& shape) {
    std::string str ="[ ";
    for(int i=0; i<shape.num_dims; i++){
      str += std::to_string(shape.dims[i]) + " ";
    }
    str += "]";
    return str;
  }

  void showInfo()
  {
    const char* dtypeMap[] = {
      "FLOAT32",
      "FLOAT16",
      "INT8",
      "UINT8",
      "INT16",
      "UINT16",
      "INT32",
      "UINT32",
    };
    printf("\n########################\n");
    printf("NetName: %s\n", m_netinfo->name);
    for(int s=0; s<m_netinfo->stage_num; s++){
      printf("---- stage %d ----\n", s);
      for(int i=0; i<m_netinfo->input_num; i++){
        auto shapeStr = shape_to_str(m_netinfo->stages[s].input_shapes[i]);
        printf("  Input %d) '%s' shape=%s dtype=%s scale=%g\n",
            i,
            m_netinfo->input_names[i],
            shapeStr.c_str(),
            dtypeMap[m_netinfo->input_dtypes[i]],
            m_netinfo->input_scales[i]);
      }
      for(int i=0; i<m_netinfo->output_num; i++){
        auto shapeStr = shape_to_str(m_netinfo->stages[s].output_shapes[i]);
        printf("  Output %d) '%s' shape=%s dtype=%s scale=%g\n",
            i,
            m_netinfo->output_names[i],
            shapeStr.c_str(),
            dtypeMap[m_netinfo->output_dtypes[i]],
            m_netinfo->output_scales[i]);
      }
    }
    printf("########################\n\n");

  }

};

class BMNNHandle: public NoCopyable {
  bm_handle_t m_handle;
  int m_dev_id;
  public:
  BMNNHandle(int dev_id=0):m_dev_id(dev_id) {
    int ret = bm_dev_request(&m_handle, dev_id);
    assert(BM_SUCCESS == ret);
  }

  ~BMNNHandle(){
    bm_dev_free(m_handle);
  }

  bm_handle_t handle() {
    return m_handle;
  }

  int dev_id() {
    return m_dev_id;
  }
};

using BMNNHandlePtr = std::shared_ptr<BMNNHandle>;

class BMNNContext : public NoCopyable {
  BMNNHandlePtr m_handlePtr;
  void *m_bmrt;
  std::vector<std::string> m_network_names;

  public:
  BMNNContext(BMNNHandlePtr handle, const char* bmodel_file):m_handlePtr(handle){
    bm_handle_t hdev = m_handlePtr->handle();
    m_bmrt = bmrt_create(hdev);
    if (NULL == m_bmrt) {
      std::cout << "bmrt_create() failed!" << std::endl;
      exit(-1);
    }

    if (!bmrt_load_bmodel(m_bmrt, bmodel_file)) {
      std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
    }

    load_network_names();


  }

  ~BMNNContext() {
    if (m_bmrt!=NULL) {
      bmrt_destroy(m_bmrt);
      m_bmrt = NULL;
    }
  }

  bm_handle_t handle() {
    return m_handlePtr->handle();
  }

  void* bmrt() {
    return m_bmrt;
  }

  void load_network_names() {
    const char **names;
    int num;
    num = bmrt_get_network_number(m_bmrt);
    bmrt_get_network_names(m_bmrt, &names);
    for(int i=0;i < num; ++i) {
      m_network_names.push_back(names[i]);
    }

    free(names);
  }

  std::string network_name(int index){
    if (index >= (int)m_network_names.size()) {
      return "Invalid index";
    }

    return m_network_names[index];
  }

  std::shared_ptr<BMNNNetwork> network(const std::string& net_name)
  {
    return std::make_shared<BMNNNetwork>(m_bmrt, net_name);
  }

  std::shared_ptr<BMNNNetwork> network(int net_index) {
    assert(net_index < (int)m_network_names.size());
    return std::make_shared<BMNNNetwork>(m_bmrt, m_network_names[net_index]);
  }

};

#endif
