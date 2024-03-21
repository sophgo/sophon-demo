// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "bmodel_utils.h"

BModelNetwork::BModelNetwork(void *bmrt, const std::string& name):m_bmrt(bmrt) {
    m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
    m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
    m_batch = m_netinfo->stages[0].input_shapes[0].dims[0];

    m_inputTensors = new bm_tensor_t[m_netinfo->input_num];
    m_outputTensors = new bm_tensor_t[m_netinfo->output_num];
    for(int i = 0; i < m_netinfo->input_num; ++i) {
        m_inputTensors[i].dtype = m_netinfo->input_dtypes[i];
        m_inputTensors[i].shape = m_netinfo->stages[0].input_shapes[i];
        m_inputTensors[i].st_mode = BM_STORE_1N;
    }

    for(int i = 0; i < m_netinfo->output_num; ++i) {
        m_outputTensors[i].dtype = m_netinfo->output_dtypes[i];
        m_outputTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
        m_outputTensors[i].st_mode = BM_STORE_1N;
    }

    struct bm_misc_info misc_info;
    bm_status_t ret = bm_get_misc_info(m_handle, &misc_info);
    assert(BM_SUCCESS == ret);
    is_soc = misc_info.pcie_soc_mode == 1;

    printf("*** Run in %s mode ***\n", is_soc?"SOC": "PCIE");
}


bm_tensor_t* BModelNetwork::get_input_tensor(int idx){
    if(idx >= m_netinfo->input_num){
      printf("Error! input tensor idx is too big!");
      return nullptr;
    }
    return m_inputTensors+idx;
}

bm_tensor_t* BModelNetwork::get_output_tensor(int idx){
    if(idx >= m_netinfo->output_num){
        printf("Error! output tensor idx is too big!");
        return nullptr;
    }
    return m_outputTensors+idx;
}


int BModelNetwork::forward(bm_device_mem_t* input_mem, bm_device_mem_t* output_mem) {
  
    for(int i=0;i<m_netinfo->input_num;i++){
        m_inputTensors[i].device_mem=input_mem[i];
    }
    for(int i=0;i<m_netinfo->output_num;i++){
        m_outputTensors[i].device_mem=output_mem[i];
    }

    bool ok=bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,
        m_outputTensors, m_netinfo->output_num, true, true);
    if (!ok) {
        std::cout << "bm_launch_tensor() failed=" << std::endl;
        return -1;
    }
    bool status = bm_thread_sync(m_handle);
    assert(BM_SUCCESS == status);

    return 0;
}



BModelContext::BModelContext(int dev_id, std::string bmodel_file){
    bm_dev_request(&m_handle,dev_id);
    m_bmrt = bmrt_create(m_handle);
    if (NULL == m_bmrt) {
        std::cout << "bmrt_create() failed!" << std::endl;
        exit(-1);
    }
    if (!bmrt_load_bmodel(m_bmrt, bmodel_file.c_str())) {
        std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
    }
    load_network_names();
}


BModelContext::~BModelContext() {
    if (m_bmrt!=NULL) {
    bmrt_destroy(m_bmrt);
    m_bmrt = NULL;
    bm_dev_free(m_handle);
    }
}


void BModelContext::load_network_names() {
    const char **names;
    int num;
    num = bmrt_get_network_number(m_bmrt);
    bmrt_get_network_names(m_bmrt, &names);
    for(int i=0;i < num; ++i) {
        m_network_names.push_back(names[i]);
    }
    free(names);
}