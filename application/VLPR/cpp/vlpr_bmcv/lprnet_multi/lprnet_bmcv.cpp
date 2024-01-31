//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "lprnet_bmcv.hpp"

// rec class
static char const* arr_chars[] = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",  "2",  "3",  "4",
    "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",
    "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",
    "V",  "W",  "X",  "Y",  "Z",  "I",  "O",  "-"};

int argmax(float* data, int num) {
  float max_value = -1e10;
  int max_index = 0;
  for (int i = 0; i < num; ++i) {
    float value = data[i];
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }
  return max_index;
}

LPRNet::LPRNet(int dev_id, std::string bmodel_path, int pre_thread_num,
               int post_thread_num, int queue_size)
    : dev_id(dev_id),
      pre_thread_num(pre_thread_num),
      post_thread_num(post_thread_num),
      bmodel_path(bmodel_path),
      m_queue_decode("lprnet_decode_queue", queue_size),
      m_queue_pre("lprnet_pre_queue", queue_size),
      m_queue_infer("lprnet_infer_queue", queue_size),
      m_queue_post("lprnet_post_queue", queue_size) {
  // end flag, decode_activate_thread_num is arbitrary number, and need to set
  // bigger 0
  decode_activate_thread_num = 1;
  pre_activate_thread_num = pre_thread_num;
  infer_activate_thread_num = 1;
  post_activate_thread_num = post_thread_num;

  // context
  m_ctx = std::make_unique<bmruntime::Context>(dev_id);
  assert(BM_SUCCESS == m_ctx->load_bmodel(bmodel_path.c_str()));

  // network
  std::vector<const char*> net_names;
  m_ctx->get_network_names(&net_names);
  m_net = std::make_unique<bmruntime::Network>(*m_ctx, net_names[0], 0);
  m_handle = m_ctx->handle();
  m_inputs = m_net->Inputs();
  m_outputs = m_net->Outputs();
  m_net_h = m_inputs[0]->tensor()->shape.dims[2];
  m_net_w = m_inputs[0]->tensor()->shape.dims[3];
  batch_size = m_inputs[0]->tensor()->shape.dims[0];
  class_num = m_outputs[0]->tensor()->shape.dims[1];
  seq_len = m_outputs[0]->tensor()->shape.dims[2];
  m_netinfo = m_ctx->get_network_info(net_names[0]);
  struct bm_misc_info misc_info;
  assert(BM_SUCCESS == bm_get_misc_info(m_handle, &misc_info));
  is_soc = misc_info.pcie_soc_mode == 1;

  // preprocess interval variables
  m_resized_imgs.resize(pre_thread_num);
  for (int i = 0; i < pre_thread_num; i++) {
    m_resized_imgs[i].resize(batch_size);
    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int j = 0; j < batch_size; j++) {
      auto ret = bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_BGR_PLANAR,
                                 DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i][j],
                                 strides);
      assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(batch_size, m_resized_imgs[i].data());
  }
  img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (m_inputs[0]->tensor()->dtype == BM_INT8) {
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }

  // converto params
  input_scale = m_netinfo->input_scales[0];
  output_scale = m_netinfo->output_scales[0];
  input_scale = input_scale * 0.0078125;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = -127.5 * input_scale;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = -127.5 * input_scale;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = -127.5 * input_scale;
}

LPRNet::~LPRNet() {
  // sync thread
  for (std::thread& t : threads) t.join();
  // free dev mem and struct mem
  for (int i = 0; i < pre_thread_num; i++) {
    bm_image_free_contiguous_mem(batch_size, m_resized_imgs[i].data());
    for (int j = 0; j < batch_size; j++) {
      bm_image_destroy(m_resized_imgs[i][j]);
    }
  }
}

void LPRNet::run() {
  threads.clear();
  auto start = std::chrono::steady_clock::now();
  // start threads
  for (int p = 0; p < pre_thread_num; p++)
    threads.emplace_back(&LPRNet::preprocess, this, p);
  threads.emplace_back(&LPRNet::inference, this);
  for (int p = 0; p < post_thread_num; p++)
    threads.emplace_back(&LPRNet::postprocess, this, p);
}

void LPRNet::preprocess(int process_id) {
  while (true) {
    int batch_idx = 0;
    std::shared_ptr<bmtensor> cur_tensor = std::make_shared<bmtensor>();

    // get batch data
    while (true) {
      std::shared_ptr<bmimage> in;
      if (m_queue_decode.pop_front(in) != 0) break;
      // resize
      bm_image image_aligned;
      bool need_copy = in->bmimg->width & (64 - 1);
      if (need_copy) {
        int stride1[3], stride2[3];
        bm_image_get_stride(*(in->bmimg), stride1);
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);
        bm_image_create(m_handle, in->bmimg->height, in->bmimg->width,
                        in->bmimg->image_format, in->bmimg->data_type,
                        &image_aligned, stride2);

        bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
        bmcv_copy_to_atrr_t copyToAttr;
        memset(&copyToAttr, 0, sizeof(copyToAttr));
        copyToAttr.start_x = 0;
        copyToAttr.start_y = 0;
        copyToAttr.if_padding = 1;

        bmcv_image_copy_to(m_handle, copyToAttr, *(in->bmimg), image_aligned);
      } else {
        image_aligned = *(in->bmimg.get());
      }
      auto ret = bmcv_image_vpp_convert(m_handle, 1, image_aligned,
                                        &m_resized_imgs[process_id][batch_idx]);
      assert(BM_SUCCESS == ret);
      batch_idx++;

      cur_tensor->channel_ids.push_back(in->channel_id);
      cur_tensor->frame_ids.push_back(in->frame_id);
      if (need_copy) bm_image_destroy(image_aligned);
      if (batch_idx == batch_size) break;
    }
    // if not have any image, exit thread
    if (batch_idx == 0) break;

    // converto
    bm_image* m_converto_imgs = new bm_image[batch_size];
    // init images
    for (int i = 0; i < batch_size; i++)
      bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_BGR_PLANAR, img_dtype,
                      &m_converto_imgs[i]);
    assert(BM_SUCCESS ==
           bm_image_alloc_contiguous_mem(batch_size, m_converto_imgs));
    CV_Assert(0 == bmcv_image_convert_to(m_handle, batch_size, converto_attr,
                                         m_resized_imgs[process_id].data(),
                                         m_converto_imgs));

    // get contiguous device mem in batch images
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(batch_size, m_converto_imgs,
                                       &input_dev_mem);

    // init output tensor
    cur_tensor->bmtensor = std::make_shared<bm_tensor_t>();
    cur_tensor->bmtensor.reset(
        new bm_tensor_t(), [this, m_converto_imgs](bm_tensor_t* p) {
          assert(BM_SUCCESS == bm_image_free_contiguous_mem(this->batch_size,
                                                            m_converto_imgs));
          for (int i = 0; i < this->batch_size; i++)
            bm_image_destroy(m_converto_imgs[i]);

          delete[] m_converto_imgs;
          delete p;
          p = nullptr;
        });
    cur_tensor->bmtensor->device_mem = input_dev_mem;
    cur_tensor->bmtensor->shape = m_inputs[0]->tensor()->shape;
    cur_tensor->bmtensor->dtype = m_inputs[0]->tensor()->dtype;
    cur_tensor->bmtensor->st_mode = m_inputs[0]->tensor()->st_mode;

    // push data
    m_queue_pre.push_back(cur_tensor);
  }

  std::unique_lock<std::mutex> lock(m_mutex_pre_end);
  pre_activate_thread_num--;
  if (pre_activate_thread_num <= 0) m_queue_pre.set_stop_flag(true);
  std::cout << "preprocess thread " << process_id << " exit..." << std::endl;
}

void LPRNet::inference() {
  while (true) {
    std::shared_ptr<bmtensor> in;
    if (m_queue_pre.pop_front(in) != 0) break;

    // set dev mem to network inference's input
    m_inputs[0]->set_device_mem(in->bmtensor->device_mem);

    // init output tensor
    std::shared_ptr<bmtensor> cur_tensor = std::make_shared<bmtensor>();
    cur_tensor->channel_ids = in->channel_ids;
    cur_tensor->frame_ids = in->frame_ids;
    cur_tensor->bmtensor = std::make_shared<bm_tensor_t>();
    bm_tensor_t* out_tensor = new bm_tensor_t();

    // compute output size
    size_t output_size = 0;
    for (int s = 0; s < m_netinfo->stage_num; s++) {
      size_t out_size =
          bmrt_shape_count(&m_netinfo->stages[s].output_shapes[0]);
      if (output_size < out_size) {
        output_size = out_size;
      }
    }
    if (BM_FLOAT32 == m_netinfo->output_dtypes[0]) output_size *= 4;

    // malloc output dev mem
    assert(BM_SUCCESS == bm_malloc_device_byte(
                             m_handle, &(out_tensor->device_mem), output_size));
    cur_tensor->bmtensor.reset(out_tensor, [this](bm_tensor_t* p) {
      if (p->device_mem.u.device.device_addr != 0) {
        bm_free_device(this->get_handle(), p->device_mem);
      }
      delete p;
      p = nullptr;
    });
    cur_tensor->bmtensor->shape = m_outputs[0]->tensor()->shape;
    cur_tensor->bmtensor->dtype = m_outputs[0]->tensor()->dtype;
    cur_tensor->bmtensor->st_mode = m_outputs[0]->tensor()->st_mode;
    m_outputs[0]->set_device_mem(cur_tensor->bmtensor->device_mem);

    // forward
    assert(BM_SUCCESS == m_net->Forward(true));

    // push data
    m_queue_infer.push_back(cur_tensor);
  }

  std::unique_lock<std::mutex> lock(m_mutex_infer_end);
  infer_activate_thread_num--;
  if (infer_activate_thread_num <= 0) m_queue_infer.set_stop_flag(true);
  std::cout << "inference thread exit..." << std::endl;
}

void LPRNet::postprocess(int process_id) {
  while (true) {
    std::shared_ptr<bmtensor> in;
    if (m_queue_infer.pop_front(in) != 0) break;

    int valid_batch_size = in->channel_ids.size();
    // get cpu mem data
    std::shared_ptr<float> cpu_data = get_cpu_data(in->bmtensor);
    float* output_data = nullptr;
    float ptr[class_num];
    int pred_num[seq_len];
    for (int idx = 0; idx < valid_batch_size; ++idx) {
      std::shared_ptr<rec_data> result(new rec_data());
      result->channel_id = in->channel_ids[idx];
      result->frame_id = in->frame_ids[idx];
      output_data = cpu_data.get() + idx * seq_len * class_num;
      for (int j = 0; j < seq_len; j++) {
        for (int k = 0; k < class_num; k++) {
          ptr[k] = *(output_data + k * seq_len + j);
        }
        int class_id = argmax(&ptr[0], class_num);
        float confidence = ptr[class_id];
        pred_num[j] = class_id;
      }
      result->rec_res = get_res(pred_num);

      // push data
      m_queue_post.push_back(result);
    }
  }

  std::unique_lock<std::mutex> lock(m_mutex_post_end);
  post_activate_thread_num--;
  if (post_activate_thread_num <= 0) m_queue_post.set_stop_flag(true);
  std::cout << "postprocess thread " << process_id << " exit..." << std::endl;
}

std::shared_ptr<float> LPRNet::get_cpu_data(
    std::shared_ptr<bm_tensor_t> m_tensor) {
  bm_status_t ret;
  float* pFP32 = nullptr;
  int count = bmrt_shape_count(&m_tensor->shape);

  // in SOC mode, device mem can be mapped to host memory, faster then using d2s
  if (is_soc) {
    if (m_tensor->dtype == BM_FLOAT32) {
      unsigned long long addr;
      ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
      assert(BM_SUCCESS == ret);
      pFP32 = (float*)addr;
    } else if (BM_INT8 == m_tensor->dtype) {
      int8_t* pI8 = nullptr;
      unsigned long long addr;
      ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
      assert(BM_SUCCESS == ret);
      pI8 = (int8_t*)addr;

      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      for (int i = 0; i < count; ++i) {
        pFP32[i] = pI8[i] * output_scale;
      }
      ret = bm_mem_unmap_device_mem(
          m_handle, pI8, bm_mem_get_device_size(m_tensor->device_mem));
      assert(BM_SUCCESS == ret);
    } else if (m_tensor->dtype == BM_INT32) {
      int32_t* pI32 = nullptr;
      unsigned long long addr;
      ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
      assert(BM_SUCCESS == ret);
      pI32 = (int32_t*)addr;
      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      for (int i = 0; i < count; ++i) {
        pFP32[i] = pI32[i] * output_scale;
      }
      ret = bm_mem_unmap_device_mem(
          m_handle, pI32, bm_mem_get_device_size(m_tensor->device_mem));
      assert(BM_SUCCESS == ret);
    } else {
      std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
    }
  } else {
    // the common method using d2s
    if (m_tensor->dtype == BM_FLOAT32) {
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem,
                                  count * sizeof(float));
      assert(BM_SUCCESS == ret);
    } else if (BM_INT8 == m_tensor->dtype) {
      int8_t* pI8 = nullptr;
      int tensor_size = bmrt_tensor_bytesize(m_tensor.get());
      pI8 = new int8_t[tensor_size];
      assert(pI8 != nullptr);

      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      ret = bm_memcpy_d2s_partial(m_handle, pI8, m_tensor->device_mem,
                                  tensor_size);
      assert(BM_SUCCESS == ret);
      for (int i = 0; i < count; ++i) {
        pFP32[i] = pI8[i] * output_scale;
      }
      delete[] pI8;
    } else if (m_tensor->dtype == BM_INT32) {
      int32_t* pI32 = nullptr;
      int tensor_size = bmrt_tensor_bytesize(m_tensor.get());
      pI32 = new int32_t[tensor_size];
      assert(pI32 != nullptr);

      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      ret = bm_memcpy_d2s_partial(m_handle, pI32, m_tensor->device_mem,
                                  tensor_size);
      assert(BM_SUCCESS == ret);
      for (int i = 0; i < count; ++i) {
        pFP32[i] = pI32[i] * output_scale;
      }
      delete[] pI32;

    } else {
      std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
    }
  }

  // define destructor for different platform
  std::shared_ptr<float> res = std::make_shared<float>();
  res.reset(pFP32, [this, m_tensor](float* p) {
    if (!(this->is_soc && m_tensor->dtype == BM_FLOAT32)) {
      delete[] p;
      p = nullptr;
    } else {
      bm_status_t ret = bm_mem_unmap_device_mem(
          this->m_handle, p, bmrt_tensor_bytesize(m_tensor.get()));
      assert(BM_SUCCESS == ret);
    }
  });

  return res;
}

std::string LPRNet::get_res(int pred_num[]) {
  int no_repeat_blank[20];
  int cn_no_repeat_blank = 0;
  int pre_c = pred_num[0];
  if (pre_c != class_num - 1) {
    no_repeat_blank[0] = pre_c;
    cn_no_repeat_blank++;
  }
  for (int i = 0; i < seq_len; i++) {
    if (pred_num[i] == pre_c) continue;
    if (pred_num[i] == class_num - 1) {
      pre_c = pred_num[i];
      continue;
    }
    no_repeat_blank[cn_no_repeat_blank] = pred_num[i];
    pre_c = pred_num[i];
    cn_no_repeat_blank++;
  }

  std::string res = "";
  for (int j = 0; j < cn_no_repeat_blank; j++) {
    res = res + arr_chars[no_repeat_blank[j]];
  }

  return res;
}

void LPRNet::push_m_queue_decode(std::shared_ptr<bmimage> in) {
  m_queue_decode.push_back(in);
}

void LPRNet::set_preprocess_exit() {
  std::unique_lock<std::mutex> lock(m_mutex_decode_end);
  decode_activate_thread_num = 0;
  m_queue_decode.set_stop_flag(true);
}

int LPRNet::pop_m_queue_post(std::shared_ptr<rec_data>& out) {
  if (m_queue_post.pop_front(out) != 0) return -1;
  return 0;
}