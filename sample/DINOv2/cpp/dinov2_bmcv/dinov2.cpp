//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include "dinov2.hpp"

using namespace std;

DINOV2::DINOV2(const std::string &bmodel_file, const std::string &head_file, int dev_id) {
  std::cout << "DINOV2(bmcv) ctor .." << std::endl;
  load_head(head_file);

  // 1. get handle
  auto ret = bm_dev_request(&handle, dev_id);
  assert(BM_SUCCESS == ret);

  // 2. judge now is pcie or soc
  ret = bm_get_misc_info(handle, &misc_info);
  assert(BM_SUCCESS == ret);

  // 3. load bmodel
  bmrt = bmrt_create(handle);
  if (NULL == bmrt) {
    throw std::runtime_error("bmrt_create() failed!");
  }
  if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
    throw std::runtime_error("load bmodel(" + bmodel_file + ") failed");
  }

  // 4. get network names from bmodel
  const char **names;
  int num = bmrt_get_network_number(bmrt);
  if (num > 1) {
    std::cout << "This bmodel have " << num << " networks, and this program will only take network 0." << std::endl;
  }
  bmrt_get_network_names(bmrt, &names);
  for (int i = 0; i < num; ++i) {
    network_names.push_back(names[i]);
  }
  free(names);

  // 5. get netinfo by netname
  netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
  if (netinfo->stage_num > 1) {
    std::cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << std::endl;
  }

  // 6. model info
  max_batch = -1;
  for (int i = 0; i < netinfo->stage_num; i++) {
    int b = netinfo->stages[i].input_shapes[0].dims[0];
    m_batches.push_back(b);
    if (max_batch < b) max_batch = b;
  }
  std::sort(m_batches.begin(), m_batches.end());
  m_net_h = netinfo->stages[0].input_shapes[0].dims[2];
  m_net_w = netinfo->stages[0].input_shapes[0].dims[3];
  // 官方 make_classification_eval_transform() 默认口径：Resize(短边=256) + CenterCrop(224)
  m_resize_size = 256;
  if (m_net_h != 224 || m_net_w != 224) {
    std::cerr << "WARNING: DINOv2 expects 224x224 input, got " << m_net_h << "x" << m_net_w << std::endl;
  }
  feature_dim = netinfo->stages[0].output_shapes[0].dims[1];
  assert(feature_dim == 1536);

  // 7. input tensor（device mem 由 bm_image 连续内存供给，此处置空）
  input_tensor.dtype = netinfo->input_dtypes[0];
  input_tensor.shape = netinfo->stages[0].input_shapes[0];
  input_tensor.st_mode = BM_STORE_1N;
  input_tensor.device_mem = bm_mem_null();

  // 8. initialize bmimages（stride 按 64 对齐；RGB_PLANAR 每像素 1 字节）
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int planar_strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  m_cropped_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  for (int i = 0; i < max_batch; i++) {
    if (bm_image_create(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                        DATA_TYPE_EXT_1N_BYTE, &m_cropped_imgs[i], planar_strides) != BM_SUCCESS) {
      throw std::runtime_error("bm_image_create cropped failed");
    }
  }
  bm_image_alloc_contiguous_mem(max_batch, m_cropped_imgs.data());

  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (netinfo->input_dtypes[0] == BM_INT8) {
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  if (bm_image_create_batch(handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR,
                            img_dtype, m_converto_imgs.data(), max_batch) != BM_SUCCESS) {
    throw std::runtime_error("bm_image_create_batch converto failed");
  }

  // 9. normalize converto（(x/255 - mean) / std）
  float input_scale = netinfo->input_scales[0];
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  converto_attr.alpha_0 = 1 / (255. * std[0]) * input_scale;
  converto_attr.alpha_1 = 1 / (255. * std[1]) * input_scale;
  converto_attr.alpha_2 = 1 / (255. * std[2]) * input_scale;
  converto_attr.beta_0 = (-mean[0] / std[0]) * input_scale;
  converto_attr.beta_1 = (-mean[1] / std[1]) * input_scale;
  converto_attr.beta_2 = (-mean[2] / std[2]) * input_scale;
}

int DINOV2::load_head(const std::string &head_file) {
  // 布局：weight[class_num * feature_dim] float32 + bias[class_num] float32（行主序）
  std::ifstream fin(head_file, std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "ERROR: cannot open linear head file: " << head_file << std::endl;
    return -1;
  }
  const int kClassNum = 1000;
  const int kFeatDim = 1536;
  size_t weight_bytes = (size_t)kClassNum * kFeatDim * sizeof(float);
  size_t bias_bytes = (size_t)kClassNum * sizeof(float);
  m_head_weight.resize(kClassNum * kFeatDim);
  m_head_bias.resize(kClassNum);
  fin.read((char*)m_head_weight.data(), weight_bytes);
  fin.read((char*)m_head_bias.data(), bias_bytes);
  fin.close();
  class_num = kClassNum;
  feature_dim = kFeatDim;
  std::cout << "loaded linear head: " << class_num << " x " << feature_dim
            << " from " << head_file << std::endl;
  return 0;
}

void DINOV2::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}

int DINOV2::batch_size() {
  return max_batch;
}

DINOV2::~DINOV2() {
  std::cout << "DINOV2(bmcv) dtor ..." << std::endl;
  // 注意：input_tensor.device_mem 若已绑定 m_converto_imgs 的连续内存，由
  // bm_image_free_contiguous_mem 释放；构造时置为 bm_mem_null()，无需单独释放。
  bm_image_free_contiguous_mem(max_batch, m_cropped_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for (int i = 0; i < max_batch; i++) {
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_cropped_imgs[i]);
  }
  if (bmrt != NULL) {
    bmrt_destroy(bmrt);
    bmrt = NULL;
  }
  bm_dev_free(handle);
}

int DINOV2::get_nearest_batch(int real_batch) {
  for (int b : m_batches) {
    if (b >= real_batch) return b;
  }
  return max_batch;
}

int DINOV2::pre_process(vector<bm_image> &images) {
  int image_n = images.size();

  // 1. resize（短边=256, BICUBIC->LINEAR）+ center crop（224x224）
  for (int i = 0; i < image_n; ++i) {
    int src_w = images[i].width;
    int src_h = images[i].height;
    // 与官方 make_classification_eval_transform 一致：短边缩到 256，长边 int() 截断
    int nw, nh;
    if (src_w < src_h) {
      nw = m_resize_size;
      nh = (int)((float)m_resize_size * src_h / src_w);
    } else {
      nh = m_resize_size;
      nw = (int)((float)m_resize_size * src_w / src_h);
    }

    // 1.1 resize（VPP 不支持 BICUBIC，官方用 BICUBIC；此处短边 256 -> crop 224 只裁不放大，
    //     双线性与 BICUBIC 差异可忽略，采用硬件支持的 BMCV_INTER_LINEAR）
    // vpp_convert 同时完成 resize 与 BGR_PACKED->RGB_PLANAR 颜色转换
    // （bm1688/cv186x 的 bmcv_image_convert_to 不支持 packed->planar，故在此处一并转换）
    bm_image resized;
    if (bm_image_create(handle, nh, nw, FORMAT_RGB_PLANAR,
                        images[i].data_type, &resized) != BM_SUCCESS) {
      throw std::runtime_error("bm_image_create resized failed");
    }
    bm_image_alloc_dev_mem(resized);

    if (bmcv_image_vpp_convert(handle, 1, images[i], &resized,
                               NULL, BMCV_INTER_LINEAR) != BM_SUCCESS) {
      bm_image_destroy(resized);
      throw std::runtime_error("bmcv_image_vpp_convert failed");
    }

    // 1.2 center crop
    int left = (int)std::round((nw - m_net_w) / 2.0);
    int top = (int)std::round((nh - m_net_h) / 2.0);
    bmcv_rect_t crop_rect;
    crop_rect.start_x = left;
    crop_rect.start_y = top;
    crop_rect.crop_w = m_net_w;
    crop_rect.crop_h = m_net_h;
    if (bmcv_image_crop(handle, 1, &crop_rect, resized, &m_cropped_imgs[i]) != BM_SUCCESS) {
      bm_image_destroy(resized);
      throw std::runtime_error("bmcv_image_crop failed");
    }
    bm_image_destroy(resized);
  }

  // 2. normalize（RGB_PLANAR uint8 -> RGB_PLANAR float，含 (x/255 - mean) / std）
  if (bmcv_image_convert_to(handle, image_n, converto_attr,
                            m_cropped_imgs.data(), m_converto_imgs.data()) != BM_SUCCESS) {
    throw std::runtime_error("bmcv_image_convert_to (normalize) failed");
  }

  // 3. attach to tensor
  int real_batch = image_n;
  if (real_batch != max_batch) real_batch = get_nearest_batch(real_batch);
  bm_device_mem_t input_dev_mem;
  if (bm_image_get_contiguous_device_mem(real_batch, m_converto_imgs.data(), &input_dev_mem) != BM_SUCCESS) {
    throw std::runtime_error("bm_image_get_contiguous_device_mem failed");
  }
  input_tensor.device_mem = input_dev_mem;
  // 3b 是静态 batch 模型，末尾不足一整个 batch 时必须让 shape 回到最近的合法 batch
  // （= max_batch），否则 bm_launch_tensor 报 "Shapes of the input tensors are not supported"
  input_tensor.shape.dims[0] = real_batch;
  return 0;
}

int DINOV2::forward(bm_tensor_t &input_tensor, std::vector<bm_tensor_t> &output_tensors) {
  bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                               output_tensors.data(), netinfo->output_num);
  if (!ok) {
    std::cerr << "bmrt_launch_tensor() failed" << std::endl;
    return -1;
  }
  auto status = bm_thread_sync(handle);
  assert(BM_SUCCESS == status);
  return 0;
}

float *DINOV2::get_cpu_data(bm_tensor_t *tensor, float scale) {
  int ret = 0;
  float *pFP32 = NULL;
  int count = bmrt_shape_count(&tensor->shape);
  if (misc_info.pcie_soc_mode == 1) {  // soc：device mem 可 mmap 到 host，比 d2s 快
    if (tensor->dtype == BM_FLOAT32) {
      unsigned long long addr;
      ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
      assert(BM_SUCCESS == ret);
      pFP32 = (float *)addr;
    } else if (BM_INT8 == tensor->dtype) {
      int8_t *pI8 = nullptr;
      unsigned long long addr;
      ret = bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
      assert(BM_SUCCESS == ret);
      pI8 = (int8_t *)addr;
      pFP32 = new float[count];
      for (int i = 0; i < count; ++i) {
        pFP32[i] = pI8[i] * scale;
      }
      ret = bm_mem_unmap_device_mem(handle, pI8, bm_mem_get_device_size(tensor->device_mem));
      assert(BM_SUCCESS == ret);
    } else {
      std::cerr << "NOT support dtype=" << tensor->dtype << std::endl;
    }
  } else {  // pcie：d2s
    if (tensor->dtype == BM_FLOAT32) {
      pFP32 = new float[count];
      ret = bm_memcpy_d2s_partial(handle, pFP32, tensor->device_mem, count * sizeof(float));
      assert(BM_SUCCESS == ret);
    } else if (BM_INT8 == tensor->dtype) {
      int8_t *pI8 = new int8_t[bmrt_tensor_bytesize(tensor)];
      pFP32 = new float[count];
      ret = bm_memcpy_d2s_partial(handle, pI8, tensor->device_mem, bmrt_tensor_bytesize(tensor));
      assert(BM_SUCCESS == ret);
      for (int i = 0; i < count; ++i) {
        pFP32[i] = pI8[i] * scale;
      }
      delete [] pI8;
    } else {
      std::cerr << "NOT support dtype=" << tensor->dtype << std::endl;
    }
  }
  return pFP32;
}

int DINOV2::post_process(vector<bm_image> &images, std::vector<bm_tensor_t> &output_tensors,
                         vector<pair<int, float>> &results) {
  results.clear();
  bm_tensor_t &out = output_tensors[0];
  float *feat = get_cpu_data(&out, netinfo->output_scales[0]);  // [B, 1536]

  for (int b = 0; b < (int)images.size(); ++b) {
    const float *f = feat + b * feature_dim;
    // 线性头 -> logits（单遍：同时记录 logits 与 argmax，避免 softmax 分母再算一遍 matvec）
    std::vector<float> logits(class_num);
    float max_logit = -FLT_MAX;
    int max_idx = -1;
    for (int j = 0; j < class_num; j++) {
      const float *w = m_head_weight.data() + (size_t)j * feature_dim;
      // 8 路手动展开：拆成 8 个独立累加器打破串行 FP 依赖链（不违反 IEEE 保序语义即可获得标量 ILP）
      float s0 = m_head_bias[j], s1 = 0.f, s2 = 0.f, s3 = 0.f;
      float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
      int k = 0;
      for (; k + 7 < feature_dim; k += 8) {
        s0 += w[k]     * f[k];
        s1 += w[k + 1] * f[k + 1];
        s2 += w[k + 2] * f[k + 2];
        s3 += w[k + 3] * f[k + 3];
        s4 += w[k + 4] * f[k + 4];
        s5 += w[k + 5] * f[k + 5];
        s6 += w[k + 6] * f[k + 6];
        s7 += w[k + 7] * f[k + 7];
      }
      for (; k < feature_dim; k++)
        s0 += w[k] * f[k];
      float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
      logits[j] = s;
      if (s > max_logit) {
        max_logit = s;
        max_idx = j;
      }
    }
    // softmax 归一化，直接复用上面存好的 logits
    float exp_sum = 0.f;
    for (int j = 0; j < class_num; j++)
      exp_sum += std::exp(logits[j] - max_logit);
    results.push_back({max_idx, 1.0f / exp_sum});
  }

  // 释放 output 设备内存（bmrt_launch_tensor 每次分配）
  for (int i = 0; i < (int)output_tensors.size(); i++) {
    if (misc_info.pcie_soc_mode == 1 && output_tensors[i].dtype == BM_FLOAT32) {
      bm_mem_unmap_device_mem(handle, feat, bm_mem_get_device_size(output_tensors[i].device_mem));
    } else {
      delete [] feat;
    }
    bm_free_device(handle, output_tensors[i].device_mem);
  }
  return 0;
}

int DINOV2::Classify(std::vector<bm_image>& input_images, std::vector<std::pair<int, float>>& results) {
  int ret = 0;
  std::vector<bm_tensor_t> output_tensors;
  output_tensors.resize(netinfo->output_num);

  ts_->save("dinov2 preprocess", max_batch);
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  ts_->save("dinov2 preprocess", max_batch);

  ts_->save("dinov2 inference", max_batch);
  ret = forward(input_tensor, output_tensors);
  CV_Assert(ret == 0);
  ts_->save("dinov2 inference", max_batch);

  ts_->save("dinov2 postprocess", max_batch);
  ret = post_process(input_images, output_tensors, results);
  CV_Assert(ret == 0);
  ts_->save("dinov2 postprocess", max_batch);

  return ret;
}