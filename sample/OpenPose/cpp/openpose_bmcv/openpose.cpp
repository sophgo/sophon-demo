//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include "openpose.hpp"
#include "pose_postprocess.hpp"
// #define DEBUG
using namespace std;

OpenPose::OpenPose(shared_ptr<BMNNContext> context):m_bmContext(context) {
  cout << "OpenPose ctor .." << endl;
}

OpenPose::~OpenPose() {
  cout << "OpenPose dtor ..." << endl;
  bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
  for(int i=0; i<max_batch; i++){
    bm_image_destroy(m_converto_imgs[i]);
    bm_image_destroy(m_resized_imgs[i]);
  }
}

int OpenPose::Init(bool use_tpu_kernel_post){
  //1. get network
  m_bmNetwork = m_bmContext->network(0);

  //2. get input
  max_batch = m_bmNetwork->maxBatch();
  auto tensor = m_bmNetwork->inputTensor(0);
  m_net_h = tensor->get_shape()->dims[2];
  m_net_w = tensor->get_shape()->dims[3];
  //3. get output
  output_num = m_bmNetwork->outputTensorNum();
  assert(output_num > 0);
  auto output_tensor = m_bmNetwork->outputTensor(0);
  auto output_shape = output_tensor->get_shape();
  auto output_dims = output_shape->num_dims;
  float output_scale = output_tensor->get_scale();
  if (output_shape->dims[1] == 57){
    m_model_type = PoseKeyPoints::EModelType::COCO_18;
  }else if (output_shape->dims[1] == 78){
    m_model_type = PoseKeyPoints::EModelType::BODY_25;
  }else{
    std::cout << "Is not a valid m_model_type! "<< std::endl;
    exit(1);
  }

#ifdef DEBUG
  cout << "net_batch = " << max_batch << endl;
  cout << "output_num = " << output_num << endl;
  cout << "output_shape = " << output_shape->dims[0] << "," << output_shape->dims[1] << "," << output_shape->dims[2] << endl;
#endif

  //4. initialize bmimages
  m_resized_imgs.resize(max_batch);
  m_converto_imgs.resize(max_batch);
  // some API only accept bm_image whose stride is aligned to 64
  int aligned_net_w = FFALIGN(m_net_w, 64);
  int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
  for(int i=0; i<max_batch; i++){
    auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
    assert(BM_SUCCESS == ret);
  }
  bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
  bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (tensor->get_dtype() == BM_INT8){
    img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  }
  auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_BGR_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
  assert(BM_SUCCESS == ret);
  // 5.converto
  float input_scale = tensor->get_scale();
  input_scale = input_scale * 1.0 / 255.f ;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = -128 * input_scale;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = -128 * input_scale;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = -128 * input_scale;

  nms_threshold = 0.05;

  if (use_tpu_kernel_post) {
      tpu_kernel_module_t tpu_module;
      std::string tpu_kernel_module_path =
          "../../tpu_kernel_module/"
          "libbm1684x_kernel_module.so";
      tpu_module = tpu_kernel_load_module_file(m_bmContext->handle(),
                                               tpu_kernel_module_path.c_str());
      func_id = tpu_kernel_get_function(
          m_bmContext->handle(), tpu_module,
          "tpu_kernel_api_openpose_part_nms_postprocess");
  }
  return 0;
}

void OpenPose::enableProfile(TimeStamp *ts){
  ts_ = ts;
}

int OpenPose::batch_size(){
  return max_batch;
};

PoseKeyPoints::EModelType OpenPose::get_model_type(){
  return m_model_type;
};

int OpenPose::Detect(const vector<bm_image>& input_images, vector<PoseKeyPoints>& vct_keypoints, std::string& performance_opt){
  int ret = 0;
  //3. preprocess
  ts_->save("openpose preprocess", max_batch);
  ret = pre_process(input_images);
  CV_Assert(ret == 0);
  ts_->save("openpose preprocess", max_batch);
  //4. forward
  ts_->save("openpose inference", max_batch);
  ret = m_bmNetwork->forward();
  CV_Assert(ret == 0);
  ts_->save("openpose inference", max_batch);
  //5. post process
  ts_->save("openpose postprocess", max_batch);
  ret = post_process(input_images, vct_keypoints, performance_opt);
  CV_Assert(ret == 0);
  ts_->save("openpose postprocess", max_batch);
  return ret;
}

int OpenPose::pre_process(const std::vector<bm_image>& images){
  shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();
  //1. resize image
  int ret = 0;
  for(int i = 0; i < image_n; ++i) {
    bm_image image1 = images[i];
    bm_image image_aligned;
    bool need_copy = image1.width & (64-1);

    if(need_copy){
      int stride1[3], stride2[3];
      bm_image_get_stride(image1, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_bmContext->handle(), image1.height, image1.width,
          image1.image_format, image1.data_type, &image_aligned, stride2);

      bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;

      bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
    } else {
      image_aligned = image1;
    }
    auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i]);
    assert(BM_SUCCESS == ret);
    // bm_image_destroy(image1);
    if(need_copy) bm_image_destroy(image_aligned);
  }
  //2. converto
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);

  //3. attach to tensor
  if(image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n); 
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(), &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
  return 0;
}

int OpenPose::post_process(const vector<bm_image> &images, vector<PoseKeyPoints>& vct_keypoints, std::string& performance_opt){
  auto out_tensor = m_bmNetwork->outputTensor(0);
  if (performance_opt == "tpu_kernel_opt" || performance_opt == "tpu_kernel_half_img_size_opt")
    OpenPosePostProcess::getKeyPointsTPUKERNEL(out_tensor, images, vct_keypoints, m_model_type, nms_threshold, performance_opt == "tpu_kernel_half_img_size_opt", m_bmContext->handle(), func_id);
  else if (performance_opt == "cpu_opt")
    OpenPosePostProcess::getKeyPointsCPUOpt(out_tensor, images, vct_keypoints, m_model_type, nms_threshold);
  else
    OpenPosePostProcess::getKeyPoints(out_tensor, images, vct_keypoints, m_model_type, nms_threshold);
  return 0;
}

