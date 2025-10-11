
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov8.hpp"
#include <cstdio>
#include <fstream>


#define USE_MULTICLASS_NMS 1  // 后处理nms方式
#define FPS 1                 // 是否计算fps
#define PRESSURE 1            // 压测，循环解码
#define INTERVAL 10         // 压测打印fps的时间间隔（秒）

extern std::chrono::time_point<std::chrono::high_resolution_clock> before_cap_init;

YOLOv8::YOLOv8(int dev_id, 
              std::string bmodel_path, 
              std::vector<std::string> input_paths, 
              std::vector<bool> is_videos,
              std::vector<int> skip_frame_nums,
              int queue_size,
              int num_pre,
              int num_post,
              float confThresh,
              float nmsThresh
              ):  
                m_dev_id(dev_id),
                m_queue_size(queue_size), 
                m_num_decode(input_paths.size()),
                m_num_pre(num_pre),
                m_num_post(num_post),
                m_stop_decode(0),
                m_stop_pre(0),
                m_stop_post(0),
                m_is_stop_decode(false),
                m_is_stop_pre(false),
                m_is_stop_infer(false),
                m_is_stop_post(false),
                m_confThreshold(confThresh),
                m_nmsThreshold(nmsThresh),
                m_queue_decode("decode", m_queue_size, 2),
                m_queue_pre("pre", m_queue_size, 2),
                m_queue_infer("infer", m_queue_size, 2),
                m_queue_post("post", m_queue_size, 2)
{


  // get handle
  auto ret = bm_dev_request(&m_handle, dev_id);
  assert(BM_SUCCESS == ret);

  // judge now is pcie or soc
  ret = bm_get_misc_info(m_handle, &misc_info);
  assert(BM_SUCCESS == ret);
  
  // create bmrt
  // void *bmrt = NULL;
  bmrt = bmrt_create(m_handle);
  if (!bmrt_load_bmodel(bmrt, bmodel_path.c_str())) {
      std::cout << "load bmodel(" << bmodel_path << ") failed" << std::endl;
  }

  // get network names from bmodel
  const char **names;
  int num = bmrt_get_network_number(bmrt);
  if (num > 1){
      std::cout << "This bmodel have " << num << " networks, and this program will only take network 0." << std::endl;
  }
  bmrt_get_network_names(bmrt, &names);
  for(int i = 0; i < num; ++i) {
      network_names.push_back(names[i]);
  }
  free(names);

  // get netinfo by netname
  netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
  if (netinfo->stage_num > 1){
      std::cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << std::endl;
  }
  m_batch_size = netinfo->stages[0].input_shapes[0].dims[0];
  m_net_h = netinfo->stages[0].input_shapes[0].dims[2];
  m_net_w = netinfo->stages[0].input_shapes[0].dims[3];
  
  for (int i = 0; i < netinfo->output_num; i++) {
      auto& shape = netinfo->stages[0].output_shapes[i];
      if (shape.num_dims == 3) {
          m_class_num = shape.dims[2] - 4;
          if (shape.dims[1] < shape.dims[2]) {
              std::cout << "Your model's output is not efficient for cpp, please refer to the docs/YOLOv8_Export_Guide.md to export model which has transposed output." << std::endl;
              m_class_num = shape.dims[1] - 4;
              is_output_transposed = false;
          }
      }
  }
  if (m_class_num == -1) {
      throw std::runtime_error("Invalid model output shape.");
  }

  
  // input attr
  img_dtype = DATA_TYPE_EXT_FLOAT32;
  if (netinfo->input_dtypes[0] == BM_INT8){
      img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
  } else if (netinfo->input_dtypes[0] == BM_UINT8){
      img_dtype = DATA_TYPE_EXT_1N_BYTE;
  }

  float input_scale = netinfo->input_scales[0] / 255.f;
  converto_attr.alpha_0 = input_scale;
  converto_attr.beta_0 = 0;
  converto_attr.alpha_1 = input_scale;
  converto_attr.beta_1 = 0;
  converto_attr.alpha_2 = input_scale;
  converto_attr.beta_2 = 0;


  // init decode 
  for (int i = 0; i < m_num_decode; i++){
    auto decode_element = std::make_shared<DecEle>();
    if (is_videos[i]){
      decode_element->is_video = true;
      decode_element->before_cap_init = std::chrono::high_resolution_clock::now();
      before_cap_init = decode_element->before_cap_init;//global variable
      decode_element->cap = cv::VideoCapture(input_paths[i], cv::CAP_ANY, dev_id);
      if(input_paths[i].find("/dev/video6") == 0 || input_paths[i].find("/dev/video7") == 0){
        printf("Usb camera input, using format MJPG.\n");
        decode_element->cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
      }
      decode_element->after_cap_init = std::chrono::high_resolution_clock::now();
      
      auto cap_init_delay = decode_element->after_cap_init - decode_element->before_cap_init;
      printf("Channel: %d, cap_init_delay: %.2f ms;\n", i, cap_init_delay.count() * 1e-6);
      
      if (!decode_element->cap.isOpened()){
        std::cerr << "Error: open video src failed in channel " << i << std::endl;
        exit(1);
      }
      decode_element->dec_frame_idx = 1;
      decode_element->skip_frame_num = skip_frame_nums[i]+1;
      decode_element->time_interval = 1/decode_element->cap.get(cv::CAP_PROP_FPS)*1e+3;
      
    }else{
      std::vector<std::string> image_paths;
      for (const auto& entry: std::filesystem::directory_iterator(input_paths[i])){
        if (entry.is_regular_file()){
          image_paths.emplace_back(entry.path().filename().string());
        }
      }
        
      decode_element->is_video = false;
      decode_element->dir_path = input_paths[i];
      decode_element->image_name_list = image_paths;
      decode_element->image_name_it = decode_element->image_name_list.begin();
    }
    m_decode_elements.emplace_back(decode_element);
    m_decode_frame_ids.emplace_back(0);
  }

  m_input_paths = input_paths;

  // init pre
  for (int i = 0; i < m_num_pre; i++){
    std::vector<bm_image> resized_bmimgs(m_batch_size);
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int i = 0; i < m_batch_size; i++){
      auto ret = bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &resized_bmimgs[i], strides);
      assert(BM_SUCCESS == ret);
    }
    auto ret = bm_image_alloc_contiguous_mem(m_batch_size, resized_bmimgs.data());
    assert(BM_SUCCESS == ret);
    m_vec_resized_bmimgs.emplace_back(resized_bmimgs);
  }

#if FPS
  m_start = std::chrono::high_resolution_clock::now();
#endif

  // init decode worker
  for (int i = 0; i < m_num_decode; i++){
    m_thread_decodes.emplace_back(&YOLOv8::worker_decode, this, i);
    time_counters.emplace_back(std::chrono::high_resolution_clock::now());
    decode_frame_counts.emplace_back(0);
  }

  // init pre worker
  for (int i = 0; i < m_num_pre; i++){
    m_thread_pres.emplace_back(&YOLOv8::worker_pre, this, i);
  }

  // init infer worker
  m_thread_infer = std::thread(&YOLOv8::worker_infer, this);

  // init post worker
  for (int i = 0; i < m_num_post; i++){
    m_thread_posts.emplace_back(&YOLOv8::worker_post, this);
  }

#if PRESSURE
  counter_pressure = std::thread(&YOLOv8::worker_pressure, this);
#endif
}


YOLOv8::~YOLOv8()
{
  if (bmrt!=NULL) {
    bmrt_destroy(bmrt);
    bmrt = NULL;
    }  
    bm_dev_free(m_handle);

  for (auto& thread: m_thread_decodes){
    if (thread.joinable())
      thread.join();
  }

  for (auto& thread: m_thread_pres){
    if (thread.joinable())
      thread.join();
  }

  if (m_thread_infer.joinable())
    m_thread_infer.join();

  for (auto& thread: m_thread_posts){
    if (thread.joinable())
      thread.join();
  }

#if PRESSURE
  if (counter_pressure.joinable()){
    counter_pressure.join();
  }
#endif


#if FPS
  m_end = std::chrono::high_resolution_clock::now();
  auto duration = m_end - m_start;
  int frame_total = 0;
  for (int i = 0; i < m_num_decode; i++){
    frame_total += get_frame_count(i);
  }
  std::cout << "yolov8 fps: " << frame_total / (duration.count() * 1e-9) << std::endl;
#endif

  // free decode
  for (auto& ele: m_decode_elements){
    if (ele->is_video)
      ele->cap.release();
  }
  
  // free pre
  for (int i = 0; i < m_num_pre; i++){
    auto ret = bm_image_free_contiguous_mem(m_batch_size, m_vec_resized_bmimgs[i].data());
    assert(ret == BM_SUCCESS);
    for (int j = 0; j < m_batch_size; j++){
      auto ret = bm_image_destroy(m_vec_resized_bmimgs[i][j]);
      assert(ret == BM_SUCCESS);
    }
    
  }
}

int YOLOv8::get_frame_count(int channel_id){
  return m_decode_frame_ids[channel_id];
}



// -------------------------线程函数----------------------------------
void YOLOv8::worker_decode(int channel_id){
  while (true){
    auto data = std::make_shared<DataDec>();
    decode(data, channel_id);
    if(m_decode_frame_ids[channel_id] == 0){
      auto time_now = std::chrono::high_resolution_clock::now();
      auto first_frame_delay = time_now - m_decode_elements[channel_id]->before_cap_init;
      printf("Channel: %d, worker_decode: first_frame_delay: %.2f ms;\n", channel_id, first_frame_delay.count() * 1e-6);
    }
    decode_frame_counts[channel_id] += 1;
    // frame_id为-1时代表读到eof，不进行后续处理
    // 只有可以放入的图片才设置frame id，保证frame id是连续的
    if (data->frame_id != -1){
      if (m_decode_elements[channel_id]->is_video){
        // 输入为视频
        // 流控
        auto time_count = std::chrono::high_resolution_clock::now();
        int sleep_time = int (m_decode_elements[channel_id]->time_interval - (time_count-time_counters[channel_id]).count()*1e-6);

        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        time_counters[channel_id] = time_count;

        // 跳帧
        if (decode_frame_counts[channel_id] % m_decode_elements[channel_id]->skip_frame_num == 0)
        {
          data->frame_id = m_decode_frame_ids[channel_id];
          m_decode_frame_ids[channel_id] += 1;
          m_queue_decode.push_back(data);
          {
            std::unique_lock<std::mutex> lock(m_mutex_map_origin);
            m_origin_image[data->channel_id][data->frame_id] = data->image;
          }
        }

      }else{
        // 输入为图片
        data->frame_id = m_decode_frame_ids[channel_id];
        m_decode_frame_ids[channel_id] += 1;
        m_queue_decode.push_back(data);
        {
          std::unique_lock<std::mutex> lock(m_mutex_map_origin);
          m_origin_image[data->channel_id][data->frame_id] = data->image;
        }

        // 保存图片名称，在输入为图片时使用
        {
          std::unique_lock<std::mutex> lock(m_mutex_map_name);
          m_image_name[data->channel_id][data->frame_id] = data->image_name;
        }

      }
    }


#if PRESSURE
    if (data->frame_id == -1){
      if (m_decode_elements[channel_id]->is_video){
        std::cout << "channel " << channel_id << " meets eof" << std::endl;
        auto &cap = m_decode_elements[channel_id]->cap;
        cap.release();
        cap.open(m_input_paths[channel_id]);
        if (!cap.isOpened()) {
          std::cerr << "Failed to reopen the video file." << std::endl;
          exit(1);
        }
      }else {
        m_decode_elements[channel_id]->image_name_it = m_decode_elements[channel_id]->image_name_list.begin();
        std::cout << "channel " << channel_id << ": All pic has been read and restart from the beginning. " << std::endl;
      }
    }
#else
    // 如果是eof，解码停止
    if (data->frame_id == -1){
      std::unique_lock<std::mutex> lock(m_mutex_stop_decode);
      m_stop_decode ++;
      // 如果所有路解码停止，向后发送信号
      if (m_stop_decode == m_num_decode){
        m_is_stop_decode = true;
        m_queue_decode.set_stop_flag(true);
      }

      return;
    }
#endif
  }
}

void YOLOv8::worker_pre(int pre_idx) {
  while (true) {
    std::vector<std::shared_ptr<DataDec>> dec_images;
    auto pre_data = std::make_shared<DataInfer>();
    std::vector<std::pair<int, int>> txy_batch;
    std::vector<std::pair<float, float>> ratios_batch;
    int ret = 0;
    bool no_data = false;

    // 取一个batch的数据做预处理
    for (int i = 0; i < m_batch_size; i++) {
      std::shared_ptr<DataDec> data;
      ret = m_queue_decode.pop_front(data);
      if (ret == 0) {
        dec_images.emplace_back(data);
      } else {
        if (i == 0) {
          no_data = true;
        }
        break;
      }
    }

    // 解码线程停止并且解码队列为空，可以结束工作线程
    if (no_data) {
      std::unique_lock<std::mutex> lock(m_mutex_stop_pre);
      if (m_is_stop_decode && ret == -1) {
        m_stop_pre++;
        if (m_stop_pre == m_num_pre) {
          m_is_stop_pre = true;
          m_queue_pre.set_stop_flag(true);
        }
        return;
      }
    }

    preprocess(dec_images, pre_data, pre_idx);
    for(int i = 0; i < m_batch_size; i++){
      if(pre_data->frame_ids[i] == 0){
        auto time_now = std::chrono::high_resolution_clock::now();
        auto first_frame_delay = time_now - m_decode_elements[i]->before_cap_init;
        printf("Channel: %d, worker_pre: first_preprocessed_frame_delay: %.2f ms;\n", pre_data->channel_ids[i], first_frame_delay.count() * 1e-6);
      }
    }

    m_queue_pre.push_back(pre_data);

  }
}

void YOLOv8::worker_infer(){
  while (true){
    auto input_data = std::make_shared<DataInfer>();
    auto output_data = std::make_shared<DataInfer>();

    auto ret = m_queue_pre.pop_front(input_data);

    // 预处理线程停止并且预处理队列为空，可以结束工作线程
    if (m_is_stop_pre && ret == -1){
      m_is_stop_infer = true;
      m_queue_infer.set_stop_flag(true);
      return;
    }

    inference(input_data, output_data);
    for(int i = 0; i < m_batch_size; i++){
      if(output_data->frame_ids[i] == 0){
        auto time_now = std::chrono::high_resolution_clock::now();
        auto first_frame_delay = time_now - m_decode_elements[i]->before_cap_init;
        printf("Channel: %d, worker_infer: first_inferenced_frame_delay: %.2f ms;\n", output_data->channel_ids[i], first_frame_delay.count() * 1e-6);
      }
    }
    m_queue_infer.push_back(output_data);

  }
}

void YOLOv8::worker_post(){
  while (true){
    auto output_data = std::make_shared<DataInfer>();
    std::vector<std::shared_ptr<DataPost>> box_datas;
    
    auto ret = m_queue_infer.pop_front(output_data);
    {
      std::unique_lock<std::mutex> lock(m_mutex_stop_post);
      if (m_is_stop_infer && ret == -1){
        m_stop_post ++;
        if (m_stop_post == m_num_post){
          m_is_stop_post = true;
          m_queue_post.set_stop_flag(true);
        }
        return;
      }
    }
    postprocess(output_data, box_datas);
    for(int i = 0; i < m_batch_size; i++){
      if(output_data->frame_ids[i] == 0){
        auto time_now = std::chrono::high_resolution_clock::now();
        auto first_frame_delay = time_now - m_decode_elements[i]->before_cap_init;
        printf("Channel: %d, worker_post: first_postprocessed_frame_delay: %.2f ms;\n", output_data->channel_ids[i], first_frame_delay.count() * 1e-6);
      }
    }
    for (int i = 0; i < box_datas.size(); i++){
      m_queue_post.push_back(box_datas[i]);
    }

  }
}



// ------------------------------处理函数---------------------------

// 调对应的vectore中的decoder
void YOLOv8::decode(std::shared_ptr<DataDec> data, int channel_id){
  auto decode_ele = m_decode_elements[channel_id];
  cv::Mat image;

  if (decode_ele->is_video){
    decode_ele->cap.read(image);
    
    // eof返回frame_id -1;
    if (image.empty()){
      data->frame_id = -1;
    }else{
      data->image = image;
      data->channel_id = channel_id;
      data->frame_id = 0;
    }
  }else {
    if (decode_ele->image_name_it == decode_ele->image_name_list.end()){
      data->frame_id = -1;
    }
    else{
      std::string name = *decode_ele->image_name_it;
      std::string image_path = decode_ele->dir_path + name;
      image = cv::imread(image_path, cv::IMREAD_COLOR, m_dev_id);
      data->image = image;
      data->channel_id = channel_id;
      data->frame_id = 0;
      data->image_name = name;
      decode_ele->image_name_it ++;
    }
  }
}

// for循环处理多batch，dec_images.size()代表有效数据的数量
void YOLOv8::preprocess(std::vector<std::shared_ptr<DataDec>> &dec_images, 
                        std::shared_ptr<DataInfer> pre_data, int idx){
  auto resized_bmimgs = m_vec_resized_bmimgs[idx];

  // resize需要单图做，但convertto不需要
  for (int i = 0; i < dec_images.size(); i++){
    auto dec_image = dec_images[i];

    bm_image bmimg;
    cv::bmcv::toBMI(dec_image->image, &bmimg);
    bm_image bmimg_aligned;
    bool need_copy = bmimg.width & (64-1);
    if (need_copy){
      int stride1[3], stride2[3];
      bm_image_get_stride(bmimg, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_handle, bmimg.height, bmimg.width,
                      bmimg.image_format, bmimg.data_type, &bmimg_aligned, stride2);
      
      bm_image_alloc_dev_mem(bmimg_aligned, 1);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;
      bmcv_image_copy_to(m_handle, copyToAttr, bmimg, bmimg_aligned);
    } else {
      bmimg_aligned = bmimg;
    }

    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(bmimg.width, bmimg.height, m_net_w, m_net_h, &isAlignWidth);
    int tx1 = 0, ty1 = 0;
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
        padding_attr.dst_crop_h = bmimg.height * ratio;
        padding_attr.dst_crop_w = m_net_w;

        ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
        padding_attr.dst_crop_sty = ty1;
        padding_attr.dst_crop_stx = 0;
    } else {
        padding_attr.dst_crop_h = m_net_h;
        padding_attr.dst_crop_w = bmimg.width * ratio;

        tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, bmimg.width, bmimg.height};
    auto ret = bmcv_image_vpp_convert_padding(m_handle, 1, bmimg_aligned, 
        &resized_bmimgs[i], &padding_attr, &crop_rect);
    assert(BM_SUCCESS == ret);

    if (need_copy){
      bm_image_destroy(bmimg_aligned);
    }
    bm_image_destroy(bmimg);

    pre_data->channel_ids.emplace_back(dec_images[i]->channel_id);
    pre_data->frame_ids.emplace_back(dec_images[i]->frame_id);
    pre_data->txy_batch.emplace_back(tx1, ty1);
    pre_data->ratios_batch.emplace_back(ratio, ratio);
  }
  // 这里先bmlib申请了连续batch_size的内存，做归一化的bmimage内存是attach的，
  // 因为后面tensor需要的是相同的dev mem，这里申请的在推理完成后会进行释放（推理函数中）
  std::vector<bm_image> converto_bmimgs(m_batch_size);
  for (int i = 0; i < m_batch_size; i++){
    bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, &converto_bmimgs[i]);
  }

  bm_tensor_t tensor;
  int ret = 0;
  ret = bmrt_tensor(&tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
  assert(true == ret);
  bm_image_attach_contiguous_mem(m_batch_size, converto_bmimgs.data(), tensor.device_mem);

  ret = bmcv_image_convert_to(m_handle, m_batch_size, converto_attr, resized_bmimgs.data(),
                                converto_bmimgs.data());
  assert(ret == 0);

  bm_image_dettach_contiguous_mem(m_batch_size, converto_bmimgs.data());

  for (int i = 0; i < m_batch_size; i ++){
    bm_image_destroy(converto_bmimgs[i]);
  }

  pre_data->tensors.emplace_back(tensor);
  
}

void YOLOv8::inference(std::shared_ptr<DataInfer> input_data, 
                      std::shared_ptr<DataInfer> output_data) {
    output_data->channel_ids = input_data->channel_ids;
    output_data->frame_ids = input_data->frame_ids;
    output_data->txy_batch = input_data->txy_batch;
    output_data->ratios_batch = input_data->ratios_batch;

    output_data->tensors.resize(netinfo->output_num);
    bool ok = bmrt_launch_tensor(bmrt, 
                               netinfo->name, 
                               &input_data->tensors[0], 
                               netinfo->input_num,
                               output_data->tensors.data(), 
                               netinfo->output_num);
    assert(ok == true);
    auto ret = bm_thread_sync(m_handle);
    assert(BM_SUCCESS == ret);
    bm_free_device(m_handle, input_data->tensors[0].device_mem);
}

int YOLOv8::postprocess(std::shared_ptr<DataInfer> output_infer, 
                            std::vector<std::shared_ptr<DataPost>>& box_data) {
    float* data_box = nullptr;
    bm_tensor_t tensor_box;
    auto& output_tensors = output_infer->tensors;
    
    for(int i = 0; i < output_tensors.size(); i++) {
        if(output_tensors[i].shape.num_dims == 3){
            tensor_box = output_tensors[i];
            data_box = get_cpu_data(output_tensors[i], netinfo->output_scales[i]);
        }
    }

    for (int batch_idx = 0; batch_idx < output_infer->frame_ids.size(); ++batch_idx) {
        auto post_data = std::make_shared<DataPost>();
        post_data->channel_id = output_infer->channel_ids[batch_idx];
        post_data->frame_id = output_infer->frame_ids[batch_idx];
        YOLOv8BoxVec& yolobox_vec = post_data->boxes;

        int box_num = is_output_transposed ? tensor_box.shape.dims[1] : tensor_box.shape.dims[2];
        int nout = is_output_transposed ? tensor_box.shape.dims[2] : tensor_box.shape.dims[1];
        float* batch_data_box = data_box + batch_idx * box_num * nout; //output_tensor: [bs, box_num, class_num + 5]
        int offset = is_output_transposed ? 1 : box_num;

        // Candidates
        for (int i = 0; i < box_num; i++) {
            int box_index = is_output_transposed ? i * nout : i;
            float* cls_conf = batch_data_box + box_index + 4 * offset; 

#if USE_MULTICLASS_NMS
            // multilabel
            for (int j = 0; j < m_class_num; j++) {
                float cur_value = cls_conf[j * offset];
                if (cur_value > m_confThreshold) {
                    YOLOv8Box box;
                    box.score = cur_value;
                    box.class_id = j;
                    float centerX = batch_data_box[box_index];
                    float centerY = batch_data_box[box_index + 1 * offset];
                    float width = batch_data_box[box_index + 2 * offset];
                    float height = batch_data_box[box_index + 3 * offset];

                    int c = agnostic ? 0 : box.class_id * max_wh;
                    box.x1 = centerX - width / 2 + c;
                    box.y1 = centerY - height / 2 + c;
                    box.x2 = box.x1 + width;
                    box.y2 = box.y1 + height;
                    yolobox_vec.push_back(box);
                }
            }
#else
            // best class
            YOLOv8Box box;
            if(is_output_transposed){
                box.class_id = argmax(batch_data_box + box_index + 4, m_class_num);
                box.score = batch_data_box[box_index + 4 + box.class_id];
            } else {
                float max_value = 0.0;
                int max_index = 0;
                for(int j = 0; j < m_class_num; j++){
                    float cur_value = cls_conf[i + j * box_num];
                    if(cur_value > max_value){
                        max_value = cur_value;
                        max_index = j;
                    }
                }
                box.class_id = max_index;
                box.score = max_value;
            }

            if(box.score <= m_confThreshold){
                continue;
            }
            int c = agnostic ? 0 : box.class_id * max_wh;
            float centerX = batch_data_box[box_index];
            float centerY = batch_data_box[box_index + 1 * offset];
            float width = batch_data_box[box_index + 2 * offset];
            float height = batch_data_box[box_index + 3 * offset];
            box.x1 = centerX - width / 2 + c;
            box.y1 = centerY - height / 2 + c;
            box.x2 = box.x1 + width;
            box.y2 = box.y1 + height;
            yolobox_vec.push_back(box);
#endif
        }

        NMS(yolobox_vec, m_nmsThreshold);

        if (yolobox_vec.size() > max_det) {
            yolobox_vec.erase(yolobox_vec.begin(), yolobox_vec.begin() + (yolobox_vec.size() - max_det));
        }

        if(!agnostic){
            for (int i = 0; i < yolobox_vec.size(); i++) {
                int c = yolobox_vec[i].class_id * max_wh;
                yolobox_vec[i].x1 = yolobox_vec[i].x1 - c;
                yolobox_vec[i].y1 = yolobox_vec[i].y1 - c;
                yolobox_vec[i].x2 = yolobox_vec[i].x2 - c;
                yolobox_vec[i].y2 = yolobox_vec[i].y2 - c;
            }
        }

        int tx1 = output_infer->txy_batch[batch_idx].first;
        int ty1 = output_infer->txy_batch[batch_idx].second;
        float ratio_x = output_infer->ratios_batch[batch_idx].first;
        float ratio_y = output_infer->ratios_batch[batch_idx].second;
        float inv_ratio_x = 1.0 / ratio_x;
        float inv_ratio_y = 1.0 / ratio_y;
        for (int i = 0; i < yolobox_vec.size(); i++) {
            yolobox_vec[i].x1 = std::round((yolobox_vec[i].x1 - tx1) * inv_ratio_x);
            yolobox_vec[i].y1 = std::round((yolobox_vec[i].y1 - ty1) * inv_ratio_y);
            yolobox_vec[i].x2 = std::round((yolobox_vec[i].x2 - tx1) * inv_ratio_x);
            yolobox_vec[i].y2 = std::round((yolobox_vec[i].y2 - ty1) * inv_ratio_y);
        }

        box_data.push_back(post_data);
    }

    for(int i = 0; i < output_tensors.size(); i++) {
        float* tensor_data = nullptr;
        if(output_tensors[i].shape.num_dims == 3){
            tensor_data = data_box;
        }

        if(misc_info.pcie_soc_mode == 1){ // soc
            if(output_tensors[i].dtype != BM_FLOAT32){
                delete [] tensor_data;
            } else {
                int tensor_size = bm_mem_get_device_size(output_tensors[i].device_mem);
                bm_status_t ret = bm_mem_unmap_device_mem(m_handle, tensor_data, tensor_size);
                assert(BM_SUCCESS == ret);
            }
        } else {
            delete [] tensor_data;
        }
        bm_free_device(m_handle, output_tensors[i].device_mem);
    }
    
    return 0;
}

int YOLOv8::get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image, std::string& image_name){

  auto ret = m_queue_post.pop_front(box_data);

  if (ret == -1){
    return 1;
  }
  int channel_id = box_data->channel_id;
  int frame_id = box_data->frame_id;
  {
    std::unique_lock<std::mutex> lock(m_mutex_map_origin);
    origin_image = std::make_shared<cv::Mat>(m_origin_image[channel_id][frame_id]);
    m_origin_image[channel_id].erase(frame_id);
  }

  if (m_decode_elements[channel_id]->is_video){
    image_name = std::to_string(channel_id) + '_' + std::to_string(frame_id) + ".jpg";
  } else{
    std::unique_lock<std::mutex> lock(m_mutex_map_name);
    image_name = m_image_name[channel_id][frame_id];
    m_image_name[channel_id].erase(frame_id);
  }

  return 0;
}

int YOLOv8::get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image){
  std::string name;
  return get_post_data(box_data, origin_image, name);
}

float YOLOv8::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* pIsAligWidth) {
    float ratio;
    float r_w = (float)dst_w / src_w;
    float r_h = (float)dst_h / src_h;
    if (r_h > r_w) {
        *pIsAligWidth = true;
        ratio = r_w;
    } else {
        *pIsAligWidth = false;
        ratio = r_h;
    }
    return ratio;
}

float* YOLOv8::get_cpu_data(bm_tensor_t& tensor, int out_idx) {
    float* cpu_data;
    bm_status_t ret;
    float *pFP32 = nullptr;
    int count = bmrt_shape_count(&tensor.shape);
    
    if(can_mmap){
        if (tensor.dtype == BM_FLOAT32) {
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
            assert(BM_SUCCESS == ret);
            pFP32 = (float*)addr;
        } else if (BM_INT8 == tensor.dtype) {
            int8_t *pI8 = nullptr;
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
            assert(BM_SUCCESS == ret);
            pI8 = (int8_t*)addr;

            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pI8[i] * m_output_scales[out_idx];
            }
            ret = bm_mem_unmap_device_mem(m_handle, pI8, bm_mem_get_device_size(tensor.device_mem));
            assert(BM_SUCCESS == ret);
        } else if (tensor.dtype == BM_INT32) {
            int32_t *pI32 = nullptr;
            unsigned long long addr;
            ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
            assert(BM_SUCCESS == ret);
            ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
            assert(BM_SUCCESS == ret);
            pI32 = (int32_t*)addr;

            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pI32[i] * m_output_scales[out_idx];
            }
            ret = bm_mem_unmap_device_mem(m_handle, pI32, bm_mem_get_device_size(tensor.device_mem));
            assert(BM_SUCCESS == ret);
        } else {
            std::cout << "NOT support dtype=" << tensor.dtype << std::endl;
        }
    } else {
        if (tensor.dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pFP32, tensor.device_mem, count * sizeof(float));
            assert(BM_SUCCESS == ret);
        } else if (BM_INT8 == tensor.dtype) {
            int8_t *pI8 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(&tensor);
            pI8 = new int8_t[tensor_size];
            assert(pI8 != nullptr);

            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pI8, tensor.device_mem, tensor_size);
            assert(BM_SUCCESS == ret);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pI8[i] * m_output_scales[out_idx];
            }
            delete [] pI8;
        } else if(tensor.dtype == BM_INT32) {
            int32_t *pI32 = nullptr;
            int tensor_size = bmrt_tensor_bytesize(&tensor);
            pI32 = new int32_t[tensor_size];
            assert(pI32 != nullptr);

            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pI32, tensor.device_mem, tensor_size);
            assert(BM_SUCCESS == ret);
            for(int i = 0; i < count; ++i) {
                pFP32[i] = pI32[i] * m_output_scales[out_idx];
            }
            delete [] pI32;
        } else {
            std::cout << "NOT support dtype=" << tensor.dtype << std::endl;
        }
    }

    cpu_data = pFP32;
    return cpu_data;
}

int YOLOv8::argmax(float* data, int num){
  float max_value = 0.0;
  int max_index = 0;
  for(int i = 0; i < num; ++i) {
    float value = data[i];
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  return max_index;
}

void YOLOv8::NMS(YOLOv8BoxVec& dets, float nmsConfidence) {
    int length = dets.size();
    int index = length - 1;

    std::sort(dets.begin(), dets.end(), [](const YOLOv8Box& a, const YOLOv8Box& b) { return a.score < b.score; });

    std::vector<float> areas(length);
    for (int i = 0; i < length; i++) {
        float width = dets[i].x2 - dets[i].x1;
        float height = dets[i].y2 - dets[i].y1;
        areas[i] = width * height;
    }

    while (index > 0) {
        int i = 0;
        while (i < index) {
            float left = std::max(dets[index].x1, dets[i].x1);
            float top = std::max(dets[index].y1, dets[i].y1);
            float right = std::min(dets[index].x2, dets[i].x2);
            float bottom = std::min(dets[index].y2, dets[i].y2);
            float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
                areas.erase(areas.begin() + i);
                dets.erase(dets.begin() + i);
                index--;
            } else {
                i++;
            }
        }
        index--;
    }
}

void YOLOv8::worker_pressure(){
  std::vector<int> start_frame_counts(m_num_decode);
  while (true)
  {
    int diff = 0;
    for(int i = 0; i < m_num_decode; i ++){
      start_frame_counts[i] = get_frame_count(i);
    }
    std::this_thread::sleep_for(std::chrono::seconds(INTERVAL));

    for(int i = 0; i < m_num_decode; i ++){
      int frame_count = get_frame_count(i);
      diff += frame_count - start_frame_counts[i];
      start_frame_counts[i] = frame_count;
    }

    std::cout << "yolov8 fps: " << float(diff)/(INTERVAL) << std::endl;
    
  }
  
}


