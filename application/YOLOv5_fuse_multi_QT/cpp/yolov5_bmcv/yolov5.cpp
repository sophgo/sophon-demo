//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov5.hpp"
#include <fstream>


#define USE_MULTICLASS_NMS 1  // 后处理nms方式
#define FPS 1                 // 是否计算fps
#define PRESSURE 1            // 压测，循环解码
#define INTERVAL 10         // 压测打印fps的时间间隔（秒）

YOLOv5::YOLOv5(int dev_id, 
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

  // context
  m_ctx = std::make_unique<bmruntime::Context>(dev_id);
  m_ctx->load_bmodel(bmodel_path.c_str());

  // network
  std::vector<const char *> net_names;
  m_ctx->get_network_names(&net_names);
  m_net = std::make_unique<bmruntime::Network>(*m_ctx, net_names[0]);
  m_handle = m_ctx->handle();
  m_inputs = m_net->Inputs();
  m_outputs = m_net->Outputs();
  m_output_num = m_net->info()->output_num;
  for (int i = 0; i < m_output_num; i++){
    m_output_scales.emplace_back(m_net->info()->output_scales[i]);
    m_output_shapes.emplace_back(m_outputs[i]->tensor()->shape);
    m_output_dtypes.emplace_back(m_outputs[i]->tensor()->dtype);
  }
  m_batch_size = m_net->info()->stages[0].input_shapes[0].dims[0];
  if(m_net->info()->stages[0].output_shapes[0].num_dims == 4){
    is_fuse_postprocess = true;
  }
  if(m_net->info()->stages[0].input_shapes[0].dims[3] == 3){
    is_fuse_preprocess = true;
  }
  if(is_fuse_preprocess){ // for yolov5_fuse
    m_net_h = m_net->info()->stages[0].input_shapes[0].dims[1];
    m_net_w = m_net->info()->stages[0].input_shapes[0].dims[2];
  }else{
    m_net_h = m_net->info()->stages[0].input_shapes[0].dims[2];
    m_net_w = m_net->info()->stages[0].input_shapes[0].dims[3];
  }

  // is soc?
  struct bm_misc_info misc_info;
  bm_status_t ret = bm_get_misc_info(m_handle, &misc_info);
  assert(BM_SUCCESS == ret);
  can_mmap = misc_info.pcie_soc_mode == 1;
  
  // input attr
  img_dtype = (m_inputs[0]->tensor()->dtype == BM_UINT8)? DATA_TYPE_EXT_1N_BYTE: DATA_TYPE_EXT_FLOAT32;
  float input_scale = m_net->info()->input_scales[0];
  input_scale = input_scale * 1.0 / 255.f;
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
      decode_element->cap = cv::VideoCapture(input_paths[i], cv::CAP_FFMPEG, dev_id);
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
    m_thread_decodes.emplace_back(&YOLOv5::worker_decode, this, i);
    time_counters.emplace_back(std::chrono::high_resolution_clock::now());
    decode_frame_counts.emplace_back(0);
  }

  // init pre worker
  for (int i = 0; i < m_num_pre; i++){
    m_thread_pres.emplace_back(&YOLOv5::worker_pre, this, i);
  }

  // init infer worker
  m_thread_infer = std::thread(&YOLOv5::worker_infer, this);

  // init post worker
  for (int i = 0; i < m_num_post; i++){
    m_thread_posts.emplace_back(&YOLOv5::worker_post, this);
  }

#if PRESSURE
  counter_pressure = std::thread(&YOLOv5::worker_pressure, this);
#endif
}


YOLOv5::~YOLOv5()
{
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
  std::cout << "yolov5 fps: " << frame_total / (duration.count() * 1e-9) << std::endl;
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

int YOLOv5::get_frame_count(int channel_id){
  return m_decode_frame_ids[channel_id];
}



// -------------------------线程函数----------------------------------
void YOLOv5::worker_decode(int channel_id){
  while (true){
    auto data = std::make_shared<DataDec>();
    decode(data, channel_id);
        decode_frame_counts[channel_id] += 1;

    // frame_id为-1时代表读到eof，不进行后续处理
    // 只有可以放入的图片才设置frame id，保证frame id是连续的
    if (data->frame_id != -1){
      if (m_decode_elements[channel_id]->is_video){
        // 输入为视频
        // 流控
        auto time_count = std::chrono::high_resolution_clock::now();
        int sleep_time = int (m_decode_elements[channel_id]->time_interval - (time_count-time_counters[channel_id]).count()*1e-6);

        if (sleep_time > 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        } else {
          auto elapsed_time = (time_count - time_counters[channel_id]).count() * 1e-6; 
          double time_interval = m_decode_elements[channel_id]->time_interval;
          sleep_time = static_cast<int>(time_interval) - static_cast<int>(elapsed_time) % static_cast<int>(time_interval);
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }
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

void YOLOv5::worker_pre(int pre_idx){
  while (true){
    std::vector<std::shared_ptr<DataDec>> dec_images;
    auto pre_data = std::make_shared<DataInfer>();
    int ret = 0;
    bool no_data = false;

    // 取一个batch的数据做预处理
    for (int i = 0; i < m_batch_size; i ++){
      std::shared_ptr<DataDec> data;
      ret = m_queue_decode.pop_front(data);
      if (ret == 0){
        dec_images.emplace_back(data);
      }else{
        if (i == 0){
          no_data = true;
        }
        break;
      }
    }

    // 解码线程停止并且解码队列为空，可以结束工作线程
    if (no_data){
      std::unique_lock<std::mutex> lock(m_mutex_stop_pre);
      if (m_is_stop_decode && ret == -1){
        m_stop_pre ++;
        if (m_stop_pre == m_num_pre){
          m_is_stop_pre = true;
          m_queue_pre.set_stop_flag(true);
        }
        return;
      }
    }

    preprocess(dec_images, pre_data, pre_idx);
    m_queue_pre.push_back(pre_data);

  }
}


void YOLOv5::worker_infer(){
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
    m_queue_infer.push_back(output_data);

  }
}


void YOLOv5::worker_post(){
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

    for (int i = 0; i < box_datas.size(); i++){
      m_queue_post.push_back(box_datas[i]);
    }

  }
}



// ------------------------------处理函数---------------------------

// 调对应的vectore中的decoder
void YOLOv5::decode(std::shared_ptr<DataDec> data, int channel_id){
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
void YOLOv5::preprocess(std::vector<std::shared_ptr<DataDec>> &dec_images, 
                        std::shared_ptr<DataInfer> pre_data, int idx){
  auto resized_bmimgs = m_vec_resized_bmimgs[idx];
  std::vector<bm_image> fuse_resize_bmimgs(m_batch_size);
  bm_device_mem_t fuse_tensor_mem;
  if(is_fuse_preprocess){
    for (int i = 0; i < m_batch_size; i++){
      bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_RGB_PACKED, img_dtype, &fuse_resize_bmimgs[i]);
    }
    int size_byte = 0;
    bm_image_get_byte_size(fuse_resize_bmimgs[0], &size_byte);
    bm_malloc_device_byte_heap(m_handle, &fuse_tensor_mem, 0, size_byte*m_batch_size);
    bm_image_attach_contiguous_mem(m_batch_size, fuse_resize_bmimgs.data(), fuse_tensor_mem);
  }

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
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
      padding_attr.dst_crop_h = bmimg.height*ratio;
      padding_attr.dst_crop_w = m_net_w;

      int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
      padding_attr.dst_crop_sty = ty1;
      padding_attr.dst_crop_stx = 0;
    }else{
      padding_attr.dst_crop_h = m_net_h;
      padding_attr.dst_crop_w = bmimg.width*ratio;

      int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, bmimg.width, bmimg.height};
    auto ret = bmcv_image_vpp_convert_padding(m_handle, 1, bmimg_aligned, 
        is_fuse_preprocess ? &fuse_resize_bmimgs[i] : &resized_bmimgs[i],
        &padding_attr, &crop_rect, BMCV_INTER_NEAREST);

    assert(BM_SUCCESS == ret);

    if (need_copy){
      bm_image_destroy(bmimg_aligned);
    }
    bm_image_destroy(bmimg);
  }
  if(is_fuse_preprocess){
    bm_image_dettach_contiguous_mem(m_batch_size, fuse_resize_bmimgs.data());
    for (int i = 0; i < m_batch_size; i ++){
      bm_image_destroy(fuse_resize_bmimgs[i]);
    }
    for (int i = 0; i < dec_images.size(); i++){
      pre_data->channel_ids.emplace_back(dec_images[i]->channel_id);
      pre_data->frame_ids.emplace_back(dec_images[i]->frame_id);
    }

    bm_tensor_t tensor;
    tensor.device_mem = fuse_tensor_mem;
    pre_data->tensors.emplace_back(tensor);
    return;
  }
  // 这里先bmlib申请了连续batch_size的内存，做归一化的bmimage内存是attach的，
  // 因为后面tensor需要的是相同的dev mem，这里申请的在推理完成后会进行释放（推理函数中）
  std::vector<bm_image> converto_bmimgs(m_batch_size);
  for (int i = 0; i < m_batch_size; i++){
    bm_image_create(m_handle, m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, &converto_bmimgs[i]);
  }

  int size_byte = 0;
  bm_device_mem_t tensor_mem;
  bm_image_get_byte_size(converto_bmimgs[0], &size_byte);
  bm_malloc_device_byte_heap(m_handle, &tensor_mem, 0, size_byte*m_batch_size);
  
  bm_image_attach_contiguous_mem(m_batch_size, converto_bmimgs.data(), tensor_mem);

  auto ret = bmcv_image_convert_to(m_handle, m_batch_size, converto_attr, resized_bmimgs.data(), converto_bmimgs.data());
  assert(BM_SUCCESS == ret);

  bm_image_dettach_contiguous_mem(m_batch_size, converto_bmimgs.data());

  for (int i = 0; i < m_batch_size; i ++){
    bm_image_destroy(converto_bmimgs[i]);
  }

  for (int i = 0; i < dec_images.size(); i++){
    pre_data->channel_ids.emplace_back(dec_images[i]->channel_id);
    pre_data->frame_ids.emplace_back(dec_images[i]->frame_id);
  }

  bm_tensor_t tensor;
  tensor.device_mem = tensor_mem;
  pre_data->tensors.emplace_back(tensor);
  
}


void YOLOv5::inference(std::shared_ptr<DataInfer> input_data, std::shared_ptr<DataInfer> output_data){

  output_data->channel_ids.assign(input_data->channel_ids.begin(), input_data->channel_ids.end());
  output_data->frame_ids.assign(input_data->frame_ids.begin(), input_data->frame_ids.end());

  m_inputs[0]->set_device_mem(input_data->tensors[0].device_mem);

  // output tensor的设备内存在这里申请，在后处理完成后释放（后处理函数中）
  for (int i = 0; i < m_output_num; i++){
    auto out_size = bmruntime::ByteSize(*m_outputs[i]);
    bm_device_mem_t out_mem;
    auto ret = bm_malloc_device_byte_heap(m_handle, &out_mem, 0, out_size);
    assert(BM_SUCCESS == ret);

    bm_tensor_t tensor;
    tensor.device_mem = out_mem;
    tensor.dtype = m_output_dtypes[i];
    // tensor.shape = m_output_shapes[i];

    // output_data->tensors.emplace_back(tensor);
    m_outputs[i]->set_device_mem(out_mem);
  }

  m_inputs[0]->Reshape(m_net->info()->stages[0].input_shapes[0]);
  auto ret = m_net->Forward();
  assert(BM_SUCCESS == ret);

  m_outputs = m_net->Outputs();
  for (int i = 0; i < m_output_num; i++){
    output_data->tensors.emplace_back(*m_outputs[i]->tensor());
  }

  bm_free_device(m_handle, input_data->tensors[0].device_mem);
  
}


void YOLOv5::postprocess(std::shared_ptr<DataInfer> output_infer, std::vector<std::shared_ptr<DataPost>> &box_data){
  if(is_fuse_postprocess){
    YoloV5BoxVec yolobox_vec;
    auto output_tensor = output_infer->tensors[0];
    auto output_data = get_cpu_data(output_tensor, 0);
    int box_num = output_tensor.shape.dims[2];
    // std::cout<<"box_num:"<<box_num<<std::endl;
    int image_nums = output_infer->channel_ids.size();
    int nout = 0;
    int box_id = 0;
    for(int batch_idx = 0; batch_idx < image_nums; ++batch_idx){
      yolobox_vec.clear();
      int channel_id = output_infer->channel_ids[batch_idx];
      int frame_id = output_infer->frame_ids[batch_idx];
      cv::Mat frame;
      {
        std::unique_lock<std::mutex> lock(m_mutex_map_origin);
        frame = m_origin_image[channel_id][frame_id];
      }
      int frame_width = frame.cols;
      int frame_height = frame.rows;

      int tx1 = 0, ty1 = 0;
      float rx = float(frame_width) / m_net_w;
      float ry = float(frame_height) / m_net_h;
      bool isAlignWidth = false;
      float ratio = get_aspect_scaled_ratio(frame_width, frame_height, m_net_w, m_net_h, &isAlignWidth);
      if (isAlignWidth) {
        ty1 = (int)((m_net_h - (int)(frame_height*ratio)) / 2);
      }else{
        tx1 = (int)((m_net_w - (int)(frame_width*ratio)) / 2);
      }
      rx = 1.0 / ratio;
      ry = 1.0 / ratio;

      while(int(*(output_data+nout)) == batch_idx && box_id < box_num){
        if(*(output_data+nout+2) < m_confThreshold){
          box_id++;
          nout += 7;
          continue;
        }
        YoloV5Box box;
        // init box data
        float centerX = *(output_data+nout+3);
        float centerY = *(output_data+nout+4);
        float width = *(output_data+nout+5);
        float height = *(output_data+nout+6);
        // get bbox
        box.x = int((centerX - width / 2 - tx1) * rx);
        if (box.x < 0) box.x = 0;
        box.y = int((centerY - height / 2  - ty1) * ry);
        if (box.y < 0) box.y = 0;
        box.width = width * rx;
        box.height = height * ry;

        box.class_id = int(*(output_data+nout+1));
        box.score = *(output_data+nout+2);
        yolobox_vec.emplace_back(box);
        box_id++;
        nout += 7;
      }

  // std::cout<<"channel_id: "<<channel_id <<"; frame_id:"<<frame_id <<"box_num: "<<yolobox_vec.size()<<std::endl;
      std::shared_ptr<DataPost> data_post = std::make_shared<DataPost>();
      data_post->channel_id = output_infer->channel_ids[batch_idx], 
      data_post->frame_id = output_infer->frame_ids[batch_idx],
      data_post->boxes = yolobox_vec;
      box_data.emplace_back(data_post);
    }

    // 释放output tensor内存
    int tensor_num = output_infer->tensors.size();
    for (int i = 0; i < tensor_num; i++){
      bm_free_device(m_handle, output_infer->tensors[i].device_mem);
    }
    if (can_mmap && BM_FLOAT32 == output_tensor.dtype){
      int tensor_size = bm_mem_get_device_size(output_tensor.device_mem);
      // std::cout<<"tensor_size:"<<tensor_size<<std::endl;
      bm_status_t ret = bm_mem_unmap_device_mem(m_handle, output_data, tensor_size);
      assert(BM_SUCCESS == ret);
    }else{
      delete output_data;
    }
    return;
  }

  YoloV5BoxVec yolobox_vec;
  auto output_tensors = output_infer->tensors;
  int image_nums = output_infer->channel_ids.size();
  std::vector<float*> tensor_datas(m_output_num);
  for (int tidx = 0; tidx < m_output_num; tidx++){
    tensor_datas[tidx] = get_cpu_data(output_tensors[tidx], tidx);
  }
  for(int batch_idx = 0; batch_idx < image_nums; ++ batch_idx)
  {
    yolobox_vec.clear();
    int channel_id = output_infer->channel_ids[batch_idx];
    int frame_id = output_infer->frame_ids[batch_idx];
    cv::Mat frame;
    {
      std::unique_lock<std::mutex> lock(m_mutex_map_origin);
      frame = m_origin_image[channel_id][frame_id];
    }
    int frame_width = frame.cols;
    int frame_height = frame.rows;

    int tx1 = 0, ty1 = 0;

    bool is_align_width = false;
    float ratio = get_aspect_scaled_ratio(frame_width, frame_height, m_net_w, m_net_h, &is_align_width);
    if (is_align_width) {
      ty1 = (int)((m_net_h - (int)(frame_height*ratio)) / 2);
    }else{
      tx1 = (int)((m_net_w - (int)(frame_width*ratio)) / 2);
    }

    int box_num = 0;

    auto output_shape = output_tensors[0].shape;
    auto output_dims = output_shape.num_dims;
    assert(output_dims == 3 || output_dims == 5);
    if(output_dims == 5){
      box_num += output_shape.dims[1] * output_shape.dims[2] * output_shape.dims[3];
    }

    int min_dim = output_dims;
   
    int nout = output_tensors[0].shape.dims[output_dims-1];
    m_class_num = nout - 5;
#if USE_MULTICLASS_NMS
    int out_nout = nout;
#else
    int out_nout = 7;
#endif
    float transformed_m_confThreshold = - std::log(1 / m_confThreshold - 1);

    float* output_data = nullptr;
    std::vector<float> decoded_data;

    if(min_dim ==3 && m_output_num !=1){
      std::cout<<"--> WARNING: the current bmodel has redundant outputs"<<std::endl;
      std::cout<<"             you can remove the redundant outputs to improve performance"<< std::endl;
      std::cout<<std::endl;
    }

    if(min_dim == 5){

      // std::cout<<"--> Note: Decoding Boxes"<<std::endl;
      // std::cout<<"          you can put the process into model during trace"<<std::endl;
      // std::cout<<"          which can reduce post process time, but forward time increases 1ms"<<std::endl;
      // std::cout<<std::endl;
      const std::vector<std::vector<std::vector<int>>> anchors{
        {{10, 13}, {16, 30}, {33, 23}},
          {{30, 61}, {62, 45}, {59, 119}},
          {{116, 90}, {156, 198}, {373, 326}}};
      const int anchor_num = anchors[0].size();
      assert(m_output_num == (int)anchors.size());
      assert(box_num>0);
      if((int)decoded_data.size() != box_num*out_nout){
        decoded_data.resize(box_num*out_nout);
      }

      float *dst = decoded_data.data();
      for(int tidx = 0; tidx < m_output_num; ++tidx) {
        auto output_tensor = output_tensors[tidx];
        int feat_c = output_tensor.shape.dims[1];
        int feat_h = output_tensor.shape.dims[2];
        int feat_w = output_tensor.shape.dims[3];
        int area = feat_h * feat_w;
        assert(feat_c == anchor_num);
        int feature_size = feat_h*feat_w*nout;
        
        float* tensor_data = tensor_datas[tidx] + batch_idx*feat_c*area*nout;
        
        for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++)
        {
          float *ptr = tensor_data + anchor_idx*feature_size;
          for (int i = 0; i < area; i++) {
            if(ptr[4] <= transformed_m_confThreshold){
              ptr += nout;
              continue;
            }
            dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
            dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * m_net_h;
            dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
            dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
            dst[4] = sigmoid(ptr[4]);
#if USE_MULTICLASS_NMS
            for(int d = 5; d < nout; d++)
                dst[d] = ptr[d];
#else
            dst[5] = ptr[5];
            dst[6] = 5;
            for(int d = 6; d < nout; d++){
              if(ptr[d] > dst[5]){
                dst[5] = ptr[d];
                dst[6] = d;
              }
            }
            dst[6] -= 5;
#endif
            dst += out_nout;
            ptr += nout;
          }
        }
      }
      output_data = decoded_data.data();
      box_num = (dst - output_data) / out_nout;

    } else {

      assert(box_num == 0 || box_num == output_tensors[0].shape.dims[1]);
      box_num = output_tensors[0].shape.dims[1];
      output_data = tensor_datas[0] + batch_idx*box_num*nout;

    }

    int max_wh = 7680;
    bool agnostic = false;
    for (int i = 0; i < box_num; i++) {
      float* ptr = output_data+i*out_nout;
      float score = ptr[4];
      float box_transformed_m_confThreshold = - std::log(score / m_confThreshold - 1);
      if(min_dim != 5)
          box_transformed_m_confThreshold = m_confThreshold /score;
#if USE_MULTICLASS_NMS
      float centerX = ptr[0];
      float centerY = ptr[1];
      float width = ptr[2];
      float height = ptr[3];
      for (int j = 0; j < m_class_num; j++) {
        float confidence = ptr[5 + j];
        int class_id = j;
        if (confidence > box_transformed_m_confThreshold)
        {
            YoloV5Box box;
            if (!agnostic)
                box.x = centerX - width / 2 + class_id * max_wh;
            else
                box.x = centerX - width / 2;
            if (box.x < 0) box.x = 0;
            if (!agnostic)
                box.y = centerY - height / 2 + class_id * max_wh;
            else
                box.y = centerY - height / 2;
            if (box.y < 0) box.y = 0;
            box.width = width;
            box.height = height;
            box.class_id = class_id;
            box.score = sigmoid(confidence) * score;
            yolobox_vec.push_back(box);
        }
      }
#else
      int class_id = ptr[6];
      float confidence = ptr[5];
      if(min_dim != 5){
        ptr = output_data+i*nout;
        score = ptr[4];
        class_id = argmax(&ptr[5], m_class_num);
        confidence = ptr[class_id + 5];
      }
      if (confidence > box_transformed_m_confThreshold)
      {
          float centerX = ptr[0];
          float centerY = ptr[1];
          float width = ptr[2];
          float height = ptr[3];

          YoloV5Box box;
          if (!agnostic)
            box.x = centerX - width / 2 + class_id * max_wh;
          else
            box.x = centerX - width / 2;
          if (box.x < 0) box.x = 0;
          if (!agnostic)
            box.y = centerY - height / 2 + class_id * max_wh;
          else
            box.y = centerY - height / 2;
          if (box.y < 0) box.y = 0;
          box.width = width;
          box.height = height;
          box.class_id = class_id;
          if(min_dim == 5)
              confidence = sigmoid(confidence);
          box.score = confidence * score;
          yolobox_vec.push_back(box);
      }
#endif
    }

    NMS(yolobox_vec, m_nmsThreshold);
    if (!agnostic)
      for (auto& box : yolobox_vec){
          box.x -= box.class_id * max_wh;
          box.y -= box.class_id * max_wh;
          box.x = (box.x - tx1) / ratio;
          if (box.x < 0) box.x = 0;
          box.y = (box.y - ty1) / ratio;
          if (box.y < 0) box.y = 0;
          box.width = (box.width) / ratio;
          if (box.x + box.width >= frame_width)
              box.width = frame_width - box.x;
          box.height = (box.height) / ratio;
          if (box.y + box.height >= frame_height)
              box.height = frame_height - box.y;
      }
    else
      for (auto& box : yolobox_vec){
          box.x = (box.x - tx1) / ratio;
          if (box.x < 0) box.x = 0;
          box.y = (box.y - ty1) / ratio;
          if (box.y < 0) box.y = 0;
          box.width = (box.width) / ratio;
          if (box.x + box.width >= frame_width)
              box.width = frame_width - box.x;
          box.height = (box.height) / ratio;
          if (box.y + box.height >= frame_height)
              box.height = frame_height - box.y;
      }

    std::shared_ptr<DataPost> data_post = std::make_shared<DataPost>();
    data_post->channel_id = output_infer->channel_ids[batch_idx], 
    data_post->frame_id = output_infer->frame_ids[batch_idx],
    data_post->boxes = yolobox_vec;
    box_data.emplace_back(data_post);
  }
 
  // 释放output tensor内存
  int tensor_num = output_infer->tensors.size();
  for (int i = 0; i < tensor_num; i++){
    bm_free_device(m_handle, output_infer->tensors[i].device_mem);
  }

  
  for (int i = 0; i < m_output_num; i++){
    if (can_mmap && BM_FLOAT32 == output_tensors[i].dtype){
      int tensor_size = bm_mem_get_device_size(output_tensors[i].device_mem);
      bm_status_t ret = bm_mem_unmap_device_mem(m_handle, tensor_datas[i], tensor_size);
      assert(BM_SUCCESS == ret);
    }else{
      delete tensor_datas[i];
    }
  }

}


int YOLOv5::get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image, std::string& image_name){

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

int YOLOv5::get_post_data(std::shared_ptr<DataPost>& box_data, std::shared_ptr<cv::Mat>& origin_image){
  std::string name;
  return get_post_data(box_data, origin_image, name);
}

float YOLOv5::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w){
    *pIsAligWidth = true;
    ratio = r_w;
  }
  else{
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

float *YOLOv5::get_cpu_data(bm_tensor_t &tensor, int out_idx) {
  float* cpu_data;
  bm_status_t ret;
  float *pFP32 = nullptr;
  int count = bmrt_shape_count(&tensor.shape);
  // in SOC mode, device mem can be mapped to host memory, faster then using d2s
  if(can_mmap){
    if (tensor.dtype == BM_FLOAT32) {
      unsigned long long  addr;
      ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
      assert(BM_SUCCESS == ret);
      pFP32 = (float*)addr;
    } else if (BM_INT8 == tensor.dtype) {
      int8_t * pI8 = nullptr;
      unsigned long long  addr;
      ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
      assert(BM_SUCCESS == ret);
      pI8 = (int8_t*)addr;

      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      for(int i = 0;i < count; ++ i) {
        pFP32[i] = pI8[i] * m_output_scales[out_idx];
      }
      ret = bm_mem_unmap_device_mem(m_handle, pI8, bm_mem_get_device_size(tensor.device_mem));
      assert(BM_SUCCESS == ret);
    }else if (tensor.dtype == BM_INT32) {
      int32_t * pI32 = nullptr;
      unsigned long long  addr;
      ret = bm_mem_mmap_device_mem(m_handle, &tensor.device_mem, &addr);
      assert(BM_SUCCESS == ret);
      ret = bm_mem_invalidate_device_mem(m_handle, &tensor.device_mem);
      assert(BM_SUCCESS == ret);
      pI32 = (int32_t*)addr;
      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      for(int i = 0;i < count; ++ i) {
        pFP32[i] = pI32[i] * m_output_scales[out_idx];
      }
      ret = bm_mem_unmap_device_mem(m_handle, pI32, bm_mem_get_device_size(tensor.device_mem));
      assert(BM_SUCCESS == ret);
    } else{
      std::cout << "NOT support dtype=" << tensor.dtype << std::endl;
    }
  } else {
    // the common method using d2s
    if (tensor.dtype == BM_FLOAT32) {
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      ret = bm_memcpy_d2s_partial(m_handle, pFP32, tensor.device_mem, count * sizeof(float));
      assert(BM_SUCCESS ==ret);
    } else if (BM_INT8 == tensor.dtype) {
      int8_t * pI8 = nullptr;
      int tensor_size = bmrt_tensor_bytesize(&tensor);
      pI8 = new int8_t[tensor_size];
      assert(pI8 != nullptr);

      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      ret = bm_memcpy_d2s_partial(m_handle, pI8, tensor.device_mem, tensor_size);
      assert(BM_SUCCESS ==ret);
      for(int i = 0;i < count; ++ i) {
        pFP32[i] = pI8[i] * m_output_scales[out_idx];
      }
      delete [] pI8;
    }else if(tensor.dtype == BM_INT32){
      int32_t *pI32=nullptr;
      int tensor_size = bmrt_tensor_bytesize(&tensor);
      pI32 =new int32_t[tensor_size];
      assert(pI32 != nullptr);

      // dtype convert
      pFP32 = new float[count];
      assert(pFP32 != nullptr);
      ret = bm_memcpy_d2s_partial(m_handle, pI32, tensor.device_mem, tensor_size);
      assert(BM_SUCCESS ==ret);
      for(int i = 0;i < count; ++ i) {
        pFP32[i] = pI32[i] * m_output_scales[out_idx];
      }
      delete [] pI32;
      
    }
      else{
      std::cout << "NOT support dtype=" << tensor.dtype << std::endl;
    }
  }

  cpu_data = pFP32;
  return cpu_data;
}


int YOLOv5::argmax(float* data, int num){
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

float YOLOv5::sigmoid(float x){
  return 1.0 / (1 + expf(-x));
}

void YOLOv5::NMS(YoloV5BoxVec &dets, float nmsConfidence)
{
  int length = dets.size();
  int index = length - 1;

  std::sort(dets.begin(), dets.end(), [](const YoloV5Box& a, const YoloV5Box& b) {
      return a.score < b.score;
      });

  std::vector<float> areas(length);
  for (int i=0; i<length; i++)
  {
    areas[i] = dets[i].width * dets[i].height;
  }

  while (index  > 0)
  {
    int i = 0;
    while (i < index)
    {
      float left    = std::max(dets[index].x,   dets[i].x);
      float top     = std::max(dets[index].y,    dets[i].y);
      float right   = std::min(dets[index].x + dets[index].width,  dets[i].x + dets[i].width);
      float bottom  = std::min(dets[index].y + dets[index].height, dets[i].y + dets[i].height);
      float overlap = std::max(0.0f, right - left + 0.00001f) * std::max(0.0f, bottom - top + 0.00001f);
      if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence)
      {
        areas.erase(areas.begin() + i);
        dets.erase(dets.begin() + i);
        index --;
      }
      else
      {
        i++;
      }
    }
    index--;
  }
}


void YOLOv5::worker_pressure(){
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

    std::cout << "yolov5 fps: " << float(diff)/(INTERVAL) << std::endl;
    
  }
  
}


