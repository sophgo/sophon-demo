//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <unistd.h>
#include "vlpr_bmcv.hpp"

#define DET_VIS 0
// print fps info during run, otherwise output fps info finally
#define RUNTIME_PERFORMANCE 0

VLPR::VLPR(demo_config& config)
    : in_frame_num(config.in_frame_num),
      out_frame_num(config.out_frame_num),
      crop_activate_thread_num(config.crop_thread_num),
      push_data_activate_thread_num(config.push_data_thread_num) {
  channel_num = config.input_paths.size();
  for (int i = 0; i < channel_num; i++) {
    m_num_map[i] = std::unordered_map<int, std::shared_ptr<rec_single_frame>>();
  }

  statics.resize(channel_num);

  this->config = config;
  total_frame_num = 0;
  assert(BM_SUCCESS == bm_dev_request(&m_handle, config.dev_id));
  runtime_performance_info_thread_exit = false;

#if DET_VIS
  // save det results
  if (access("vis", 0) != F_OK)
    mkdir("vis", S_IRWXU);
#endif
}

VLPR::~VLPR() {
  if (lprnet != nullptr)
   delete lprnet;
  if (yolov5 != nullptr)
    delete yolov5;
}

void VLPR::run() {
  // start thread
  auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;

  // skip frame vector
  std::vector<int> skip_frame_nums;
  for (int i = 0; i < channel_num; i++)
    skip_frame_nums.push_back(config.frame_sample_interval);

  // start yolov5 threads
  yolov5 = new YOLOv5(config.dev_id, 
                config.yolov5_bmodel_path, 
                config.input_paths, 
                config.is_videos,
                skip_frame_nums,
                config.yolov5_queue_size,
                config.yolov5_num_pre,
                config.yolov5_num_post,
                config.yolov5_conf_thresh,
                config.yolov5_nms_thresh);

  for (int i = 0; i < config.crop_thread_num; i++){
    threads.emplace_back(&VLPR::crop, this, i);
  }

  // start lprnet threads
  lprnet = new LPRNet(config.dev_id, config.lprnet_bmodel_path,
         config.lprnet_num_pre, config.lprnet_num_post, config.lprnet_queue_size);
  lprnet->run();

  for (int p = 0; p < config.push_data_thread_num; p++)
    threads.emplace_back(&VLPR::push_data, this, p);
  for (int p = 0; p < channel_num; p++)
    threads.emplace_back(&VLPR::worker, this, p);

#if RUNTIME_PERFORMANCE
  std::thread print_thread(&VLPR::output_runtime_performance_info, this);
#endif

  // sync all threads
  for (std::thread& t : threads) t.join();
  delete yolov5;
  yolov5 = nullptr;
  delete lprnet;
  lprnet = nullptr;
#if RUNTIME_PERFORMANCE
  runtime_performance_info_thread_exit = true;
  runtime_performance_info_thread_exit_cv.notify_all();
  if (print_thread.joinable())
    print_thread.join();
#endif

#if !RUNTIME_PERFORMANCE
  // output run info
  auto end = std::chrono::steady_clock::now();
  std::cout << "total frame num: " << total_frame_num << std::endl;
  std::cout << "total time cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms." << std::endl;
  std::cout << "total FPS: " << total_frame_num * 1000.0 * 1000.0 / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
#endif
}

void VLPR::output_runtime_performance_info() {
  while (!runtime_performance_info_thread_exit) {
    auto start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(m_mutex_total_frame_num);
    total_frame_num = 0;
    runtime_performance_info_thread_exit_cv.wait_for(lock, std::chrono::seconds(config.perf_out_time_interval));

    auto end = std::chrono::steady_clock::now();
    std::cout << "current frame num: " << total_frame_num << std::endl;
    std::cout << "current time cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms." << std::endl;
    std::cout << "current FPS: " << total_frame_num * 1000.0 * 1000.0 / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
  }
}

void VLPR::crop(int process_id) {
  // for dilating small boxes to contain whole charss
  float dilated_ratio_h = 0.1, dilated_ratio_w = 0;
  while (true) {
    std::shared_ptr<DataPost> box_data;
    std::shared_ptr<cv::Mat> image;
    // pop yolov5 boxes and images data
    if (yolov5->get_post_data(box_data, image) != 0)
      break;

    // save base infor
    int frame_id = box_data->frame_id;
    int channel_id = box_data->channel_id;
    auto& boxes = box_data->boxes;
    {
      std::unique_lock<std::mutex> lock(m_mutex_num_map);
      m_num_map[channel_id][frame_id] = std::make_shared<rec_single_frame>();
      m_num_map[channel_id][frame_id]->num = boxes.size();
    }

    // some API only accept bm_image whose stride is aligned to 64
    bm_image bmimg;
    cv::bmcv::toBMI(*(image.get()), &bmimg);
    bm_image image_aligned;
    bool need_copy = bmimg.width & (64 - 1);
    if (need_copy) {
      int stride1[3], stride2[3];
      bm_image_get_stride(bmimg, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_handle, bmimg.height, bmimg.width, bmimg.image_format,
                      bmimg.data_type, &image_aligned, stride2);

      bm_image_alloc_dev_mem(image_aligned, VPP_HEAP_ID);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;

      bmcv_image_copy_to(m_handle, copyToAttr, bmimg, image_aligned);
    } else {
      image_aligned = bmimg;
    }

    // crop code
    for (auto& bbox : boxes) {
      // dilate boxes
      int center_x = bbox.x + bbox.width / 2;
      int center_y = bbox.y + bbox.height / 2;
      int dilated_w = bbox.width + bbox.width * dilated_ratio_w * 2;
      dilated_w = std::min((image->cols - center_x) * 2, dilated_w);
      dilated_w = std::min(center_x * 2, dilated_w);
      int dilated_h = bbox.height + bbox.height * dilated_ratio_h * 2;
      dilated_h = std::min((image->rows - center_y) * 2, dilated_h);
      dilated_h = std::min(center_y * 2, dilated_h);
      int dilated_left = center_x - dilated_w / 2;
      int dilated_top = center_y - dilated_h / 2;

      if (dilated_h < 16 || dilated_w < 16)
      {
        // drop invalid box
        std::unique_lock<std::mutex> lock(m_mutex_num_map);
        m_num_map[channel_id][frame_id]->num--;
        continue;
      }

      std::shared_ptr<bmimage> croped_bmimg = std::make_shared<bmimage>();
      croped_bmimg->bmimg = std::make_shared<bm_image>();
      croped_bmimg->bmimg.reset(new bm_image(), [](bm_image* p) {
        bm_image_destroy(*p);
        delete p;
        p = nullptr;
      });
      int aligned_box_w = FFALIGN(dilated_w, 64);
      int ratio = 3 * 1; // for format: FORMAT_BGR_PACKED, data_type: DATA_TYPE_EXT_1N_BYTE
      int strides[3] = {aligned_box_w * ratio, aligned_box_w * ratio, aligned_box_w * ratio};
      bm_image_create(m_handle, dilated_h, dilated_w,
                      image_aligned.image_format, image_aligned.data_type,
                      croped_bmimg->bmimg.get(), strides);
      croped_bmimg->frame_id = frame_id;
      croped_bmimg->channel_id = channel_id;
      bmcv_rect_t crop_rect = {.start_x = dilated_left,
                               .start_y = dilated_top,
                               .crop_w = dilated_w,
                               .crop_h = dilated_h};
      assert(BM_SUCCESS == bmcv_image_vpp_convert(m_handle, 1, image_aligned,
                                                  croped_bmimg->bmimg.get(),
                                                  &crop_rect));
      lprnet->push_m_queue_decode(croped_bmimg);

#if DET_VIS
      cv::rectangle(*(image.get()), cv::Point(dilated_left, dilated_top), cv::Point(dilated_left + dilated_w, dilated_top + dilated_h), cv::Scalar(0, 0, 255), 2.5);
    }

      std::string img_file = "vis/" + std::to_string(channel_id) + "_" + std::to_string(frame_id) + ".jpg";
      cv::imwrite(img_file, *(image.get()));
#else
    }
#endif
    if (need_copy) bm_image_destroy(image_aligned);
    bm_image_destroy(bmimg);
  }

  {
    std::unique_lock<std::mutex> lock(m_mutex_crop);
    crop_activate_thread_num--;
    if (crop_activate_thread_num <= 0) lprnet->set_preprocess_exit();
  }
  std::cout << "crop thread " << process_id << " exit..." << std::endl;
}

void VLPR::worker(int channel_id) {
  // NOTE: frame id maybe bigger than INT_MAX, please mod this variable
  int cur_frame_id = 0;
  while (true) {
    // get data
    std::shared_ptr<rec_single_frame> out;
    {
      std::unique_lock<std::mutex> lock(m_mutex_num_map);
      if (m_num_map[channel_id].size() == 0 &&
          push_data_activate_thread_num <= 0)
        break;
      if (m_num_map[channel_id].find(cur_frame_id) ==
              m_num_map[channel_id].end() ||
          m_num_map[channel_id][cur_frame_id]->cur_num !=
              m_num_map[channel_id][cur_frame_id]->num) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      out = m_num_map[channel_id][cur_frame_id];
      m_num_map[channel_id].erase(cur_frame_id);
    }
    {
      std::unique_lock<std::mutex> lock(m_mutex_total_frame_num);
      total_frame_num++;
    }

    // judge 'out' or 'in' vehicle license plate
    std::vector<std::string> erase_list;
    for (auto& kv : statics[channel_id]) {
      if (out->rec_res.find(kv.first) == out->rec_res.end() &&
          cur_frame_id - kv.second.first >= out_frame_num)
        erase_list.push_back(kv.first);
      else if (out->rec_res.find(kv.first) != out->rec_res.end()) {
        if (kv.second.first != cur_frame_id - 1) {
          kv.second.first = cur_frame_id;
          kv.second.second = 1;
        } else {
          kv.second.first = cur_frame_id;
          kv.second.second += 1;
        }
        if (kv.second.second == in_frame_num)
          std::cout << "detected valid vehicle license plate: " << kv.first
                    << ", frame id: " << cur_frame_id
                    << std::endl;
      }
    }
    // remove 'out' vehicle license plate
    for (auto& erase_item : erase_list) statics[channel_id].erase(erase_item);
    // add new vehicle license plate
    for (auto& kv : out->rec_res) {
      if (statics[channel_id].find(kv.first) == statics[channel_id].end())
        statics[channel_id][kv.first] = std::make_pair(cur_frame_id, 1);
    }
    cur_frame_id++;
  }
}

void VLPR::push_data(int process_id) {
  while (true) {
    std::shared_ptr<rec_data> out;
    if (lprnet->pop_m_queue_post(out) != 0) {
      std::unique_lock<std::mutex> lock(m_mutex_num_map);
      push_data_activate_thread_num--;
      std::cout << "push_data thread exit..." << std::endl;
      break;
    }
    std::unique_lock<std::mutex> lock(m_mutex_num_map);
    m_num_map[out->channel_id][out->frame_id]->rec_res[out->rec_res] = 1;
    m_num_map[out->channel_id][out->frame_id]->cur_num += 1;
  }
  std::cout << "push_data thread " << process_id << " exit..." << std::endl;
}