// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#ifndef YOLOV5_H_
#define YOLOV5_H_


#include <memory>
#include <deque>
#include <mutex>
#include <thread>
#include <cassert>
#include <atomic>

#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include "bmodel_utils.h"
#include "datapipe.h"
#include "common.h"
#include "opencv2/opencv.hpp"


#define USE_ASPECT_RATIO 1
#define MAX_CHANNEL_NUM 32



class Yolov5{
public:
    Yolov5(int dev_id, std::string bmodel_file,std::string tpu_kernel_module_path,int que_size=1,int skip_num=0,float nmsThreshold=0.5,float confThreshold=0.5)
    :que_size_(que_size),pre_forward_que_(que_size),forward_post_que_(que_size),out_que_(que_size),skip_num_(skip_num),nmsThreshold_(nmsThreshold),confThreshold_(confThreshold){
        channel_skip_nums_ = std::vector<int>(MAX_CHANNEL_NUM,0);
        init(dev_id,bmodel_file,tpu_kernel_module_path);
        start();
    }
    ~Yolov5(){
        stop();
        release();
    }

    void start(){
        stop_flag_=false;
        thread_vec_.push_back(std::make_shared<std::thread>(&Yolov5::forward_thread_dowork, this));
        thread_vec_.push_back(std::make_shared<std::thread>(&Yolov5::post_thread_dowork, this));

    }

    void stop(){
        stop_flag_=true;
        pre_forward_que_.stop();
        forward_post_que_.stop();
        out_que_.stop();
        for(int i=0; i<thread_vec_.size();i++){
            thread_vec_[i]->join();
        }
    }

    int push_img(int channel_id,std::shared_ptr<bm_image> img){
        int ret=0;
        if(channel_skip_nums_[channel_id]==skip_num_)
        {
            std::unique_lock<std::mutex> lock(global_frame_que_mtx_);
            global_frame_que_.push_back(std::make_shared<FrameInfoDetect>(channel_id,img));
            lock.unlock();
            ret = preprocess(img);
            channel_skip_nums_[channel_id]=0;
        }
        else{
            out_que_.push_back(std::make_shared<FrameInfoDetect>(channel_id,img));
            channel_skip_nums_[channel_id]++;
        }
        return ret;

    }

    std::shared_ptr<FrameInfoDetect> get_img(){
        return out_que_.pop_front();
    }
    

private:
    bm_handle_t handle_;
    int que_size_;
    int net_width_,net_height_;
    int in_tensor_num_,out_tensor_num_;
    bm_status_t ret_;
    std::atomic<bool> stop_flag_;
    std::vector<std::shared_ptr<std::thread>> thread_vec_;
    
    //skip frame
    int skip_num_;
    std::vector<int> channel_skip_nums_;
    //preprocess
    bm_image resized_img_;
    bmcv_convert_to_attr converto_attr_;
    std::vector<bm_image> converto_imgs_;

    //forward
    std::vector<bm_device_mem_t> input_mem_buffer_;
    int input_mem_idx_;
    std::vector<std::vector<bm_device_mem_t>> output_mem_buffer_;
    int output_mem_idx_;
    std::shared_ptr<BModelContext> ctx_ptr_;
    std::shared_ptr<BModelNetwork> net_ptr_;

    //tpu kernal post process
    float nmsThreshold_;
    float confThreshold_;
    tpu_kernel_api_yolov5NMS_t api_;
    tpu_kernel_function_t func_id;
    bm_device_mem_t boxs_devmem_;
    bm_device_mem_t detect_num_devmem_;
    std::vector<float> boxs_sysmem_;
    int32_t detect_num_sysmem_;
    const std::vector<std::vector<std::vector<int>>> anchors{{{10, 13}, {16, 30}, {33, 23}}, {{30, 61}, {62, 45}, {59, 119}}, {{116, 90}, {156, 198}, {373, 326}}} ;

    //queue
    std::mutex global_frame_que_mtx_;
    std::deque<std::shared_ptr<FrameInfoDetect>> global_frame_que_;
    DataPipe<bm_device_mem_t*> pre_forward_que_;
    DataPipe<bm_device_mem_t*> forward_post_que_;
    DataPipe<std::shared_ptr<FrameInfoDetect>> out_que_;
    // DataPipe<std::shared_ptr<FrameInfo>> out_que_;

    int init(int dev_id,std::string bmodel_file,std::string tpu_kernel_module_path);
    int release();

    int get_input_mem_id(){
        return (++input_mem_idx_)%(que_size_+2);
    }
    int get_output_mem_id(){
        return (++output_mem_idx_)%(que_size_+2);
    }

    
    int preprocess(std::shared_ptr<bm_image> img);
    void forward_thread_dowork();
    void post_thread_dowork();
    void draw_thread_dowork();
    void draw_bmcv(bm_handle_t& handle,int classId,float conf,int left,int top,int width,int height,bm_image& frame,bool put_text_flag);
};



#endif
