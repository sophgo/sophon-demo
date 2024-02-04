// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#ifndef CHANNEL_DECODER_H_
#define CHANNEL_DECODER_H_

#include"ff_codec/ff_video_decode.h"
#include"ff_codec/ff_avframe_convert.h"
#include<string>
#include<thread>
#include<mutex>
#include<condition_variable>
#include<deque>
#include<memory>
#include<utility>
#include<atomic>


class ChannelDecoder{
public:

    ChannelDecoder(std::string url,int dev_id=0,int que_size=1):url_(url),dev_id_(dev_id),max_que_size_(que_size),stop_flag_(true){
        bm_dev_request(&handle_, dev_id_);

    }
    ~ChannelDecoder(){
        stop();
        thread_->join();
        bm_dev_free(handle_);
    }

    int get_image(std::shared_ptr<bm_image>& out_img);

    void start(){
        stop_flag_ = false;
        std::cout<<"Stream: "+url_+" start!"<<std::endl;
        thread_=std::make_shared<std::thread>(&ChannelDecoder::dowork,this);
    }
    void stop(){
        stop_flag_ = true;
        std::cout<<"Stream: "+url_+" stop!"<<std::endl;
    }

private:
  
    void push_image(std::shared_ptr<bm_image> bmimg);
    void dowork();

    std::string url_;
    int dev_id_;
    int max_que_size_;
    bm_handle_t handle_;

    std::deque<std::shared_ptr<bm_image>> img_que_;
    std::mutex img_que_mtx_;
    std::condition_variable que_empty_cv_;
    
    std::atomic<bool> stop_flag_;
    std::shared_ptr<std::thread> thread_;

    VideoDec_FFMPEG reader;
    AVFrame *frame;
};

#endif