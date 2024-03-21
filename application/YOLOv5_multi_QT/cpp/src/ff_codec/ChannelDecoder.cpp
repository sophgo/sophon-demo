// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "ChannelDecoder.h"

int ChannelDecoder::get_image(std::shared_ptr<bm_image>& out_img){
    while(true){
        if(stop_flag_) break;

        std::unique_lock<std::mutex> lock(img_que_mtx_);
        if(img_que_.size()==0){
            return -1;
        }
        out_img = img_que_.front();
        img_que_.pop_front();
        return 0;   
    }
    return 1;
}

void ChannelDecoder::push_image(std::shared_ptr<bm_image> bmimg){
    std::lock_guard<std::mutex> lock(img_que_mtx_);
    if(img_que_.size()>=max_que_size_){
        img_que_.pop_front();
    }
    img_que_.push_back(bmimg);
}


void ChannelDecoder::dowork(){
    unsigned int reconnet_times = 0;
    while(!stop_flag_)
    {
        std::cout<<"reconnet stream times: "<<reconnet_times++<<"  url:"+url_<<std::endl;
        int ret = reader.openDec(url_.c_str(), 0,"no", 101, dev_id_,1);
        if (ret < 0){
            printf("open input media failed\n");
            usleep(1000 * 1000);
            continue;
        }
        while (true) {

            if(stop_flag_)break;

            frame = av_frame_alloc();
            int got_frame = reader.grabFrame(frame);
            if(!got_frame)
            {
                printf("no frame!\n");
                av_frame_unref(frame);
                av_frame_free(&frame);
                break;
            }

            bm_image temp_img;
            if(avframe_to_bm_image(handle_,*frame,temp_img)!= BM_SUCCESS){
                printf("avframe to bm_image failed!\n");
                break;
            }
            std::shared_ptr<bm_image> img_out(new bm_image, [](bm_image* img){bm_image_destroy(*img);
                                                                                delete img;});
            // int w = FFALIGN(temp_img.width, 64);
            // int strides[3] = {w*3, w*3, w*3};   
            bm_image_create(handle_,temp_img.height,temp_img.width,FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE,img_out.get());
            bmcv_image_storage_convert(handle_,1,&temp_img,img_out.get());
            push_image(img_out);
            // auto currentTime1 = std::chrono::high_resolution_clock::now();
            // std::cout << dec_id << " get frame : " << std::chrono::duration_cast<std::chrono::milliseconds>(currentTime1.time_since_epoch()).count() << std::endl;
            bm_image_destroy(temp_img);
            av_frame_unref(frame);
            av_frame_free(&frame);
        }
        reader.closeDec();
    }
}