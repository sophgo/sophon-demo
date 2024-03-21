// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "DecoderConsole.h"

int DecoderConsole::get_channel_num(){
    std::lock_guard<std::mutex> lock(dec_map_mtx_);
    return dec_map_.size();
}


void DecoderConsole::addChannel(std::string url, int dev_id){
    int channel_idx = get_channel_num();
    dec_map_.insert(std::make_pair(channel_idx,std::make_shared<ChannelDecoder>(url,dev_id,que_size_)));
    dec_map_[channel_idx]->start();
    std::cout<<"Decoder channel_idx "<<channel_idx<<" start!"<<std::endl;
}


int DecoderConsole::read(int channel_idx, std::shared_ptr<bm_image>& out_image){
    std::lock_guard<std::mutex> lock(dec_map_mtx_);
    if(dec_map_.find(channel_idx)==dec_map_.end()){
        std::cout<<"error!channel_idx doesn't exist!"<<std::endl;
        return -1;
    }
    return dec_map_[channel_idx]->get_image(out_image);
}
