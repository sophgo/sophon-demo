//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef DATAPIPE_H_
#define DATAPIPE_H_

#include <deque>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include "adapt.hpp"


/******************************************
 *  DataPipe主要实现了一个线程安全的队列，可设置队列的最大长度
 *  可对队列已满时的处理策略进行配置：DISCARD_MODE 0 drop front, 1 drop back, 2 block
******************************************/

#define PIPE_INFO 1     // 是否打印管道已满的log

template <typename T>
class DataPipe{
public:
    DataPipe(std::string name, int max_len,int discard_mode=2): name_(name), max_size_(max_len),discard_mode_(discard_mode),stop_flag_(false){};
    ~DataPipe(){
        {
            std::lock_guard<std::mutex> lock(stop_flag_mtx_);
            stop_flag_=true;
        }
        que_empty_cv_.notify_all();
        que_full_cv_.notify_all();
    }

    int size(){
        std::lock_guard<std::mutex> lock(data_que_mtx_);
        return data_que_.size();
    }

    bool get_stop_flag(){
        std::lock_guard<std::mutex> lock(stop_flag_mtx_);
        return stop_flag_;
    }

    void set_stop_flag(bool stop_flag){
        std::lock_guard<std::mutex> lock(stop_flag_mtx_);
        stop_flag_=stop_flag;
    }

    void push_back(T data){
        while(true){
            if(get_stop_flag()) break;
            std::unique_lock<std::mutex> lock(data_que_mtx_);
            if(data_que_.size() >= max_size_) {
#if PIPE_INFO
                std::cout << name_ << " pipe full" << std::endl;
#endif
                if(discard_mode_==0){
                    data_que_.pop_front();
                }
                else if(discard_mode_==1){
                    data_que_.pop_back();    
                }
                else{
                    que_full_cv_.wait_for(lock, std::chrono::milliseconds(5));
                }
                continue;
            }
            else {
                data_que_.emplace_back(data);
                que_empty_cv_.notify_one();
                break;
            }
        }
    }

    int pop_front(T& ret){
        while(true){
            std::unique_lock<std::mutex> lock(data_que_mtx_);
            if(data_que_.size() == 0) {
                if (get_stop_flag())
                    return -1;
                que_empty_cv_.wait_for(lock, std::chrono::milliseconds(5));
                continue;
            }
            else {
                ret = data_que_.front();
                data_que_.pop_front();
                que_full_cv_.notify_one();
                return 0;
            }
        }
    }

private:
    std::string name_;
    std::deque<T> data_que_;
    std::mutex data_que_mtx_;
    std::condition_variable que_full_cv_, que_empty_cv_;

    int max_size_;
    //discard_mode: 0 drop front, 1 drop back, 2 block
    int discard_mode_;
    bool stop_flag_;
    std::mutex stop_flag_mtx_;
};


#endif