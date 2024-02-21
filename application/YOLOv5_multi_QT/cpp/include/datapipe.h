// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#ifndef DATAPIPE_H_
#define DATAPIPE_H_

#include<deque>
#include<mutex>
#include<atomic>
#include<condition_variable>


/******************************************
 *  DataPipe主要实现了一个线程安全的队列，可设置队列的最大长度
 *  可对队列已满时的处理策略进行配置：DISCARD_MODE 0 drop front, 1 drop back, 2 block
******************************************/

#define DISCARD_MODE 2
template <typename T>
class DataPipe{
public:
    DataPipe(int max_len,int discard_mode=DISCARD_MODE):max_size_(max_len),discard_mode_(discard_mode),stop_flag_(false){};
    ~DataPipe(){
        stop();
        que_empty_cv_.notify_all();
        que_full_cv_.notify_all();
    }

    int size(){
        std::lock_guard<std::mutex> lock(data_que_mtx_);
        return data_que_.size();
    }

    void stop(){
        stop_flag_=true;
    }

    void push_back(T data){
        while(true){
            if(stop_flag_) break;
            if(size()>=max_size_)
            {
                if(discard_mode_==0){
                    std::lock_guard<std::mutex> lock(data_que_mtx_);
                    data_que_.pop_front();
                }
                else if(discard_mode_==1){
                    std::lock_guard<std::mutex> lock(data_que_mtx_);
                    data_que_.pop_back();    
                }
                else{
                    std::unique_lock<std::mutex> lock(data_que_mtx_);
                    que_full_cv_.wait_for(lock, std::chrono::milliseconds(5));
                }
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(data_que_mtx_);
                data_que_.emplace_back(data);
                que_empty_cv_.notify_one();
                break;
            }
        }
    }

    T pop_front(){
        while(true){
            if(stop_flag_) break;
            if(size()==0)
            {
                std::unique_lock<std::mutex> lock(data_que_mtx_);
                que_empty_cv_.wait_for(lock,std::chrono::milliseconds(5));
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(data_que_mtx_);
                T ret = data_que_.front();
                data_que_.pop_front();
                que_full_cv_.notify_one();
                return ret;
            }
        }
    }

private:
    std::deque<T> data_que_;
    std::mutex data_que_mtx_;
    std::condition_variable que_full_cv_, que_empty_cv_;

    int max_size_;
    //discard_mode: 0 drop front, 1 drop back, 2 block
    int discard_mode_;
    std::atomic<bool> stop_flag_;
    std::mutex stop_flag_mtx_;
};


#endif