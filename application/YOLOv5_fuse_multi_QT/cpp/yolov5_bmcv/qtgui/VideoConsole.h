// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#ifndef VIDEO_CONSOLE_H_
#define VIDEO_CONSOLE_H_

#include <memory>
#include <deque>
#include <mutex>
#include <thread>
#include <QThread>
#include "BMLabel.h"

#define QWIDGET_WIDTH 1920
#define QWIDGET_HEIGHT 1080
template <typename T>
class VideoConsole{

public:
    VideoConsole(int rows, int cols, int thread_num=1, int max_que_size=5)
            :rows_(rows),cols_(cols),thread_num_(thread_num),max_que_size_(max_que_size),label_mtx_(rows*cols){
        qwidget_ptr = new QWidget;
        qwidget_ptr->setGeometry(0,0,QWIDGET_WIDTH,QWIDGET_HEIGHT);
        layout = new QGridLayout(qwidget_ptr);
        qwidget_ptr->setLayout(layout);
        for(int row=0;row<rows_;row++){
            for(int col=0;col<cols_;col++){
                addLabel(row,col);
            }
        }
        start();//开启线程，处理工作队列
    };

    ~VideoConsole(){
        stop();
        for(int i=0;i<thread_num_;i++){
            thread_vec_[i]->join();
        }
        delete qwidget_ptr;
    };

    void start(){
        stop_flag_=false;
        for(int i=0;i<thread_num_;i++){
            thread_vec_.emplace_back(std::make_shared<std::thread>(&VideoConsole::dowork,this));
        }
        qwidget_ptr->show();//用于显示窗口部件
    }

    void stop(){
        std::lock_guard<std::mutex> lock(stop_flag_mtx_);
        stop_flag_=true;
    }

    void push_img(int channel_idx, std::shared_ptr<T> img_ptr);//把img推到工作队列

private:
    int rows_,cols_;
    int thread_num_;
    int max_que_size_;

    std::atomic<bool> stop_flag_;
    std::mutex stop_flag_mtx_;
    
    std::deque<std::pair<int,std::shared_ptr<T>>> img_que_;
    std::mutex img_que_mtx_;

    std::vector<std::shared_ptr<BMLabel>> label_vec_;
    std::vector<std::mutex> label_mtx_;
    std::vector<std::shared_ptr<std::thread>> thread_vec_;
    
    QWidget* qwidget_ptr;
    QGridLayout* layout;

    void addLabel(int row, int col, int dev_id=0);
    void dowork();

    bool get_stop_flag(){
        std::lock_guard<std::mutex> lock(stop_flag_mtx_);
        return stop_flag_;
    }

};

template <typename T>
void VideoConsole<T>::addLabel(int row, int col, int dev_id){
    std::shared_ptr<BMLabel> label_ptr = std::make_shared<BMLabel>(dev_id,qwidget_ptr);
    layout->addWidget(label_ptr.get(),row,col);
    label_vec_.push_back(label_ptr);
}

template <typename T>
void VideoConsole<T>::push_img(int channel_idx, std::shared_ptr<T> img_ptr){
    if(channel_idx >= label_vec_.size()){
        std::cout<<"VideoConsole: channel_idx "<<channel_idx<<" doesn't exist!"<<std::endl;
        return;
    }
    std::lock_guard<std::mutex> lock(img_que_mtx_);
    if(img_que_.size()>=max_que_size_){
        img_que_.pop_front();
    }
    img_que_.emplace_back(std::make_pair(channel_idx,img_ptr));
}

template <typename T>
void VideoConsole<T>::dowork(){
    while(true){
        if(get_stop_flag()){
            break;
        }
        std::unique_lock<std::mutex> lock(img_que_mtx_);
        if(img_que_.size()>0){
            std::shared_ptr<T> img_ptr;
            int channel_idx;
            {
                channel_idx = img_que_.front().first; //第一个元素
                img_ptr = img_que_.front().second;  //第二个元素
                img_que_.pop_front();
                lock.unlock();
            }
            {
                std::lock_guard<std::mutex> lock_label(label_mtx_[channel_idx]);
                label_vec_[channel_idx]->show_img(img_ptr);//发射信号在show_img实现，注意emit是异步接口，需考虑线程安全问题
                continue;
            }
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}


#endif