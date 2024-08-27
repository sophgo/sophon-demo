//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef WSS_BOOST_H_
#define WSS_BOOST_H_

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <chrono>
#include <codecvt>
#include <condition_variable>
#include <locale>
#include <mutex>
#include "json.hpp"
#include <queue>
#include <string>
#include <vector>
#include <fstream>
#include <thread>

#include <iostream>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;


class FlexibleBarrier {
public:
    using Callback = std::function<void()>; // 定义回调类型

    FlexibleBarrier(int count, Callback callback = []{}) 
        : thread_count(count), count_to_wait(count), on_completion(callback) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        --count_to_wait;

        if (count_to_wait == 0) {
            // 执行回调函数
            on_completion();

            count_to_wait = thread_count; // 重置等待计数
            lock.unlock();
            cv.notify_all(); // 唤醒所有等待线程
        } else {
            cv.wait(lock);
        }
    }

    void add_thread() {
        std::lock_guard<std::mutex> lock(mtx);
        ++thread_count;
        ++count_to_wait;
    }
    void del_thread() {
        std::lock_guard<std::mutex> lock(mtx);
        --thread_count;
        --count_to_wait;
    }
    // 允许在运行时更改回调函数
    void set_on_completion(Callback callback) {
        std::lock_guard<std::mutex> lock(mtx);
        on_completion = callback;
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    int thread_count; // 总线程数
    int count_to_wait; // 当前需要等待的线程数
    Callback on_completion; // 完成时的回调函数
};

class WebSocketServer {
 public:
  WebSocketServer(unsigned short port, int fps)
      : ioc_(),
        acceptor_(ioc_, tcp::endpoint(tcp::v4(), port)),
        fps_(fps),
        strand_(net::make_strand(ioc_)),
        timer_(ioc_) {barrier_= std::make_shared<FlexibleBarrier>(0,[this](){
          std::unique_lock<std::mutex> lock(mutex_);
        message_queue_.pop();});}

  void run();

  bool is_open();

  void reconnect();

  void pushImgDataQueue(const std::string& data);

 private:
  void do_accept();

  void close_sessions();

  void send_frame();

  const int MAX_WSS_QUEUE_LENGTH = 5;

  class Session : public std::enable_shared_from_this<Session> {
   public: Session(tcp::socket socket, std::queue<std::string>& message_queue,
            std::mutex& mutex, std::condition_variable& cv,std::shared_ptr<FlexibleBarrier> &barrier)
        : ws_(std::move(socket)),
        message_queue_(message_queue),
          mutex_(mutex),
          cv_(cv),barrier_(barrier){
      ws_.read_message_max(64 * 1024 * 1024);  // 64 MB
      //   ws_.write_buffer_size(64 * 1024 * 1024);  // 64 MB
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session(){futureObj.wait();
      if (writeThread_.joinable()) {
    writeThread_.detach();
  } 
      // barrier_->del_thread();
    }
    
    static bool shouldExit_;

    
    void run() {
      // ws_.async_accept(
      //     [self = shared_from_this()](boost::system::error_code ec) {
      //       if (!ec) {
      //         self->do_write();
      //       }
      //     });
      boost::system::error_code ec;
      ws_.accept(ec);
      if (!ec) {
         std::promise<void> promiseObj;
        futureObj = promiseObj.get_future();
        writeThread_ = std::thread(&Session::do_write, shared_from_this(), std::move(promiseObj));
      }
    }

    bool is_open() {
      return ws_.is_open();
    }

    void close() {
        boost::system::error_code ec;
        ws_.close(websocket::close_code::normal, ec);
    }

    void do_write(std::promise<void> promiseObj) {
      barrier_->add_thread();
      while(!shouldStop) {
        std::string message;
        {std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !message_queue_.empty(); });
        
        message = message_queue_.front();
        }
        
        
        barrier_->arrive_and_wait();
        
        std::vector<uint8_t> binary_message(message.begin(), message.end());
        // ws_.text(true);
        ws_.binary(true);
        // if (!is_valid_utf8(message) || !is_valid_json(message)) {
        //   IVS_ERROR("Invalid webSocket message, process exit... {}", message);
        //   exit(1);
        // }
        boost::system::error_code ec;
        // Synchronously write the message to the WebSocket
        ws_.write(net::buffer(binary_message), ec);

        // Check if the operation succeeded
        if (!ec) {
          // If the write succeeded, call do_write() again to write the next message
          // std::thread ttt(do_write()) ;
        } else {
          Session::shouldExit_ = true;
           barrier_->del_thread();
          close();break ;
        }
        
    }
   promiseObj.set_value();
  }
   
   private:
    /**
     * @brief 从message_queue_中取出消息，异步发送。发送完成后递归调用自身
     *
     */
    std::thread writeThread_;
    websocket::stream<tcp::socket> ws_;       // websocket会话的流对象
    beast::flat_buffer buffer_;               // 从客户端接收到的数据
    std::queue<std::string>& message_queue_;  // 要发送的消息队列
    std::mutex& mutex_;
  std::atomic<bool> shouldStop;
    std::condition_variable& cv_;    std::future<void> futureObj;
  std::shared_ptr<FlexibleBarrier> barrier_;//std::thread writeThread_;

  };

  net::io_context ioc_;  // 管理IO上下文
  tcp::acceptor acceptor_;  // 侦听传入的连接请求，创建新的tcp::socket
  int fps_;
  net::strand<net::io_context::executor_type>
      strand_;               // 提供串行处理机制，避免竞态
  net::steady_timer timer_;  // 定时操作
  std::queue<std::string> message_queue_;
  std::mutex mutex_;    
  std::mutex ws_mutex;
  std::condition_variable cv_;
  std::vector<std::shared_ptr<Session>> sessions_;
    std::shared_ptr<FlexibleBarrier> barrier_;
};


#endif