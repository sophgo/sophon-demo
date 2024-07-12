//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "wss_boost.h"


bool WebSocketServer::Session::shouldExit_ = false;

void WebSocketServer::run() {
    do_accept();
    ioc_.run();
}

void WebSocketServer::pushImgDataQueue(const std::string& data) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (message_queue_.size() < MAX_WSS_QUEUE_LENGTH) {
 
          message_queue_.push(data);
      
    cv_.notify_all();
  }
}

// void WebSocketServer::do_accept() {
//   acceptor_.async_accept(
//     [this](boost::system::error_code ec, tcp::socket socket) {
//         if (!ec) {
//             auto newSession = std::make_shared<Session>(std::move(socket), mutex_,
//                                       cv_);
//             sessions_.push_back(newSession);
//             newSession->run();
//         } 
//           do_accept();
//       });
// }

void WebSocketServer::do_accept() {
    while(1) {        
      tcp::socket socket(ioc_);   
    
      acceptor_.accept(socket);

      auto session_ = std::make_shared<Session>(std::move(socket),message_queue_ ,mutex_,
                                          cv_,barrier_);
      session_->run();
    }
}




