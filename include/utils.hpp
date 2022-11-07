//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include <ctime>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <iomanip>
using namespace std::chrono;
using time_stamp_t = time_point<steady_clock, microseconds>;

#define MAX_TAGS 128
#define MAX_RECORDS 200

#define LOG_TS(p_ts, tag) if ((p_ts)) (p_ts)->save((tag));

struct TsInfo {
  std::string title;
  std::string desc;
};

class TimeStamp {
public:
  //TimeStamp(): records_(), tags_(), timeline_() {}

  TimeStamp(): records_(), tags_(), timeline_() {
    tags_.reserve(MAX_TAGS);
    records_.reserve(MAX_TAGS);

    ts_ = new std::vector<time_stamp_t>[MAX_TAGS]();
    for (int i = 0; i < MAX_TAGS; i++)
      ts_[i].reserve(MAX_RECORDS);

    num_tags_ = 0;
    base_ = time_point_cast<microseconds>(steady_clock::now());
  }

  ~TimeStamp() {
    delete []ts_;
  }

  void save(const std::string &tag) {
        time_stamp_t t = time_point_cast<microseconds>(steady_clock::now());
        std::unordered_map<std::string, std::vector<time_stamp_t> *>::iterator it = records_.find(tag);
        if (it == records_.end()) {
            if (num_tags_ == MAX_TAGS)
                return;
            records_[tag] = &ts_[num_tags_++];
            tags_.push_back(tag);
        }
        records_[tag]->push_back(t);

        if (records_[tag]->size() > MAX_RECORDS) {
            records_[tag]->erase(records_[tag]->begin());
        }
  }

  void calbr_basetime(time_stamp_t basetime) {
    base_ = basetime;
  }

  void show_duration(const std::string &head) {
    std::cout << std::endl;
    std::cout << "############################" << std::endl;
    std::cout << "DURATIONS: " << head << std::endl;
    std::cout << "############################" << std::endl;
    for (size_t i = 0; i < tags_.size(); i++) {
      std::vector<time_stamp_t> ts = *records_[tags_[i]];
      if (ts.size() % 2) {
        std::cout << "[" << tags_[i] << "] invalid records #: " << ts.size() << std::endl;
        continue;
      }
      microseconds sum(0);
      for (size_t j = 0; j < ts.size(); j += 2) {
        microseconds duration = duration_cast<microseconds>(ts[j + 1] - ts[j]);
        std::cout << "[" << std::setw(20) << tags_[i] << "] - iteration <" << (j / 2) << "> : "
                  << duration.count() << " us" << std::endl;
        sum += duration;
      }
    }
  }

  void show_summary(const std::string &head) {
    std::cout << std::endl;
    std::cout << "############################" << std::endl;
    std::cout << "SUMMARY: " << head << std::endl;
    std::cout << "############################" << std::endl;
    for (size_t i = 0; i < tags_.size(); i++) {
      std::vector<time_stamp_t> ts = *records_[tags_[i]];
      if (ts.size() % 2) {
        std::cout << "[" << tags_[i] << "] invalid records #: " 
                  << ts.size() << " us"<< std::endl;
        continue;
      }
      microseconds sum(0);
      for (size_t j = 0; j < ts.size(); j += 2) {
        microseconds duration = duration_cast<microseconds>(ts[j + 1] - ts[j]);
        sum += duration;
      }
      std::cout << "[" << std::setw(20) << tags_[i] << "] "
                << " loops: "  << std::setw(4) << (ts.size() / 2) 
                << " avg: " << (sum / (ts.size() / 2)).count() 
                << " us"<< std::endl;
    }
  }

  void build_timeline(const std::string &head) {
    for (size_t i = 0; i < tags_.size(); i++) {
      std::vector<time_stamp_t> ts = *records_[tags_[i]];
      if (ts.size() % 2) {
        std::cout << "[" << tags_[i] << "] invalid records #: " 
                  << ts.size() << " us" << std::endl;
        continue;
      }
      for (size_t j = 0; j < ts.size(); j += 2) {
        struct TsInfo ti;
        ti.title = head;
        ti.desc = " >> ["+ std::to_string(j / 2) + "]" + tags_[i];
        timeline_[ts[j]].push_back(ti);
        ti.title = head;
        ti.desc = " << [" + std::to_string(j / 2) + "]" + tags_[i];
        timeline_[ts[j + 1]].push_back(ti);
      }
    }
  }

  void merge_timeline(TimeStamp *ts) {
    for (auto i = ts->timeline_.begin(); i != ts->timeline_.end(); i++)
      for (auto& j : i->second)
        timeline_[i->first].push_back(j);
  }

  void show_timeline() {
    std::cout << std::endl;
    std::cout << "############################" << std::endl;
    std::cout << "Timeline (vs basetime)" << std::endl;
    std::cout << "############################" << std::endl;

    for (auto i = timeline_.begin(); i != timeline_.end(); i++) {
      microseconds duration = duration_cast<microseconds>(i->first - base_);
      for (auto& j : i->second)
        std::cout << j.title << " | " << std::setw(12)
                  << duration.count() << " (us)" << j.desc << std::endl;
    }
  }

  void clear() {
    for (auto i = timeline_.begin(); i != timeline_.end(); i++) {
      i->second.clear();
    }
    timeline_.clear();
    for (auto i = records_.begin(); i != records_.end(); i++) {
      i->second->clear();
    }
    records_.clear();
    tags_.clear();
    for (int i = 0; i < MAX_TAGS; i++)
      ts_[i].clear();
    num_tags_ = 0;
  }

  std::unordered_map<std::string, std::vector<time_stamp_t> *> records_;
  std::vector<std::string> tags_;
  std::vector<time_stamp_t> *ts_;
  std::map<time_stamp_t, std::vector<struct TsInfo>> timeline_;
  int num_tags_;
  time_stamp_t base_;
};

#endif /*UTILS_HPP*/
