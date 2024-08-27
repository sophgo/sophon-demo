//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>

#include "cvdemo.hpp"
#include "json.hpp"
using json = nlohmann::json;
using namespace std;
#define WITH_ENCODE 1
int main(int argc, char* argv[]) {
  // cout.setf(ios::fixed);
  // get params
  const char* keys =
      "{dwa_l_path | config/dwa_L.json | dwa_L path}"
      "{dwa_r_path | config/dwa_R.json | dwa_R path}"
      "{decode_path | config/camera_cv_demo.json | decode path}"
      "{blend_path | config/blend.json | blend path}"
      "{help | 0 | print help information.}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  // initialize net
  std::vector<std::string> dwa_path = {parser.get<std::string>("dwa_l_path"),
                                       parser.get<std::string>("dwa_r_path")};
  CvDemo cvdemo(9002, 25, parser.get<std::string>("decode_path"), dwa_path,
                parser.get<std::string>("blend_path"));
  while (true) {
    cvdemo.Detect();
  }

  // profiling

  return 0;
}
