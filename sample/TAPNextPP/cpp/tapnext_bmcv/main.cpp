//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <dirent.h>
#include <algorithm>
#include <cmath>
#include "json.hpp"
#include "tapnext.h"

using json = nlohmann::json;
using std::cout;
using std::endl;
using std::string;
using std::vector;

/// Parse query points "y1,x1;y2,x2" (image pixels, t=0) into [Q, 3] float32.
/// Scaled to model pixels using the source video dimensions.
static vector<float> parse_query(const string& s, int src_h, int src_w) {
  vector<float> pts;
  std::stringstream ss(s);
  string token;
  while (std::getline(ss, token, ';')) {
    float y, x;
    char comma;
    std::stringstream ts(token);
    ts >> y >> comma >> x;
    // [t, y, x] in model pixels
    pts.push_back(0.0f);
    pts.push_back(y * TAPNEXT_MODEL_SIZE / src_h);
    pts.push_back(x * TAPNEXT_MODEL_SIZE / src_w);
  }
  return pts;  // [Q, 3] row-major
}

/// Load query points from a JSON file [[t,y,x],...] in model pixels.
static vector<float> parse_query_file(const string& path) {
  std::ifstream f(path);
  json j;
  f >> j;
  vector<float> pts;
  for (auto& row : j) {
    pts.push_back(row[0].get<float>());
    pts.push_back(row[1].get<float>());
    pts.push_back(row[2].get<float>());
  }
  return pts;  // [Q, 3] row-major
}

int main(int argc, char* argv[]) {
  cout.setf(std::ios::fixed);
  cout.precision(2);

  // --- parse args ---
  const char* keys =
      "{input | | input video file or image directory}"
      "{init_bmodel | ../models/BM1688/tapnext_init_fp16_1b.bmodel | init graph bmodel}"
      "{step_bmodel | ../models/BM1688/tapnext_step_fp16_1b.bmodel | step graph bmodel}"
      "{dev_id | 0 | TPU device id}"
      "{query | 128,128 | query points \"y1,x1;y2,x2\" in image pixels (t=0)}"
      "{query_file | | JSON file [[t,y,x],...] in model pixels}"
      "{max_frames | 0 | max frames to process (0 = all)}"
      "{output_dir | ./results | output directory}"
      "{help | 0 | print help}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string input = parser.get<string>("input");
  string init_bmodel = parser.get<string>("init_bmodel");
  string step_bmodel = parser.get<string>("step_bmodel");
  int dev_id = parser.get<int>("dev_id");
  string query_str = parser.get<string>("query");
  string query_file = parser.get<string>("query_file");
  int max_frames = parser.get<int>("max_frames");
  string output_dir = parser.get<string>("output_dir");

  if (input.empty()) {
    cout << "ERROR: --input is required" << endl;
    return 1;
  }

  struct stat info;
  if (stat(init_bmodel.c_str(), &info) != 0) {
    cout << "ERROR: cannot find init bmodel: " << init_bmodel << endl;
    return 1;
  }
  if (stat(step_bmodel.c_str(), &info) != 0) {
    cout << "ERROR: cannot find step bmodel: " << step_bmodel << endl;
    return 1;
  }

  // --- decode frames ---
  // Decoded with OpenCV (sophon-opencv) in software mode: the VPU hard-decode
  // path rejects some codec profiles, while software decode accepts any
  // FFmpeg-supported video.  Decode cost (~a few ms/frame) is negligible
  // next to the per-frame step-graph inference.
  // The converted bm_images need a bm_handle on the same device; the tracker
  // requests its internal handle separately.
  bm_handle_t bh;
  bm_dev_request(&bh, dev_id);

  // profiling
  TimeStamp ts;

  vector<bm_image> frames;
  int src_h = 0, src_w = 0;

  if (stat(input.c_str(), &info) == 0 && (info.st_mode & S_IFREG)) {
    // video file
    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
      cout << "ERROR: cannot open video: " << input << endl;
      return 1;
    }
    src_w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    src_h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    while (true) {
      ts.save("decode time");
      cv::Mat mat;
      cap >> mat;
      if (mat.empty()) {
        ts.save("decode time");
        break;
      }
      // upload host BGR -> device bm_image (bmimg owns the device mem).
      // Explicit create + copy (not toBMI) so the bm_image is the sole owner
      // of its device mem and the cv::Mat can be released immediately.
      bm_image bmimg;
      bm_image_create(bh, mat.rows, mat.cols, FORMAT_BGR_PACKED,
                      DATA_TYPE_EXT_1N_BYTE, &bmimg);
      void* plane = mat.data;  // BGR_PACKED is a single interleaved plane
      bm_image_copy_host_to_device(bmimg, &plane);
      frames.push_back(bmimg);
      ts.save("decode time");
      if (max_frames && (int)frames.size() >= max_frames) break;
    }
    cap.release();
  } else if (stat(input.c_str(), &info) == 0 && (info.st_mode & S_IFDIR)) {
    // image directory
    vector<string> files;
    DIR* pDir = opendir(input.c_str());
    if (!pDir) {
      cout << "ERROR: cannot open directory: " << input << endl;
      return 1;
    }
    struct dirent* ptr;
    while ((ptr = readdir(pDir)) != 0) {
      if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        files.push_back(input + "/" + ptr->d_name);
    }
    closedir(pDir);
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
      ts.save("decode time");
      cv::Mat mat = cv::imread(f, cv::IMREAD_COLOR);
      if (!mat.empty()) {
        bm_image bmimg;
        bm_image_from_mat(bh, mat, bmimg);
        frames.push_back(bmimg);
      }
      ts.save("decode time");
      if (max_frames && (int)frames.size() >= max_frames) break;
    }
    if (!frames.empty()) {
      src_h = frames[0].height;
      src_w = frames[0].width;
    }
  } else {
    cout << "ERROR: invalid input path: " << input << endl;
    return 1;
  }

  int n = (int)frames.size();
  cout << "[main] loaded " << n << " frames (" << src_w << "x" << src_h << ")"
       << endl;
  if (n == 0) {
    cout << "ERROR: no frames loaded" << endl;
    return 1;
  }

  // --- parse query points ---
  vector<float> qp;
  if (!query_file.empty())
    qp = parse_query_file(query_file);
  else
    qp = parse_query(query_str, src_h, src_w);
  int num_queries = (int)qp.size() / 3;
  cout << "[main] " << num_queries << " query points (model pixels)" << endl;
  for (int q = 0; q < num_queries; ++q)
    cout << "  q" << q << ": (t=" << qp[q * 3] << ", y=" << qp[q * 3 + 1]
         << ", x=" << qp[q * 3 + 2] << ")" << endl;

  // --- run tracking ---
  TAPNext tracker(init_bmodel, step_bmodel, dev_id);
  tracker.m_ts = &ts;

  auto t_start = std::chrono::steady_clock::now();
  vector<float> tracks, vis;
  int ret = tracker.track(frames, qp.data(), num_queries, tracks, vis);
  double total =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
          .count();
  if (ret != 0) {
    cout << "ERROR: tracking failed" << endl;
    return 1;
  }
  cout << "\n[main] total: " << total / n * 1000 << " ms/frame  (" << total
       << " s), throughput: " << n / total << " FPS" << endl;

  // --- save results ---
  if (access(output_dir.c_str(), 0) != F_OK)
    mkdir(output_dir.c_str(), S_IRWXU);

  // JSON: list of per-frame [{y, x, visible}, ...]
  json results = json::array();
  for (int t = 0; t < n; ++t) {
    json frame_res = json::array();
    for (int q = 0; q < num_queries; ++q) {
      float y = tracks[(size_t)t * num_queries * 2 + q * 2 + 0];
      float x = tracks[(size_t)t * num_queries * 2 + q * 2 + 1];
      float v = vis[(size_t)t * num_queries + q];
      bool visible = (1.0f / (1.0f + std::exp(-v)) > 0.5f);
      frame_res.push_back({{"y", y}, {"x", x}, {"visible", visible}});
    }
    results.push_back(frame_res);
  }
  string json_path = output_dir + "/tracks.json";
  std::ofstream(json_path) << std::setw(2) << results;
  cout << "[main] results saved to " << json_path << endl;

  // Print first-frame track for quick sanity check
  if (num_queries > 0) {
    cout << "[main] frame 0 track: y=" << tracks[0] << " x=" << tracks[1]
         << " vis=" << vis[0] << endl;
  }

  // --- cleanup ---
  for (auto& img : frames) bm_image_destroy(img);
  bm_dev_free(bh);

  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("tapnext++ test");
  ts.show_summary("tapnext++ test");
  ts.clear();

  cout << "all done." << endl;
  return 0;
}
