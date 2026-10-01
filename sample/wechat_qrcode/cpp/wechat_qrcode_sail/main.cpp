//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// WeChatQRCode QR detection + recognition demo via the sail delivery API
// (sail::wechat_qrcode::WeChatQRCode). No wechat_qrcode implementation is
// compiled here; link the full libsail.so (carries the wechat_qrcode symbols).
//
// Usage:
//   wechat_qrcode_sail [detect.bmodel] [sr.bmodel] [input_path] [dev_id] [iters] [core_id]
//
// iters <= 0: single-frame mode (see wechat_qrcode_bmcv/main.cpp);
// iters >  0: bench mode.
//
// core_id: BM1688 dual-core pinned NPU core; -1 = auto (default).

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "cvwrapper.h" // sail::BMImage / Decoder / Bmcv
#include "tensor.h"    // sail::Handle
#include "json.hpp"
#include "wechat_qrcode.hpp" // sail public header

// Image source for the device pass-through (sail::wechat_qrcode exposes
// detectAndDecode(bm_handle_t, bm_image), so the demo feeds a device image,
// not a host cv::Mat):
//   USE_OPENCV_DECODE=1 -> sophon-opencv cv::imread(path, flags, dev_id) (VPU
//     decode to device) + sail::Bmcv::mat_to_bm_image (toBMI that also caches
//     the Mat so the bm_image stays valid);
//   USE_OPENCV_DECODE=0 -> sail::Decoder (hardware decode path).
// Default 0 (sail::Decoder), matching the yolov5_sail sample; pass
// -DUSE_OPENCV_DECODE=1 to fall back to the opencv device-decode branch.
#ifndef USE_OPENCV_DECODE
#define USE_OPENCV_DECODE 0
#endif

namespace {

using json = nlohmann::json;

bool isImageFile(const std::string &path) {
  const size_t p = path.find_last_of('.');
  const std::string ext = (p == std::string::npos) ? "" : path.substr(p);
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

bool isDirectory(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// List image files (sorted) under a directory tree (POSIX, C++11-safe). The
// BoofCV dataset nests images one level deep (detection/<category>/imageNNN.jpg),
// so a plain readdir listing would miss them.
void listImagesRecursive(const std::string &dir, std::vector<std::string> &out) {
  DIR *dp = opendir(dir.c_str());
  if (dp == nullptr)
    return;
  struct dirent *ent;
  while ((ent = readdir(dp)) != nullptr) {
    const std::string name(ent->d_name);
    if (name == "." || name == "..")
      continue;
    const std::string full = dir + "/" + name;
    struct stat st;
    if (stat(full.c_str(), &st) != 0)
      continue;
    if (S_ISDIR(st.st_mode))
      listImagesRecursive(full, out);
    else if (isImageFile(full))
      out.push_back(full);
  }
  closedir(dp);
}

std::vector<std::string> listImages(const std::string &dir) {
  std::vector<std::string> out;
  listImagesRecursive(dir, out);
  std::sort(out.begin(), out.end());
  return out;
}

void printPoints(const cv::Mat &point) {
  std::cout << "     corners(4x2 float32):";
  std::cout << std::fixed << std::setprecision(1);
  for (int r = 0; r < point.rows; ++r) {
    std::cout << " (" << point.at<float>(r, 0) << ","
              << point.at<float>(r, 1) << ")";
  }
  std::cout << std::endl;
}

// A QR payload is raw bytes and may legally contain sequences that are not
// valid UTF-8 (e.g. BoofCV v4 damaged/image047, image048 decode to byte 0xA3).
// nlohmann::json requires valid UTF-8 in every string, so replace each invalid
// byte with U+FFFD before assigning text into the result JSON; this guarantees
// result.dump() never throws json.exception.type_error.316.
std::string sanitizeUtf8(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  const unsigned char *p = reinterpret_cast<const unsigned char *>(s.data());
  const unsigned char *end = p + s.size();
  while (p < end) {
    const unsigned char c = *p;
    int len = 0;
    if (c < 0x80)
      len = 1;
    else if ((c & 0xE0) == 0xC0)
      len = 2;
    else if ((c & 0xF0) == 0xE0)
      len = 3;
    else if ((c & 0xF8) == 0xF0)
      len = 4;
    bool ok = len > 0 && (p + len <= end);
    if (ok) {
      for (int i = 1; i < len; ++i)
        if ((p[i] & 0xC0) != 0x80) {
          ok = false;
          break;
        }
    }
    if (ok) {
      out.append(reinterpret_cast<const char *>(p), len);
      p += len;
    } else {
      out.append("\xEF\xBF\xBD"); // UTF-8 encoding of U+FFFD
      ++p;
    }
  }
  return out;
}

// Decode one image into a device sail::BMImage for the device pass-through.
// Returns 0 on success, non-zero on failure. On the opencv branch the source
// Mat is kept alive by the BMImage (mat_to_bm_image caches it), so the returned
// bm_image may outlive the temporary Mat used to build it.
int decodeBMImage(const std::string &path, sail::Handle &handle, int dev_id,
                  sail::BMImage *out) {
  (void)handle;
#if USE_OPENCV_DECODE
  cv::Mat m = cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION,
                         dev_id);
  if (m.empty())
    return -1;
  return sail::Bmcv::mat_to_bm_image(m, *out);
#else
  sail::Decoder decoder(path, true, dev_id);
  return decoder.read(handle, *out);
#endif
}

} // namespace

int main(int argc, char *argv[]) {
  std::string detect_bmodel = "../models/BM1684X/detect_f32_fused.bmodel";
  std::string sr_bmodel = "../models/BM1684X/sr_f16_fused.bmodel";
  std::string input_path = "../images/qr_small.png";
  int dev_id = 0;
  int iters = 0;
  int core_id = -1;

  if (argc > 1)
    detect_bmodel = argv[1];
  if (argc > 2)
    sr_bmodel = argv[2];
  if (argc > 3)
    input_path = argv[3];
  if (argc > 4)
    dev_id = std::stoi(argv[4]);
  if (argc > 5)
    iters = std::stoi(argv[5]);
  if (argc > 6)
    core_id = std::stoi(argv[6]);

  std::unique_ptr<sail::wechat_qrcode::WeChatQRCode> qr;
  try {
    qr = std::make_unique<sail::wechat_qrcode::WeChatQRCode>(
        detect_bmodel, sr_bmodel, dev_id, core_id);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // ---- bench mode ----
  if (iters > 0) {
    sail::Handle handle(dev_id);
    sail::BMImage bmimg;
    if (decodeBMImage(input_path, handle, dev_id, &bmimg) != 0) {
      std::cerr << "cannot decode image: " << input_path << std::endl;
      return 1;
    }

    for (int i = 0; i < 3; ++i)
      qr->detectAndDecode(handle.data(), bmimg.data());

    qr->resetBenchStats();
    const auto t0 = std::chrono::high_resolution_clock::now();
    int n_decoded = 0;
    for (int i = 0; i < iters; ++i) {
      if (!qr->detectAndDecode(handle.data(), bmimg.data()).empty())
        n_decoded++;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const sail::wechat_qrcode::BenchStats s = qr->getBenchStats();
    std::cout << "=== bench: " << input_path << " (" << bmimg.width() << "x"
              << bmimg.height() << ", iters=" << iters << ", decoded="
              << n_decoded << ") ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "e2e    : " << total_ms / iters << " ms/img, "
              << 1000.0 * iters / total_ms << " fps" << std::endl;
    std::cout << "detect : "
              << (s.detect_calls ? s.detect_ms / s.detect_calls : 0.0)
              << " ms/img (" << s.detect_calls << " calls, TPU)" << std::endl;
    std::cout << "sr     : " << (s.sr_calls ? s.sr_ms / s.sr_calls : 0.0)
              << " ms/img (" << s.sr_calls << " calls, TPU)" << std::endl;
    std::cout << "zxing  : " << (s.zxing_calls ? s.zxing_ms / s.zxing_calls : 0.0)
              << " ms/img (" << s.zxing_calls << " calls, CPU)" << std::endl;
    return 0;
  }

  // ---- single-frame mode ----
  std::vector<std::string> files;
  if (isDirectory(input_path))
    files = listImages(input_path);
  else
    files.push_back(input_path);

  json result = json::object();
  int n_total = 0;
  for (const auto &file : files) {
    // Directory / single-frame mode (the accuracy path; see
    // wechat_qrcode_bmcv/main.cpp): sophon-opencv's cv::imread JPU-decodes by
    // default, and the returned Mat feeds the host algorithm path (CPU
    // cv::resize INTER_CUBIC + grayscale) rather than the device VPP path used
    // by the (bench) decodeBMImage branch. IGNORE_ORIENTATION pins the read to
    // the stored (raw EXIF) frame so the four detected corners land in the
    // same coordinate frame as the BoofCV GT, independent of the SDK OpenCV
    // version.
    cv::Mat img = cv::imread(file, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (img.empty()) {
      std::cerr << "cannot read image: " << file << std::endl;
      continue;
    }

    std::vector<cv::Mat> points;
    std::vector<std::string> texts = qr->detectAndDecode(img, &points);

    std::cout << "[" << file << "]" << std::endl;
    std::cout << "  decoded " << texts.size() << " code(s)" << std::endl;
    json file_result = json::array();
    for (size_t i = 0; i < texts.size(); ++i) {
      std::cout << "  [" << i << "] text=" << texts[i] << std::endl;
      if (points.size() > i && !points[i].empty())
        printPoints(points[i]);

      json item;
      item["text"] = sanitizeUtf8(texts[i]);
      json pts = json::array();
      if (points.size() > i && !points[i].empty()) {
        for (int r = 0; r < points[i].rows; ++r) {
          pts.push_back({points[i].at<float>(r, 0), points[i].at<float>(r, 1)});
        }
      }
      item["points"] = pts;
      file_result.push_back(item);
    }
    result[file] = file_result;
    n_total += static_cast<int>(texts.size());
  }

  std::cout << "== result json ==" << std::endl;
  std::cout << result.dump(2) << std::endl;

  // Persist the result JSON so tools/eval_qrcode.py can consume it directly
  // (mirrors the other demos dumping into ./results).
  struct stat rst;
  if (stat("results", &rst) != 0)
    mkdir("results", 0755);
  {
    std::ofstream ofs("results/wechat_qrcode_results.json");
    if (ofs)
      ofs << result.dump(2) << std::endl;
  }

  if (files.size() == 1)
    return n_total > 0 ? 0 : 1;
  return 0;
}