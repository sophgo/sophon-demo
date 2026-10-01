//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// WeChatQRCode QR detection + recognition demo (pure bmrt + bmcv, no sail).
//
// Usage:
//   wechat_qrcode_bmcv [detect.bmodel] [sr.bmodel] [input_path] [dev_id] [iters]
//
// iters <= 0: single-frame mode -- decode each image (a file, or every image in
//             a directory), print text + 4 corners per QR, and dump a JSON result;
// iters >  0: bench mode -- decode one image iters times (3 warmups), print
//             end-to-end and detect/sr/zxing per-stage timings.
//
// Defaults:
//   detect.bmodel = ../models/BM1684X/detect_f32_fused.bmodel
//   sr.bmodel     = ../models/BM1684X/sr_f16_fused.bmodel
//   input_path    = ../images/qr_small.png
//   dev_id        = 0
//   iters         = 0
//
// Exit code: single-file mode returns 0 iff at least one QR was decoded;
// directory / bench modes return 0 on success.

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

#include "json.hpp"
#include "wechat_qrcode.hpp"

namespace {

using json = nlohmann::json;

bool isImageFile(const std::string &path) {
  const auto ext = [&]() -> std::string {
    const size_t p = path.find_last_of('.');
    return (p == std::string::npos) ? "" : path.substr(p);
  }();
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

} // namespace

int main(int argc, char *argv[]) {
  std::string detect_bmodel = "../models/BM1684X/detect_f32_fused.bmodel";
  std::string sr_bmodel = "../models/BM1684X/sr_f16_fused.bmodel";
  std::string input_path = "../images/qr_small.png";
  int dev_id = 0;
  int iters = 0;

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

  bm_handle_t handle = nullptr;
  if (bm_dev_request(&handle, dev_id) != 0) {
    std::cerr << "bm_dev_request failed (dev_id=" << dev_id << ")"
              << std::endl;
    return 1;
  }

  std::unique_ptr<wechat_qrcode::WeChatQRCode> qr;
  try {
    qr = std::make_unique<wechat_qrcode::WeChatQRCode>(handle, detect_bmodel,
                                                       sr_bmodel);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    bm_dev_free(handle);
    return 1;
  }

  // ---- bench mode ----
  if (iters > 0) {
    // sophon-opencv device decode (VPU) + toBMI: the bm_image wraps the same
    // device buffer as the Mat, so detect runs on-device with no host->device
    // copy (the device path that closes the gap vs. Python bmcv). img must stay
    // alive while bmimg references its memory; toBMI only attaches.
    cv::Mat img = cv::imread(input_path,
                             cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION,
                             dev_id);
    if (img.empty()) {
      std::cerr << "cannot read image: " << input_path << std::endl;
      bm_dev_free(handle);
      return 1;
    }
    bm_image bmimg;
    if (cv::bmcv::toBMI(img, &bmimg, true) != BM_SUCCESS) {
      std::cerr << "toBMI failed: " << input_path << std::endl;
      bm_dev_free(handle);
      return 1;
    }

    // warm up (the first TPU inference includes kernel loading)
    for (int i = 0; i < 3; ++i)
      qr->detectAndDecode(handle, bmimg);

    qr->resetBenchStats();
    const auto t0 = std::chrono::high_resolution_clock::now();
    int n_decoded = 0;
    for (int i = 0; i < iters; ++i) {
      if (!qr->detectAndDecode(handle, bmimg).empty())
        n_decoded++;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const wechat_qrcode::BenchStats s = qr->getBenchStats();
    std::cout << "=== bench: " << input_path << " (" << bmimg.width << "x"
              << bmimg.height << ", iters=" << iters << ", decoded=" << n_decoded
              << ") ===" << std::endl;
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
    bm_image_destroy(bmimg);
    qr.reset(); // release bmrt/bm_image before the device handle goes away
    bm_dev_free(handle);
    return 0;
  }

  // ---- single-frame mode (file or directory) ----
  std::vector<std::string> files;
  if (isDirectory(input_path))
    files = listImages(input_path);
  else
    files.push_back(input_path);

  json result = json::object();
  int n_total = 0;
  for (const auto &file : files) {
    // Image loading for directory / single-frame mode (the accuracy path).
    // Note: sophon-opencv's cv::imread hardware-decodes via the JPU by default
    // (its third arg `id` defaults to 0 and is ignored on SoC, and soft decode
    // is only the opt-in IMREAD_RETRY_SOFTDEC fallback), so this reads the same
    // way as the bench loop below -- the difference is what happens AFTER
    // decode. Here we keep the cv::Mat and feed the host algorithm path
    // (detectAndDecode(Mat)): the pixels are downloaded to the host once,
    // detect does cv::resize(INTER_CUBIC) + grayscale on the CPU, and only the
    // resized gray tile goes to the TPU. The bench loop instead toBMI-attaches
    // the same device buffer zero-copy and resizes on-device via VPP (LINEAR).
    // IGNORE_ORIENTATION pins the read to the stored (raw EXIF) frame so the
    // four detected corners land in the same coordinate frame as the BoofCV
    // GT, independent of the SDK OpenCV version (4.1 auto-applies EXIF, 4.8
    // does not).
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

  qr.reset(); // release bmrt/bm_image before the device handle goes away
  bm_dev_free(handle);
  // single-file mode: 0 iff at least one QR decoded; directory: always 0
  if (files.size() == 1)
    return n_total > 0 ? 0 : 1;
  return 0;
}