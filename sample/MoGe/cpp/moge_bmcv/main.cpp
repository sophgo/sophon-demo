//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "json.hpp"
#include "moge.hpp"

using json = nlohmann::json;
using namespace std;

// ---------------------------------------------------------------------------
// small helpers: NumPy .npy writer and depth/normal colorization, mirroring the
// Python demo (sample/MoGe/python/{postprocess_numpy.py,moge_bmcv.py}) exactly.
// ---------------------------------------------------------------------------

// Write a C-order float32 array as a NumPy ``.npy`` (version 1.0) file.
static void save_npy(const std::string& path, const float* data, const std::vector<size_t>& shape) {
    // dict text, matching numpy.lib.format._write_array_header
    std::ostringstream s;
    s << "{'descr': '<f4', 'fortran_order': False, 'shape': ";
    if (shape.empty()) {
        s << "()";
    } else if (shape.size() == 1) {
        s << "(" << shape[0] << ",)";
    } else {
        s << "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s << ", ";
            s << shape[i];
        }
        s << ")";
    }
    s << ", }";
    std::string header = s.str();

    // reserve growth-axis space (numpy GROWTH_AXIS_MAX_DIGITS = 21)
    if (!shape.empty()) {
        int n = (int)std::to_string(shape[0]).size();
        int pad = 21 - n;
        if (pad > 0) header.append((size_t)pad, ' ');
    }

    size_t hlen = header.size() + 1;                 // +1 for the trailing '\n'
    size_t padlen = 64 - ((8 + 2 + hlen) % 64);      // MAGIC_LEN(8) + uint16(2)
    uint16_t field = (uint16_t)(hlen + padlen);      // header length stored in file

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("cannot open " + path);
    fwrite("\x93NUMPY\x01\x00", 1, 8, fp);
    fwrite(&field, sizeof(uint16_t), 1, fp);  // little-endian host (all target boards are LE)
    fwrite(header.data(), 1, header.size(), fp);
    for (size_t i = 0; i < padlen; ++i) fputc(' ', fp);
    fputc('\n', fp);

    size_t total = 1;
    for (size_t d : shape) total *= d;
    fwrite(data, sizeof(float), total, fp);
    fclose(fp);
}

// HSV -> RGB (h/s/v in [0, 1]), matching postprocess_numpy._hsv_to_rgb.
static void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    h *= 6.0f;
    int i = (int)std::floor(h);
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
}

static float quantile_sorted(const std::vector<float>& sorted, float q) {
    float pos = q * (sorted.size() - 1);
    int lo = (int)std::floor(pos);
    int hi = (int)std::ceil(pos);
    float f = pos - lo;
    return sorted[lo] * (1.0f - f) + sorted[hi] * f;
}

// Disparity-based depth colorization -> BGR cv::Mat ready for cv::imwrite.
static cv::Mat colorize_depth(const std::vector<float>& depth, const std::vector<uint8_t>& mask, int H, int W) {
    std::vector<float> disp;
    disp.reserve((size_t)H * W);
    for (int i = 0; i < H * W; ++i)
        if (mask[i] && depth[i] > 0.0f) disp.push_back(1.0f / depth[i]);

    cv::Mat rgb(H, W, CV_8UC3);
    if (disp.size() >= 2) {
        std::vector<float> sorted = disp;
        std::sort(sorted.begin(), sorted.end());
        float lo = quantile_sorted(sorted, 0.02f);
        float hi = quantile_sorted(sorted, 0.98f);
        if (hi <= lo) hi = lo + 1e-6f;

        for (int i = 0; i < H; ++i) {
            uchar* px = rgb.ptr<uchar>(i);
            for (int j = 0; j < W; ++j) {
                int id = i * W + j;
                uchar* c = px + j * 3;
                if (mask[id] && depth[id] > 0.0f) {
                    float d = 1.0f / depth[id];
                    float t = (d - lo) / (hi - lo);
                    t = std::max(0.0f, std::min(1.0f, t));
                    float h = (1.0f - t) * (240.0f / 360.0f);  // far blue -> near red
                    float r, g, b;
                    hsv_to_rgb(h, 1.0f, 1.0f, r, g, b);
                    c[0] = (uchar)(r * 255.0f);
                    c[1] = (uchar)(g * 255.0f);
                    c[2] = (uchar)(b * 255.0f);
                } else {
                    c[0] = c[1] = c[2] = 0;
                }
            }
        }
    } else {
        rgb.setTo(cv::Scalar(0, 0, 0));
    }
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

// Normal map colorization: n * [0.5, -0.5, -0.5] + 0.5 -> BGR.
static cv::Mat colorize_normal(const std::vector<float>& normal, const std::vector<uint8_t>& mask, int H, int W) {
    cv::Mat rgb(H, W, CV_8UC3);
    for (int i = 0; i < H; ++i) {
        uchar* px = rgb.ptr<uchar>(i);
        for (int j = 0; j < W; ++j) {
            int id = i * W + j;
            uchar* c = px + j * 3;
            if (mask[id]) {
                const float* n = normal.data() + (size_t)id * 3;
                auto f = [](float v, float m) { return std::max(0.0f, std::min(1.0f, v * m + 0.5f)); };
                c[0] = (uchar)(f(n[0], 0.5f) * 255.0f);
                c[1] = (uchar)(f(n[1], -0.5f) * 255.0f);
                c[2] = (uchar)(f(n[2], -0.5f) * 255.0f);
            } else {
                c[0] = c[1] = c[2] = 0;
            }
        }
    }
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

// collect image files in a directory (sorted, so results are reproducible)
static void list_images(const std::string& input, std::vector<std::string>& files) {
    DIR* dp = opendir(input.c_str());
    if (!dp) throw std::runtime_error("cannot open directory " + input);
    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string ext = name.substr(name.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == "jpg" || ext == "png" || ext == "jpeg" || ext == "bmp" || ext == "webp")
            files.push_back(input + "/" + name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
}

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    const char* keys =
        "{input | ../../datasets/test | input path, one image or a directory}"
        "{bmodel | ../../models/BM1688/moge-2-vitl-normal_fp16_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    string input = parser.get<string>("input");
    string bmodel_file = parser.get<string>("bmodel");
    int dev_id = parser.get<int>("dev_id");

    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        cout << "Cannot find bmodel file: " << bmodel_file << endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0) {
        cout << "Cannot find input path: " << input << endl;
        exit(1);
    }

    // initialize net
    MoGe moge(bmodel_file, dev_id);
    TimeStamp moge_ts;
    moge.m_ts = &moge_ts;

    // create result dirs
    if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK) mkdir("results/images", S_IRWXU);

    vector<string> files;
    if (info.st_mode & S_IFDIR) {
        list_images(input, files);
    } else {
        files.push_back(input);
    }

    int H = moge.net_h, W = moge.net_w;
    vector<json> results_json;
    int cn = files.size(), id = 0;
    for (const string& img_file : files) {
        id++;
        cout << id << "/" << cn << ", img_file: " << img_file << endl;

        // decode (sophon-opencv hardware JPEG decode) + wrap as bm_image
        moge_ts.save("moge decode");
        bm_image bmimg;
        cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
        if (mat.empty()) {
            cout << "Decode error! Skipping current img." << endl;
            moge_ts.save("moge decode");
            continue;
        }
        cv::bmcv::toBMI(mat, &bmimg);
        moge_ts.save("moge decode");

        // full pipeline
        MoGeResult res;
        int ret = moge.Process(bmimg, res);
        assert(0 == ret);
        bm_image_destroy(bmimg);

        size_t idx = img_file.rfind("/");
        string name = img_file.substr(idx + 1);
        string fname = name.substr(0, name.find_last_of('.'));

        // raw float32 outputs (same layout as the Python demo's .npy files)
        save_npy("results/" + fname + "_depth.npy", res.depth.data(), {(size_t)H, (size_t)W});
        save_npy("results/" + fname + "_points.npy", res.points.data(), {(size_t)H, (size_t)W, 3});
        {
            // normal saved as (3, H, W) to match the Python demo
            std::vector<float> normal_chw(res.normal.size());
            int k = 0;
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < H * W; ++i)
                    normal_chw[k++] = res.normal[(size_t)i * 3 + c];
            save_npy("results/" + fname + "_normal.npy", normal_chw.data(), {3, (size_t)H, (size_t)W});
        }

        // colorized previews
        cv::imwrite("results/images/" + fname + "_depth.png", colorize_depth(res.depth, res.mask, H, W));
        cv::imwrite("results/images/" + fname + "_normal.png", colorize_normal(res.normal, res.mask, H, W));

        json res_json;
        res_json["image_name"] = name;
        res_json["intrinsics"] = {res.fx, 0.0f, 0.5f, 0.0f, res.fy, 0.5f, 0.0f, 0.0f, 1.0f};
        res_json["fov_x_deg"] = res.fov_x_deg;
        res_json["fov_y_deg"] = res.fov_y_deg;
        res_json["focal"] = res.focal;
        res_json["shift"] = res.shift;
        res_json["depth_min"] = res.depth_min;
        res_json["depth_max"] = res.depth_max;
        results_json.push_back(res_json);

        cout << "  fov_x=" << res.fov_x_deg << " deg, fov_y=" << res.fov_y_deg
             << " deg, depth=[" << res.depth_min << ", " << res.depth_max << "] m" << endl;
    }

    // save results
    size_t idx = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(idx + 1);
    string json_file = "results/" + model_name + "_cpp_result.json";
    ofstream(json_file) << std::setw(4) << results_json;
    cout << "================" << endl;
    cout << "result saved in " << json_file << endl;

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    moge_ts.calbr_basetime(base_time);
    moge_ts.build_timeline("moge test");
    moge_ts.show_summary("moge test");
    moge_ts.clear();
    return 0;
}