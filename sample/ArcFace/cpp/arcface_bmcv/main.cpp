//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "arcface.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684X/arcface_resnet50_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/test | input path, images directory.}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    string bmodel_file = parser.get<string>("bmodel");
    string input = parser.get<string>("input");
    int dev_id = parser.get<int>("dev_id");

    // check params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        cout << "Cannot find valid model file." << endl;
        exit(1);
    }

    if (stat(input.c_str(), &info) != 0) {
        cout << "Cannot find input path." << endl;
        exit(1);
    }

    // initialize ArcFace
    ArcFace arcface(bmodel_file, dev_id);

    // profiling
    TimeStamp arcface_ts;
    arcface.m_ts = &arcface_ts;

    // get batch_size
    int batch_size = arcface.batch_size;

    // test images
    if (info.st_mode & S_IFDIR) {
        // get files
        vector<string> files_vector;
        DIR* pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        if (pDir == nullptr) {
            std::cerr << "Error: cannot open directory: " << input << std::endl;
            return -1;
        }
        while ((ptr = readdir(pDir)) != 0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);

        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        int cn = 0;
        int total = files_vector.size();
        std::sort(files_vector.begin(), files_vector.end());
        for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            string img_file = *iter;
            cn++;
            cout << cn << "/" << total << ", img_file: " << img_file << endl;

            arcface_ts.save("decode time");
            bm_image bmimg;
            cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
            if (mat.empty()) {
                cout << "Decode error! Skipping current img." << endl;
                continue;
            }
            cv::bmcv::toBMI(mat, &bmimg);
            arcface_ts.save("decode time");

            // BUG-003: toBMI 产出的 bm_image 复用同一块设备缓冲（SE9 上同一 batch 各图全部指向最后一张）。
            // 立即拷贝到独立内存，保证 batch 内每张图互不覆盖。包装图不持有缓冲，destroy 只释放句柄。
            bm_image bmimg_indep;
            bm_image_create(arcface.get_handle(), bmimg.height, bmimg.width,
                            bmimg.image_format, bmimg.data_type, &bmimg_indep);
            bm_image_alloc_dev_mem(bmimg_indep, BMCV_IMAGE_FOR_IN);
            bmcv_copy_to_atrr_t copy_attr;
            memset(&copy_attr, 0, sizeof(copy_attr));
            copy_attr.start_x = 0;
            copy_attr.start_y = 0;
            copy_attr.if_padding = 1;
            bmcv_image_copy_to(arcface.get_handle(), copy_attr, bmimg, bmimg_indep);
            bm_image_destroy(bmimg);
            batch_imgs.push_back(bmimg_indep);

            size_t index = img_file.rfind("/");
            string img_name = img_file.substr(index + 1);
            batch_names.push_back(img_name);

            iter++;
            bool end_flag = (iter == files_vector.end());
            iter--;
            if ((batch_imgs.size() == (size_t)batch_size || end_flag) && !batch_imgs.empty()) {
                // Embed
                vector<vector<float>> embeddings;
                auto ret = arcface.Embed(batch_imgs, embeddings);
                assert(0 == ret);

                for (size_t i = 0; i < embeddings.size(); i++) {
                    // print norm and first 5 values
                    float norm = 0.0f;
                    for (size_t j = 0; j < embeddings[i].size(); j++) {
                        norm += embeddings[i][j] * embeddings[i][j];
                    }
                    norm = std::sqrt(norm);
                    cout << "  " << batch_names[i] << ": norm=" << norm;
                    cout << " first5=[";
                    for (int j = 0; j < 5; j++) {
                        cout << embeddings[i][j];
                        if (j < 4) cout << ", ";
                    }
                    cout << "]" << endl;
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_imgs.clear();
                batch_names.clear();
            }
        }
    } else {
        std::cout << "Open input failed! Only support image dir now." << std::endl;
    }

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    arcface_ts.calbr_basetime(base_time);
    arcface_ts.build_timeline("arcface test");
    arcface_ts.show_summary("arcface test");
    arcface_ts.clear();
    return 0;
}
