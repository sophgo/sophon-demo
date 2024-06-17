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

#include "ff_decode.hpp"
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "yolov9.hpp"
using json = nlohmann::json;
using namespace std;
#define USE_OPENCV_DRAW_BOX 1

// #define DEBUG 1
char* rleToString(vector<int> R) {
    /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
    unsigned long p = 0;
    long x;
    int more;
    unsigned long m = R.size();
    char* s = (char*)malloc(sizeof(char) * m * 6);
    for (int i = 0; i < m; i++) {
        x = (long)R[i];
        if (i > 2)
            x -= (long)R[i - 2];
        more = 1;
        while (more) {
            char c = x & 0x1f;
            x >>= 5;
            more = (c & 0x10) ? x != -1 : x != 0;
            if (more)
                c |= 0x20;
            c += 48;
            s[p++] = c;
        }
    }
    s[p] = 0;
    return s;
}
// 对一维数组进行RLE编码
json runLengthEncode(const cv::Mat ucharMat, int x1, int y1, int bboxw, int bboxh, int source_w, int source_h) {
    json results;
    int sum = x1 * (source_h) + y1;
    uchar currentPixel = false;
    vector<int> result;
    for (int row = 0; row < ucharMat.cols; ++row) {
        std::vector<unsigned char> contourPoints;

        for (int col = 0; col < ucharMat.rows; ++col) {
            unsigned char pixelValue = ucharMat.at<unsigned char>(col, row);
            if (pixelValue == currentPixel)
                sum++;
            else {
                result.push_back(sum);
                currentPixel = pixelValue;
                sum = 1;
            }
        }
        if (row != ucharMat.cols - 1) {
            if (currentPixel == false)
                sum += source_h - bboxh;
            else {
                result.push_back(sum);
                currentPixel = false;
                sum = source_h - bboxh;
            }
        } else {
            if (currentPixel == false)
                sum += source_h - bboxh - y1;
            else {
                result.push_back(sum);
                currentPixel = false;
                sum = source_h - bboxh - y1;
            }
        }
    }
    sum += (source_w - bboxw - x1) * source_h;

    result.push_back(sum);
    int tot = 0;
    for (int j = 0; j < result.size(); j++)
        tot += result[j];
    results["size"] = {source_h, source_w};
    results["counts"] = rleToString(result);
    return results;
}

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684X/yolov9c_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{conf_thresh | 0.25 | confidence threshold for filter boxes}"
        "{nms_thresh | 0.7 | iou threshold for nms}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/test | input path, images direction or video file path}"
        "{classnames | ../../datasets/coco.names | class names file path}";
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
    string coco_names = parser.get<string>("classnames");
    if (stat(coco_names.c_str(), &info) != 0) {
        cout << "Cannot find classnames file." << endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0) {
        cout << "Cannot find input path." << endl;
        exit(1);
    }

    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
    cout << "set device id: " << dev_id << endl;
    bm_handle_t h = handle->handle();

    // load bmodel
    shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

    // initialize net
    YoloV9 yolov9(bm_ctx);
    CV_Assert(0 == yolov9.Init(parser.get<float>("conf_thresh"), parser.get<float>("nms_thresh"), coco_names));

    // profiling
    TimeStamp yolov9_ts;
    TimeStamp* ts = &yolov9_ts;
    yolov9.enableProfile(&yolov9_ts);

    // get batch_size
    int batch_size = yolov9.batch_size();

    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    // test images
    if (info.st_mode & S_IFDIR) {
        // get files
        vector<string> files_vector;
        DIR* pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        while ((ptr = readdir(pDir)) != 0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);
        std::sort(files_vector.begin(), files_vector.end());

        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        vector<YoloV9BoxVec> boxes;
        vector<json> results_json;
        int cn = files_vector.size();
        int id = 0;
        for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            string img_file = *iter;
            id++;
            cout << id << "/" << cn << ", img_file: " << img_file << endl;
            ts->save("decode time");
            bm_image bmimg;
            picDec(h, img_file.c_str(), bmimg);
            ts->save("decode time");
            size_t index = img_file.rfind("/");
            string img_name = img_file.substr(index + 1);
            batch_imgs.push_back(bmimg);
            batch_names.push_back(img_name);

            iter++;
            bool end_flag = (iter == files_vector.end());
            iter--;
            if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
                // predict
                CV_Assert(0 == yolov9.Detect(batch_imgs, boxes));
                for (int i = 0; i < batch_imgs.size(); i++) {
                    vector<json> bboxes_json;
                    vector<json> segs_json;
                    for (auto bbox : boxes[i]) {
#if DEBUG
                        cout << "  class id=" << bbox.class_id << ", score = " << bbox.score << " (x=" << bbox.x
                             << ",y=" << bbox.y << ",w=" << bbox.width << ",h=" << bbox.height << ")" << endl;
#endif
                        float bboxwidth = bbox.x2 - bbox.x1;
                        float bboxheight = bbox.y2 - bbox.y1;
                        json bbox_json;
                        bbox_json["category_id"] = bbox.class_id;
                        bbox_json["score"] = bbox.score;
                        bbox_json["bbox"] = {bbox.x1, bbox.y1, bboxwidth, bboxheight};
                        bboxes_json.push_back(bbox_json);
                        segs_json.push_back(runLengthEncode(bbox.mask_img, bbox.x1, bbox.y1, bboxwidth, bboxheight,
                                                            batch_imgs[i].width, batch_imgs[i].height));
                    }

                    json res_json;
                    res_json["image_name"] = batch_names[i];
                    res_json["bboxes"] = bboxes_json;
                    res_json["segs"] = segs_json;
                    results_json.push_back(res_json);
#if USE_OPENCV_DRAW_BOX
                    cv::Mat img;
                    cv::bmcv::toMAT(&batch_imgs[i], img, 1, 1, NULL, -1, true, true);
                    yolov9.draw_result(img, boxes[i]);
                    string img_file = "results/images/res_bmcv_" + batch_names[i];
                    cv::imwrite(img_file.c_str(), img);
#else
                    yolov9.draw_bmcv(h, batch_imgs[i], boxes[i], 1);

                    void* jpeg_data = NULL;
                    size_t out_size = 0;
                    int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
                    if (ret == BM_SUCCESS) {
                        string img_file = "results/images/res_bmcv_" + batch_names[i];
                        FILE* fp = fopen(img_file.c_str(), "wb");
                        fwrite(jpeg_data, out_size, 1, fp);
                        fclose(fp);
                    }
                    free(jpeg_data);
#endif
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_imgs.clear();
                batch_names.clear();
                boxes.clear();
            }
        }
        // save results
        size_t index = input.rfind("/");
        if (index == input.length() - 1) {
            input = input.substr(0, input.length() - 1);
            index = input.rfind("/");
        }
        string dataset_name = input.substr(index + 1);
        index = bmodel_file.rfind("/");
        string model_name = bmodel_file.substr(index + 1);
        string json_file = "results/" + model_name + "_" + dataset_name + "_bmcv_cpp" + "_result.json";
        cout << "================" << endl;
        cout << "result saved in " << json_file << endl;
        ofstream(json_file) << std::setw(4) << results_json;
    }
    // test video
    else {
        VideoDecFFM decoder;
        decoder.openDec(&h, input.c_str());
        int id = 0;
        vector<bm_image> batch_imgs;
        vector<YoloV9BoxVec> boxes;
        bool end_flag = false;
        while (!end_flag) {
            bm_image* img = decoder.grab();
            if (!img) {
                end_flag = true;
            } else {
                batch_imgs.push_back(*img);
            }
            if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
                // predict
                cv::Mat img;
                CV_Assert(0 == yolov9.Detect(batch_imgs, boxes));
                for (int i = 0; i < batch_size; i++) {
                    id++;
                    cout << id << ", det_nums: " << boxes[i].size() << endl;
#if USE_OPENCV_DRAW_BOX
                    cv::Mat img;
                    cv::bmcv::toMAT(&batch_imgs[i], img, 1, 1, NULL, -1, true, true);
                    yolov9.draw_result(img, boxes[i]);
                    string img_file = "results/images/res_bmcv_" + to_string(id) + ".jpg";
                    imwrite(img_file.c_str(), img);
#else
                    yolov9.draw_bmcv(h, batch_imgs[i], boxes[i], 1);

                    void* jpeg_data = NULL;
                    size_t out_size = 0;
                    int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
                    if (ret == BM_SUCCESS) {
                        string img_file = "results/images/res_bmcv_" + to_string(id) + ".jpg";
                        FILE* fp = fopen(img_file.c_str(), "wb");
                        fwrite(jpeg_data, out_size, 1, fp);
                        fclose(fp);
                    }
                    free(jpeg_data);
#endif
                    bm_image_destroy(batch_imgs[i]);
                }
                batch_imgs.clear();
                boxes.clear();
            }
        }
    }
    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    yolov9_ts.calbr_basetime(base_time);
    yolov9_ts.build_timeline("yolov9 test");
    yolov9_ts.show_summary("yolov9 test");
    yolov9_ts.clear();
    return 0;
}
