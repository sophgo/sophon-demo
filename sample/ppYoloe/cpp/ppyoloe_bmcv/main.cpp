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
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "ppyoloe.hpp"
using json = nlohmann::json;
using namespace std;

int main(int argc, char* argv[]) {
    // init format
    cout.setf(ios::fixed);

    // get params
    const char* keys =
        "{bmodel | ../../models/BM1684/ppyoloe_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{conf_thresh | 0.4 | confidence threshold for filter boxes}"
        "{nms_thresh | 0.6 | iou threshold for nms}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/coco/val2017_1000 | input path, images direction or video file path}"
        "{classnames | ../../datasets/coco.names | class names file path}";
    cv::CommandLineParser parser(argc, argv, keys);

    // print help message
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    // init key params
    string bmodel_file = parser.get<string>("bmodel");
    string input       = parser.get<string>("input");
    string coco_names = parser.get<string>("classnames");
    int    dev_id      = parser.get<int>("dev_id");

    // check key params
    struct stat info;
    if (stat(bmodel_file.c_str(), &info) != 0) {
        cout << "Cannot find valid model file." << endl;
        exit(1);
    }
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
    shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle,bmodel_file.c_str());

    // initialize net
    ppYoloe ppyoloe(bm_ctx);
    CV_Assert(0 == ppyoloe.Init(parser.get<float>("conf_thresh"), parser.get<float>("nms_thresh"), coco_names));

    // profiling
    TimeStamp ppyoloe_ts;
    TimeStamp* ts = &ppyoloe_ts;
    ppyoloe.enableProfile(&ppyoloe_ts);

    // get batch_size
    int batch_size = ppyoloe.batch_size();

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

        // init inference vars
        vector<string> batch_names;
        vector<ppYoloeBoxVec> boxes;
        vector<json> results_json;
        vector<bm_image> batch_imgs;
        int cn = files_vector.size();
        int id = 0;
        for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            for (int i = 0; i < batch_size; i++) {
                if (iter == files_vector.end()){
                    iter--;
                    cout << "using last img file to complete img batch" << endl;
                }
                string img_file = *iter;
                string img_name = img_file.substr(img_file.rfind("/")+1);
                id++;
                cout << id << "/" << cn << ", img_file: " << img_file << endl;
                ts->save("decode time");
                // ffmpeg decode
                bm_image bmimg;
                picDec(h, img_file.c_str(), bmimg);
                ts->save("decode time");
#if DEBUG
                cout<<"batch img:"<<batch_imgs[i].format()<<" "<<batch_imgs[i].dtype()<<endl;
#endif
                batch_imgs.push_back(bmimg);
                batch_names.push_back(img_name); //batch_names has real batch_size

                iter++;
            }
            iter--;
            // predict
            CV_Assert(0 == ppyoloe.Detect(batch_imgs, boxes));
            for (int i = 0; i < batch_size; i++) { //use real batch size
                vector<json> bboxes_json;
                // storage convert
                if (batch_imgs[i].image_format != 0){
                    bm_image frame;
                    bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
                    bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
                    bm_image_destroy(batch_imgs[i]);
                    batch_imgs[i] = frame;
                }
                for (auto bbox : boxes[i]) {
#if DEBUG
                    cout << "  class id=" << bbox.class_id << ", score = " << bbox.score << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",h=" << bbox.height << ")" << endl;
#endif
                    // draw image
                    ppyoloe.draw_bmcv(h,bbox.class_id,bbox.score, bbox.x, bbox.y, bbox.width, bbox.height,batch_imgs[i], false);

                    // save json result
                    json bbox_json;
                    bbox_json["category_id"] = bbox.class_id;
                    bbox_json["score"] = bbox.score;
                    bbox_json["bbox"] = {bbox.x, bbox.y, bbox.width, bbox.height};
                    bboxes_json.push_back(bbox_json);
                }

                // save json results
                json res_json;
                res_json["image_name"] = batch_names[i];
                res_json["bboxes"] = bboxes_json;
                results_json.push_back(res_json);

                // save image
                void* jpeg_data = NULL;
                size_t out_size = 0;
                int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
                if(ret == BM_SUCCESS) {
                    string img_file = "results/images/" + batch_names[i];
                    FILE *fp = fopen(img_file.c_str(), "wb");
                    fwrite(jpeg_data, out_size, 1, fp);
                    fclose(fp);
                }
                free(jpeg_data);
                bm_image_destroy(batch_imgs[i]);
            }
            batch_imgs.clear();
            batch_names.clear();
            boxes.clear();
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
        ofstream(json_file) << setw(4) << results_json;
    }
    // test video
    else {
        // get video
        VideoDecFFM decoder;
        decoder.openDec(&h, input.c_str());
        int id = 0;
        vector<bm_image> batch_imgs;
        vector<ppYoloeBoxVec> boxes;
        // loop to detect
        bool exit_flag = true;
        while(exit_flag){
            bm_image *img = decoder.grab();
            // video is over
            if(!img && batch_imgs.empty())
                break;

            // video is over but remain imgs
            else if(!img && !batch_imgs.empty()){
                exit_flag = false;
            }else{
                batch_imgs.push_back(*img);
            }
            //
            if((int)batch_imgs.size() == batch_size || !exit_flag) {
                CV_Assert(0 == ppyoloe.Detect(batch_imgs, boxes));
                for(int i = 0; i < batch_imgs.size(); i++){
                id++;
                cout << id << ", det_nums: " << boxes[i].size() << endl;
                if (batch_imgs[i].image_format != 0){
                    bm_image frame;
                    bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
                    bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
                    bm_image_destroy(batch_imgs[i]);
                    batch_imgs[i] = frame;
                }
                for (auto bbox : boxes[i]) {
#if DEBUG
                    cout << "  class id=" << bbox.class_id << ", score = " << bbox.score << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",h=" << bbox.height << ")" << endl;
#endif
                    ppyoloe.draw_bmcv(h, bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.width, bbox.height, batch_imgs[i], false);
                }
                string img_file = "results/images/" + to_string(id) + ".jpg";
                void* jpeg_data = NULL;
                size_t out_size = 0;
                int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
                if (ret == BM_SUCCESS) {
                    FILE *fp = fopen(img_file.c_str(), "wb");
                    fwrite(jpeg_data, out_size, 1, fp);
                    fclose(fp);
                }
                free(jpeg_data);
                bm_image_destroy(batch_imgs[i]);
                }
                batch_imgs.clear();
                boxes.clear();
            }
        }
    }
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ppyoloe_ts.calbr_basetime(base_time);
  ppyoloe_ts.build_timeline("ppyoloe test");
  ppyoloe_ts.show_summary("ppyoloe test");
  ppyoloe_ts.clear();

  return 0;
}
