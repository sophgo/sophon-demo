//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <iomanip>
#include <fstream>
#include <dirent.h>
#include <sstream>
#include <string>
#include <numeric>
#include <chrono>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "spdlog/spdlog.h"
#include "cvwrapper.h"
#include "engine.h"
#include "opencv2/opencv.hpp"
#include "processor.h"
#include "json.hpp"

using json = nlohmann::json;



// 全局变量
// 记录模型信息
std::vector<std::string> gh_names;       // graph名
std::vector<std::string> input_names;    // 网络输入名
std::vector<int>         input_shape;    // 输入shape
std::vector<std::string> output_names;   // 网络输出名
std::vector<int>         output_shape;   // 输出shape
bm_data_type_t           input_dtype;    // 输入数据类型
bm_data_type_t           output_dtype;   // 输出数据类型

std::vector<std::string> class_names;   // coco labels
TimeStamp centernet_ts;
TimeStamp* ts = &centernet_ts;
 
const std::vector<std::vector<int>> colors = {
    {255, 0, 0},    {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {170, 255, 0},  {85, 255, 0},    {0, 255, 0},     {0, 255, 85},
    {0, 255, 170},  {0, 255, 255},   {0, 170, 255},   {0, 85, 255},
    {0, 0, 255},    {85, 0, 255},    {170, 0, 255},   {255, 0, 255},
    {255, 0, 170},  {255, 0, 85},    {255, 0, 0},     {255, 0, 255},
    {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255},
    {85, 255, 255}};
 
// 输出bmodel输出层信息
void PrintModelInputInfo(sail::Engine& engine) {
    // graph名
    gh_names = engine.get_graph_names();
    std::string gh_info;
    for_each(gh_names.begin(), gh_names.end(), [&](std::string& s) {
        gh_info += "0: " + s + "; ";
    });
    std::cout << "grapgh name -> " << gh_info << "\n";
 
    // 网络输入名
    input_names = engine.get_input_names(gh_names[0]);
    assert(input_names.size() > 0);
    std::string input_tensor_names;
    for_each(input_names.begin(), input_names.end(), [&](std::string& s) {
        input_tensor_names += "0: " + s + "; ";
    });
    std::cout << "net input name -> " << input_tensor_names << "\n";
 
    // 网络输出名字
    output_names = engine.get_output_names(gh_names[0]);
    assert(output_names.size() > 0);
    std::string output_tensor_names;
    for_each(output_names.begin(), output_names.end(), [&](std::string& s) {
        output_tensor_names += "0: " + s + "; ";
    });
    std::cout << "net output name -> " << output_tensor_names << "\n";
 
    // 网络输入尺寸
    input_shape = engine.get_input_shape(gh_names[0], input_names[0]);
    std::string input_tensor_shape;
    for_each(input_shape.begin(), input_shape.end(), [&](int s) {
        input_tensor_shape += std::to_string(s) + " ";
    });
    std::cout << "input tensor shape -> " << input_tensor_shape << "\n";
    
    // 网络输出尺寸
    output_shape = engine.get_output_shape(gh_names[0], output_names[0]);
    std::string output_tensor_shape;
    for_each(output_shape.begin(), output_shape.end(), [&](int s) {
        output_tensor_shape += std::to_string(s) + " ";
    });
    std::cout << "output tensor shape -> " << output_tensor_shape << "\n";
 
    // 网络输入数据类型
    input_dtype = engine.get_input_dtype(gh_names[0], input_names[0]);
    std::cout << "input dtype -> "<< input_dtype << ", is fp32=" << ((input_dtype == BM_FLOAT32) ? "true" : "false") << "\n";
 
    // 网络输出数据类型
    output_dtype = engine.get_output_dtype(gh_names[0], output_names[0]);
    std::cout << "output dtype -> "<< output_dtype << ", is fp32=" << ((output_dtype == BM_FLOAT32) ? "true" : "false") << "\n";
}
 
 
 
int main(int argc, char** argv) {
    const char *keys="{ bmodel | ../../models/BM1684/centernet_fp32_1b.bmodel | bmodel file path}"
                     "{ dev_id | 0    | TPU device id}"
                     "{ conf_thresh   | 0.35 | confidence threshold for filter boxes}"
                     "{ input  | ../../datasets/test | input directory path}"
                     "{ help   | 0    | Print help information.}"
                     "{classnames | ../../datasets/coco.names | class names file path}";
 
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
 
    // 模型路径
    std::string bmodel_file = parser.get<std::string>("bmodel");
    // 图片名
    std::string input = parser.get<std::string>("input");
    // 设备号
    int dev_id              = parser.get<int>("dev_id");
    // 置信度
    float confidence        = parser.get<float>("conf_thresh");
    
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

    std::ifstream ifs(coco_names);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            line = line.substr(0, line.length() - 1);
            class_names.push_back(line);
        }
    }

    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

 
    // 生成engine，加载模型
    sail::Engine engine(dev_id);
    if (!engine.load(bmodel_file)) {
        // 加载模型失败
        std::cout << "Engine load bmodel "<< bmodel_file << "failed" << "\n";
        exit(0);
    }
 
    // 输出模型信息
    PrintModelInputInfo(engine);
 
    sail::Handle handle = engine.get_handle();
    sail::Tensor input_tensor(handle,  input_shape,  input_dtype,  false, false);
    sail::Tensor output_tensor(handle, output_shape, output_dtype, true,  true);
 
    std::map<std::string, sail::Tensor*> input_tensors  = {{ input_names[0],  &input_tensor}}; 
    std::map<std::string, sail::Tensor*> output_tensors = {{ output_names[0], &output_tensor}}; 
 
    engine.set_io_mode(gh_names[0], sail::SYSO);
    sail::Bmcv bmcv(handle);
 
    // 根据网络输入类型确定网络图片输入类型
    bm_image_data_format_ext img_dtype = bmcv.get_bm_image_data_format(input_dtype);
 
    CenterNetPreprocessor preprocessor(bmcv, input_shape[3], input_shape[2], 
                                       engine.get_input_scale(gh_names[0], input_names[0]));
    
    CenterNetPostprocessor postprocessor(output_shape, confidence, engine.get_output_scale(gh_names[0], output_names[0]));
 
    // 网络输入的batch
    int input_batch_size = input_shape[0];

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

    vector<json> results_json;

    int cn = files_vector.size();
    int id = 0;
    for (vector<string>::iterator iter = files_vector.begin();iter != files_vector.end(); iter++){
        if (input_batch_size == 1) {
            sail::BMImage imgs_0;
            sail::BMImage imgs_1(handle, input_shape[2], input_shape[3],
                                 FORMAT_BGR_PLANAR, img_dtype);
            
            string img_file = *iter;
            string img_name = img_file.substr(img_file.rfind("/")+1);
            id++;
            cout << id << "/" << cn << ", img_file: " << img_file << endl;
            ts->save("read image");
            sail::Decoder decoder((const string)img_file, true, dev_id);
            imgs_0 = decoder.read(handle);
            ts->save("read image");
            sail::BMImage rgb_img = bmcv.convert_format(imgs_0);
            std::cout << "Preprocess begin" << "\n";
            bool  align_width;
            float ratio;
            LOG_TS(ts, "Centernet preprocess");
            preprocessor.Process(rgb_img, imgs_1, align_width, ratio);
            bmcv.bm_image_to_tensor(imgs_1, input_tensor);
            LOG_TS(ts, "Centernet preprocess");
            std::cout << "Preprocess end" << "\n";
            std::cout << "Inference begin" << "\n";
            LOG_TS(ts, "Centernet inference");
            engine.process(gh_names[0], input_tensors, output_tensors);
            LOG_TS(ts, "Centernet inference");
            std::cout << "Inference end" << "\n";
 
            std::cout << "Postprocess begin" << "\n";
            LOG_TS(ts, "Centernet postprocess");
            float* output_data = reinterpret_cast<float*>(output_tensor.sys_data());
            postprocessor.Process(output_data, align_width, ratio);
            std::shared_ptr<std::vector<BMBBox>> pVectorBBox =
                postprocessor.CenternetCorrectBBox(rgb_img.height(), rgb_img.width());
            LOG_TS(ts, "Centernet postprocess");
            std::cout << "Postprocess end" << "\n";

            vector<json> bboxes_json;
            for (auto iterbox = pVectorBBox->begin(); iterbox != pVectorBBox->end(); iterbox++) {
                if (confidence <= iterbox->conf){
                    // std::cout << "Got one object, confidence:" << iterbox->conf << "\n";
                    std::cout << "Got one object, confidence:" << iterbox->conf << "; x: "<< iterbox->x<<", y: "<< iterbox->y<<", w: "<< iterbox->w<<", h: "<< iterbox->h<<", cls: "<< iterbox->c<< '\n';
                    int colors_num = colors.size();
                    int classId = iterbox->c;
                    std::tuple<int,int,int> color = std::make_tuple(colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
                    bmcv.rectangle(rgb_img, iterbox->x, iterbox->y,
                                iterbox->w, iterbox->h, color, 3);
                                      
                }
                // save result
                json bbox_json;
                bbox_json["category_id"] = iterbox->c;
                bbox_json["score"] = iterbox->conf;
                bbox_json["bbox"] = {iterbox->x, iterbox->y, iterbox->w,
                                        iterbox->h};
                bboxes_json.push_back(bbox_json);

                
            }

            json res_json;
            res_json["image_name"] = img_name;
            res_json["bboxes"] = bboxes_json;
            results_json.push_back(res_json);
            // save image
            bmcv.imwrite("./results/images/" + img_name, rgb_img);
            std::cout << "save result" << "\n";
        } else if (input_batch_size == 4) {
            std::vector<sail::BMImage> imgs_0;
            imgs_0.resize(4);
            sail::BMImageArray<4> imgs_1(handle, input_shape[2], input_shape[3],
                                          FORMAT_BGR_PLANAR, img_dtype);
            // read 4 images from image files or a video file
            std::vector<std::string> name_list;
            for (int j = 0; j < input_batch_size; ++j) {
                if (iter == files_vector.end()){
                    iter--;
                }
                string img_file = *iter;
                string img_name = img_file.substr(img_file.rfind("/")+1);
                name_list.push_back(img_name);
                id++;
                cout << id << "/" << cn << ", img_file: " << img_file << endl;
                ts->save("read image");
                sail::Decoder decoder((const string)img_file, true, dev_id);
                int ret = decoder.read(handle, imgs_0[j]);
                ts->save("read image");
                if (ret != 0) {
                    std::cout << "read failed" << "\n";
                }
                iter++;
            }
            iter--;
            std::vector<bool> align_width(4);
            std::vector<float> ratio(4);
            LOG_TS(ts, "Centernet preprocess");
            preprocessor.Process(imgs_0, imgs_1, align_width, ratio);
            bmcv.bm_image_to_tensor(imgs_1, input_tensor);
            LOG_TS(ts, "Centernet preprocess");
            std::cout << "Inference begin" << "\n";
            LOG_TS(ts, "Centernet inference");
            engine.process(gh_names[0], input_tensors, output_tensors);
            LOG_TS(ts, "Centernet inference");
            std::cout << "Inference end" << "\n";
 
            std::cout << "Postprocess begin" << "\n";
 
            float* output_data = reinterpret_cast<float*>(output_tensor.sys_data());
            for (int b = 0; b < output_shape[0]; ++b) {
                vector<json> bboxes_json;
                LOG_TS(ts, "Centernet postprocess");
                postprocessor.Process(output_data + b * postprocessor.GetBatchOffset(),
                                      align_width[b], ratio[b]);
                std::shared_ptr<std::vector<BMBBox>> pVectorBBox =
                        postprocessor.CenternetCorrectBBox(imgs_0[b].height(), imgs_0[b].width());
                LOG_TS(ts, "Centernet postprocess");
                std::cout << "Postprocess end" << "\n";

                for (auto iterbox = pVectorBBox->begin(); iterbox != pVectorBBox->end(); iterbox++) {
                    if (confidence <= iterbox->conf){
                        // std::cout << "Got one object, confidence:" << iterbox->conf << "\n";
                        std::cout << "Got one object, confidence:" << iterbox->conf << "; x: "<< iterbox->x<<", y: "<< iterbox->y<<", w: "<< iterbox->w<<", h: "<< iterbox->h<<", cls: "<< iterbox->c<< '\n';
                        int colors_num = colors.size();
                        int classId = iterbox->c;
                        std::tuple<int,int,int> color = std::make_tuple(colors[classId % colors_num][0], colors[classId % colors_num][1], colors[classId % colors_num][2]);
                        bmcv.rectangle(imgs_0[b], iterbox->x, iterbox->y,
                                    iterbox->w, iterbox->h, color, 3);
                        
                    }
                    // save result
                    json bbox_json;
                    bbox_json["category_id"] = iterbox->c;
                    bbox_json["score"] = iterbox->conf;
                    bbox_json["bbox"] = {iterbox->x, iterbox->y, iterbox->w,
                                         iterbox->h};
                    bboxes_json.push_back(bbox_json);
                    
                }

                if (b>0 && name_list[b] == name_list[b-1])
                    continue;

                json res_json;
                res_json["image_name"] = name_list[b];
                res_json["bboxes"] = bboxes_json;
                results_json.push_back(res_json);

                // save image
            
                bmcv.imwrite("./results/images/" + name_list[b], imgs_0[b]);
                std::cout << "save result" << "\n";

            }
        }
    }


    // save results
    size_t index = input.rfind("/");
    if(index == input.length() - 1){
        input = input.substr(0, input.length() - 1);
        index = input.rfind("/");
    }
    string dataset_name = input.substr(index + 1);
    index = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(index + 1);
    string json_file = "results/" + model_name + "_" + dataset_name +
                       "_sail_cpp" + "_result.json";
    cout << "================" << endl;
    cout << "result saved in " << json_file << endl;
    ofstream(json_file) << std::setw(4) << results_json;

    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    centernet_ts.calbr_basetime(base_time);
    centernet_ts.build_timeline("centernet test");
    centernet_ts.show_summary("centernet test");
    centernet_ts.clear();

    return 0;
}