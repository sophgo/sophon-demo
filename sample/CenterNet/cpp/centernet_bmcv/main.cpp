//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sstream>
#include <string>
#include <numeric>
#include <chrono>
#include <stdlib.h>
#include <string.h>
#include "spdlog/spdlog.h"
#include "cvwrapper.h"
#include "engine.h"
#include "opencv2/opencv.hpp"
#include "processor.h"

// 全局变量
// 记录模型信息
std::vector<std::string> gh_names;       // graph名
std::vector<std::string> input_names;    // 网络输入名
std::vector<int>         input_shape;    // 输入shape
std::vector<std::string> output_names;   // 网络输出名
std::vector<int>         output_shape;   // 输出shape
bm_data_type_t           input_dtype;    // 输入数据类型
bm_data_type_t           output_dtype;   // 输出数据类型


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
    const char *keys="{ bmodel | ../../data/models/BM1684/centernet_fp32_1b.bmodel | bmodel file path}"
                     "{ tpu_id | 0    | TPU device id}"
                     "{ conf   | 0.35 | confidence threshold for filter boxes}"
                     "{ image  | ../../data/ctdet_test.jpg | input stream file path}"
                     "{ help   | 0    | Print help information.}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    // 模型路径
    std::string bmodel_file = parser.get<std::string>("bmodel");
    // 图片名
    std::string image_file  = parser.get<std::string>("image");
    // 设备号
    int tpu_id              = parser.get<int>("tpu_id");
    // 置信度
    float confidence        = parser.get<float>("conf");
    // 测试次数
    int test_loop           = 1;

    // 生成engine，加载模型
    sail::Engine engine(tpu_id);
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
    sail::Decoder decoder((const string)image_file, true, tpu_id);
    CenterNetPostprocessor postprocessor(output_shape, confidence, engine.get_output_scale(gh_names[0], output_names[0]));

    // 网络输入的batch
    int input_batch_size = input_shape[0];

    for (int i = 0; i < test_loop; i++) {
        if (input_batch_size == 1) {
            sail::BMImage imgs_0;
            sail::BMImage imgs_1(handle, input_shape[2], input_shape[3],
                                 FORMAT_BGR_PLANAR, img_dtype);
            imgs_0 = decoder.read(handle);
            sail::BMImage rgb_img = bmcv.convert_format(imgs_0);
            std::cout << "Preprocess begin" << "\n";
            bool  align_width;
            float ratio;
            preprocessor.Process(rgb_img, imgs_1, align_width, ratio);
            bmcv.bm_image_to_tensor(imgs_1, input_tensor);
            std::cout << "Preprocess end" << "\n";
            std::cout << "Inference begin" << "\n";
            engine.process(gh_names[0], input_tensors, output_tensors);
            std::cout << "Inference end" << "\n";

            std::cout << "Postprocess begin" << "\n";
            float* output_data = reinterpret_cast<float*>(output_tensor.sys_data());
            postprocessor.Process(output_data, align_width, ratio);
            std::shared_ptr<std::vector<BMBBox>> pVectorBBox =
                postprocessor.CenternetCorrectBBox(rgb_img.height(), rgb_img.width());
            std::cout << "Postprocess end" << "\n";

            for (auto iter = pVectorBBox->begin(); iter != pVectorBBox->end(); iter++) {
                std::cout << "Got one object, confidence:" << iter->conf << "\n";
                bmcv.rectangle(rgb_img, iter->x, iter->y,
                                iter->w, iter->h, std::make_tuple(255, 0, 0), 3);
            }
            // save result
            auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::stringstream ss;
            ss << std::put_time(std::localtime(&t), "%Y-%m-%d-%H-%M-%S");
            bmcv.imwrite("./results/ctdet_result_" + ss.str() + ".jpg", rgb_img);
            std::cout << "save result" << "\n";
        } else if (input_batch_size == 4) {
            std::vector<sail::BMImage> imgs_0;
            imgs_0.resize(4);
            sail::BMImageArray<4> imgs_1(handle, input_shape[2], input_shape[3],
                                          FORMAT_BGR_PLANAR, img_dtype);
            // read 4 images from image files or a video file
            std::vector<std::pair<int,int>> ost_size_list;
            for (int j = 0; j < input_batch_size; ++j) {
                int ret = decoder.read(handle, imgs_0[j]);
                if (ret != 0) {
                    std::cout << "read failed" << "\n";
                }
            }

            bool align_width;
            float ratio;
            preprocessor.Process(imgs_0, imgs_1, align_width, ratio);
            bmcv.bm_image_to_tensor(imgs_1, input_tensor);
            std::cout << "Inference begin" << "\n";
            engine.process(gh_names[0], input_tensors, output_tensors);
            std::cout << "Inference end" << "\n";

            std::cout << "Postprocess begin" << "\n";

            float* output_data = reinterpret_cast<float*>(output_tensor.sys_data());
            for (int b = 0; b < output_shape[0]; ++b) {
                postprocessor.Process(output_data + b * postprocessor.GetBatchOffset(),
                                      align_width, ratio);
                std::shared_ptr<std::vector<BMBBox>> pVectorBBox =
                        postprocessor.CenternetCorrectBBox(imgs_0[b].height(), imgs_0[b].width());
                std::cout << "Postprocess end" << "\n";

                // bm_image to cvmat, to avoid YUV444 case that can not draw
                cv::Mat mat1;
                cv::bmcv::toMAT(&imgs_0[b].data(), mat1);
                for (auto iter = pVectorBBox->begin(); iter != pVectorBBox->end(); iter++) {
                    std::cout << "Got one object, confidence:" << iter->conf << "\n";
                    cv::rectangle(mat1,
                                  cv::Point(iter->x, iter->y),
                                  cv::Point(iter->x + iter->w, iter->y + iter->h),
                                  cv::Scalar (255, 0, 0), 2);
//                bmcv.rectangle(imgs_0[0], iter->x, iter->y,
//                               iter->w, iter->h, std::make_tuple(255, 0, 0), 3);
                }
                // save result
                auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                std::stringstream ss;
                ss << std::put_time(std::localtime(&t), "%Y-%m-%d-%H-%M-%S");
                cv::imwrite("./results/centernet_result_" + ss.str() + "_b" + std::to_string(b) + ".jpg", mat1);
//            bmcv.imwrite("ctdet_result_" + ss.str() + ".jpg", imgs_0[0]);
                std::cout << "save result" << "\n";
            }
        }
    }
    return 0;
}