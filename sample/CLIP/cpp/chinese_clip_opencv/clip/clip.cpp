//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "clip.hpp"
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include "memory.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <fstream>
#include <string>

void CLIP::init(const std::string& image_model, const std::string& text_model, const int &dev_id) {
    bm_status_t status = bm_dev_request(&bm_handle, dev_id);
    assert(BM_SUCCESS == status);
    std::cout << "set device id: " << dev_id << std::endl;
    // Create bmruntime for text model
    p_bmrt_text = bmrt_create(bm_handle); 
    assert(NULL != p_bmrt_text);
    bmrt_set_flags(p_bmrt_text, BM_RUNTIME_SHARE_MEM);
    // Load text model by file
    printf("Text Model[%s] loading ....\n", text_model.c_str());
    bool ret_text = bmrt_load_bmodel(p_bmrt_text, text_model.c_str());
    assert(true == ret_text);
    printf("Text Model Done!\n");

    // Create bmruntime for image model
    p_bmrt_image = bmrt_create(bm_handle); 
    assert(NULL != p_bmrt_image);
    bmrt_set_flags(p_bmrt_image, BM_RUNTIME_SHARE_MEM);
    // Load image model by file
    printf("Image Model[%s] loading ....\n", image_model.c_str());
    bool ret_image = bmrt_load_bmodel(p_bmrt_image, image_model.c_str());
    assert(true == ret_image);
    printf("Image Model Done!\n");

    std::string image_name = "cn_clip_image_vitb16";
    image_net = const_cast<bm_net_info_t*>(bmrt_get_network_info(p_bmrt_image, image_name.c_str()));

    const char* image_net_input_name = image_net->input_names[0];
    const char* image_net_output_name = image_net->output_names[0];
    image_net_input_shape = image_net->stages[0].input_shapes;
    image_net_output_shape = image_net->stages[0].output_shapes;
    image_net_batch_size = image_net_input_shape->dims[0];

    image_resolution = image_net_input_shape->dims[2]; // 224 for vit32-b
    embed_dim = image_net_output_shape->dims[1]; // 512 for vit32-b

    std::string text_name = "cn_clip_text_vitb16";
    text_net = const_cast<bm_net_info_t*>(bmrt_get_network_info(p_bmrt_text, text_name.c_str()));
    const char* text_net_input_name = text_net->input_names[0];
    const char* text_net_output_name = text_net->output_names[0];
    text_net_input_shape = text_net->stages[0].input_shapes;
    text_net_output_shape = text_net->stages[0].output_shapes;
    text_net_batch_size = text_net_input_shape->dims[0];
    top_k = 5;

    // load text_projection
    std::filesystem::path script_path = std::filesystem::current_path();
    std::ifstream file(script_path / "../../models/text_projection_512_512.npy", std::ios::binary);
    char header[128];
    file.read(header, 128);
    size_t header_length = 0;
    while (header[header_length] != '\n') header_length++;
    file.seekg(header_length + 1, std::ios::beg);

    const size_t rows = 512, cols = 512;
    text_projection.resize(rows, std::vector<float>(cols));

    std::vector<float> flat_data(rows * cols);
    file.read(reinterpret_cast<char*>(flat_data.data()), flat_data.size() * sizeof(float));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            text_projection[i][j] = flat_data[i * cols + j];
        }
    }

    encode_image_time = 0.0;
    encode_text_time = 0.0;
    preprocess_time = 0.0;
}

void CLIP::deinit() {
    if (p_bmrt_text) {
        bmrt_destroy(p_bmrt_text);
        p_bmrt_text = nullptr;
    }
    if (p_bmrt_image) {
        bmrt_destroy(p_bmrt_image);
        p_bmrt_image = nullptr;
    }
    bm_dev_free(bm_handle);
    text_projection.clear();
    
    encode_image_time = 0.0;
    encode_text_time = 0.0;
    preprocess_time = 0.0;
}

std::pair<std::vector<float>, std::vector<int>> CLIP::topk(const std::vector<float>& x, int k) {
    std::vector<int> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int a, int b) {
        return x[a] > x[b];
    });
    std::vector<float> values(k);
    for (int i = 0; i < k; ++i) {
        values[i] = x[indices[i]];
    }
    return {values, std::vector<int>(indices.begin(), indices.begin() + k)};
}

std::tuple<cv::Mat, std::pair<float, float>, std::pair<float, float>> CLIP:: letterbox(const cv::Mat& im, 
    const cv::Size& new_shape, 
    const cv::Scalar& color, 
    bool auto_pad, 
    bool scaleFill, 
    bool scaleup, 
    int stride) {
    cv::Size shape = im.size(); // [width, height]
    float r = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);
    if (!scaleup) {
        r = std::min(r, 1.0f);
    }

    cv::Size new_unpad(static_cast<int>(round(shape.width * r)), static_cast<int>(round(shape.height * r)));
    float dw = new_shape.width - new_unpad.width;
    float dh = new_shape.height - new_unpad.height;
    
    if (auto_pad) {
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_unpad = new_shape;
        r = static_cast<float>(new_shape.width) / shape.width;
    }

    dw /= 2;
    dh /= 2;

    cv::Mat resized_img;
    if (shape != new_unpad) {
        cv::resize(im, resized_img, new_unpad, 0, 0, cv::INTER_CUBIC);
    } else {
        resized_img = im.clone();
    }

    int top = static_cast<int>(round(dh - 0.1));
    int bottom = static_cast<int>(round(dh + 0.1));
    int left = static_cast<int>(round(dw - 0.1));
    int right = static_cast<int>(round(dw + 0.1));
    
    cv::Mat letterboxed_img;
    cv::copyMakeBorder(resized_img, letterboxed_img, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return std::make_tuple(letterboxed_img, std::make_pair(r, r), std::make_pair(dw, dh));
}


cv::Mat CLIP::preprocess_cpu_letterbox(const cv::Mat& image) {
    cv::Size new_shape(image_resolution, image_resolution);
    auto [letterbox_img, ratio, padding] = letterbox(image, new_shape);

    // Convert to RGB and normalize
    cv::Mat rgb_image;
    cv::cvtColor(letterbox_img, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0); // Convert to float and scale to [0, 1]

    std::vector<float> mean = {0.48145466, 0.4578275, 0.40821073};
    std::vector<float> std = {0.26862954, 0.26130258, 0.27577711};
    for (int c = 0; c < 3; ++c) {
                rgb_image.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* position) -> void {
                    pixel[c] = (pixel[c] - mean[c]) / std[c];
                });
            }
    cv::Mat blob;
    cv::dnn::blobFromImage(rgb_image, blob);
    int batchSize = blob.size[0];
    int channels = blob.size[1];
    int height = blob.size[2];
    int width = blob.size[3];

    // hwc -> chw
    cv::Mat outputImage = blob.reshape(1, height);
    outputImage = outputImage.reshape(channels, height);
    return outputImage;
}

std::vector<float> CLIP::preprocess(const cv::Mat& image) {
    auto start_time = std::chrono::high_resolution_clock::now();
    cv::Mat processed_image = preprocess_cpu_letterbox(image);
    std::vector<float> image_vector(bmrt_shape_count(image_net_input_shape));
    std::memcpy(image_vector.data(), processed_image.data, bmrt_shape_count(image_net_input_shape) * sizeof(float));

    preprocess_time += std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();
    return image_vector;
}

std::vector<float> CLIP::encode_image(const std::vector<float>& image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    auto &in0_mem = image_net->stages[0].input_mems[0];
    auto &out_mem = image_net->stages[0].output_mems[0];
    uint64_t in_shape = bmrt_shape_count(image_net_input_shape);
    uint64_t out_shape = bmrt_shape_count(image_net_output_shape);
    
    auto ret =  bm_memcpy_s2d_partial(bm_handle, in0_mem, (void*)image.data(), in_shape * sizeof(float));
    assert(BM_SUCCESS == ret);
    net_launch(image_net, p_bmrt_image);
    size_t batch_size = 1;
    size_t total_size = out_shape * batch_size;

    std::vector<float> output_data(out_shape, 0);
    ret =  bm_memcpy_d2s_partial(bm_handle, output_data.data(), out_mem, output_data.size() * sizeof(float));
    assert(BM_SUCCESS == ret);
    normalize(output_data);
    encode_image_time += std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();
    
    return output_data;
}

std::vector<float> CLIP::encode_text(const std::vector<int>& text) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto &in0_mem = text_net->stages[0].input_mems[0];
    auto &out_mem = text_net->stages[0].output_mems[0];
    uint64_t in_shape = bmrt_shape_count(text_net_input_shape);   
    uint64_t out_shape = bmrt_shape_count(text_net_output_shape);   

    auto ret = bm_memcpy_s2d_partial(bm_handle, in0_mem, (void*)text.data(), text.size() * sizeof(int));
    assert(BM_SUCCESS == ret);
    net_launch(text_net, p_bmrt_text);
    std::vector<float> result(embed_dim, 0.0f);
    ret = bm_memcpy_d2s_partial(bm_handle, result.data(), out_mem, result.size() * sizeof(float));
    assert(BM_SUCCESS == ret);
    normalize(result);
    encode_text_time += std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();

    return result;
}

void CLIP::normalize(std::vector<float>& features) {
    float norm = std::sqrt(std::inner_product(features.begin(), features.end(), features.begin(), 0.0f));
    for (auto& f : features) {
        f /= norm;
    }
}

std::vector<float> CLIP::calculate_similarity(const std::vector<float>& image_features,
                                        const std::vector<std::vector<float>>& text_features) {
    size_t num_text_features = text_features.size();
    std::vector<float> similarity(num_text_features);
    for (size_t i = 0; i < num_text_features; ++i) {
        similarity[i] = 100.0f * std::inner_product(image_features.begin(), image_features.end(),
                                                      text_features[i].begin(), 0.0f);
    }

    return softmax(similarity);
}

std::vector<float> CLIP::softmax(const std::vector<float>& x) {
    std::vector<float> e_x(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        e_x[i] = std::exp(x[i] - max_val);
        sum += e_x[i];
    }
    for (size_t i = 0; i < e_x.size(); ++i) {
        e_x[i] /= sum;
    }
    return e_x;
}

void CLIP::net_launch(const bm_net_info_t *net, void *p_bmrt) {
    bm_tensor_t input_tensors[1];
    bm_tensor_t output_tensors[1];

    for (int i = 0; i < net->input_num; i++) {
        bmrt_tensor_with_device(
            input_tensors, net->stages[0].input_mems[i],
            net->input_dtypes[i], net->stages[0].input_shapes[i]);
    }
    for (int i = 0; i < net->output_num; i++) {
        bmrt_tensor_with_device(
            output_tensors, net->stages[0].output_mems[i],
            net->output_dtypes[i], net->stages[0].output_shapes[i]);
    }

    auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, input_tensors,
                                      net->input_num, output_tensors,
                                      net->output_num, true, false);
                                      
    assert(ret);
    bm_thread_sync(bm_handle);
}