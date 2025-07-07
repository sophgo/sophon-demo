//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV5_H
#define YOLOV5_H
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "runtime_c5.h"
#include "tpuv7_modelrt.h"
#include "tpuv7_rt.h"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

struct YoloV5Box {
    float x, y, width, height;
    float score;
    int class_id;
};

using YoloV5BoxVec = std::vector<YoloV5Box>;

using tensorSizeType = unsigned long long;

class YoloV5 {
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;
    std::vector<std::shared_ptr<tpuRtTensor_t>> inputTensors;
    std::vector<std::shared_ptr<tpuRtTensor_t>> outputTensors;

    // configuration
    float m_confThreshold = 0.5;
    float m_nmsThreshold = 0.5;
    bool use_cpu_opt;

    std::vector<std::string> m_class_names;
    std::vector<int> batches;
    int m_class_num = 80;  // default is coco names
    int m_net_h, m_net_w;
    int max_batch = -1;
    int input_num = -1;
    int output_num = -1;
    int min_dim = -1;
    bmcv_convert_to_attr converto_attr;

    TimeStamp* m_ts;

 private:
    tpuRtNetContext_t net_ctx;
    tpuRtNet_t net{};
    std::vector<std::string> network_names = {};
    tpuRtNetInfo_t net_info{};
    bm_handle_t handle = nullptr;
    tpuRtStream_t stream;

    int pre_process(const std::vector<bm_image>& images);
    int forward(std::vector<std::shared_ptr<tpuRtTensor_t>>& input_tensors,
                std::vector<std::shared_ptr<tpuRtTensor_t>>& output_tensors);
    int post_process(const std::vector<bm_image>& images,
                     std::vector<YoloV5BoxVec>& boxes);
    int post_process_cpu_opt(const std::vector<bm_image>& images,
                             std::vector<YoloV5BoxVec>& detected_boxes);
    int argmax(float* data, int dsize);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w,
                                         int dst_h, bool* alignWidth);
    static float sigmoid(float x);
    void NMS(YoloV5BoxVec& dets, float nmsConfidence);

 public:
    YoloV5(bm_handle_t h, std::string bmodel_file, int dev_id, bool cpu_opt);
    virtual ~YoloV5();
    int Init(float confThresh = 0.5, float nmsThresh = 0.5,
             const std::string& coco_names_file = "");
    void enableProfile(TimeStamp* ts);
    int batch_size();
    int Detect(const std::vector<bm_image>& images,
               std::vector<YoloV5BoxVec>& boxes);
    void drawPred(int classId, float conf, int left, int top, int right,
                  int bottom, cv::Mat& frame);
    void draw_bmcv(bm_handle_t handle, int classId, float conf, int left,
                   int top, int right, int bottom, bm_image& frame,
                   bool put_text_flag = false);
};

inline int get_nearest_batch(std::vector<int>& batches, int real_batch) {
    int max_batch = -1;
    for (auto batch : batches) {
        if (batch >= real_batch) return batch;
        if (batch > max_batch) max_batch = batch;
    }
    return max_batch;
}

inline tensorSizeType getTensorBytes(const tpuRtTensor_t& tensor) {
    tensorSizeType ret = 1;
    for (int i = 0; i < tensor.shape.num_dims; ++i) {
        ret *= tensor.shape.dims[i];
    }
    switch (tensor.dtype) {
        case TPU_FLOAT32:
        case TPU_INT32:
        case TPU_UINT32:
            ret *= 4;
            break;
        case TPU_FLOAT16:
        case TPU_UINT16:
        case TPU_INT16:
        case TPU_BFLOAT16:
            ret *= 2;
            break;
        case TPU_INT8:
        case TPU_UINT8:
            break;
        case TPU_INT4:
        case TPU_UINT4:
            ret /= 2;
            break;
    }
    return ret;
}

inline char* get_cpu_data(std::shared_ptr<tpuRtTensor_t> tensor,
                   tpuRtStream_t& stream) {
    auto size = getTensorBytes(*tensor);
    char* cpu_data = nullptr;
    auto ret = tpuRtMallocHost((void**)&cpu_data, size);
    assert(ret == tpuRtSuccess);
    ret = tpuRtMemcpyD2SAsync(cpu_data, tensor->data, size, stream);
    assert(ret == tpuRtSuccess);
    ret = tpuRtStreamSynchronize(stream);
    assert(ret == tpuRtSuccess);
    return cpu_data;
}

#endif  // YOLOV5_H
