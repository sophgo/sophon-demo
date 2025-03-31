//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV8_SEG_FUSE_H
#define YOLOV8_SEG_FUSE_H

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bm_wrapper.hpp"
#define DEBUG 0

struct YoloV8Box {
    float x1, y1, x2, y2;
    float score;
    int class_id;
    cv::Mat mask_img;
};

struct ImageInfo {
    cv::Size raw_size;
    cv::Vec4d trans;
};

struct Paras {
    int r_x;
    int r_y;
    int r_w;
    int r_h;
    int width;
    int height;
};

using YoloV8BoxVec = std::vector<YoloV8Box>;

class YoloV8SegFuse {
    bm_handle_t handle;
    void *bmrt = NULL;
    const bm_net_info_t *netinfo = NULL;
    std::vector<std::string> network_names;
    bm_misc_info misc_info;

    // configuration
    std::vector<std::string> m_class_names;
    int m_net_h, m_net_w;
    bmcv_convert_to_attr converto_attr;
    TimeStamp tmp_ts;

private:
    int pre_process(const std::vector<bm_image>& images, 
                    bm_tensor_t& input_tensor,
                    std::vector<std::pair<int, int>>& txy_batch, 
                    std::vector<std::pair<float, float>>& ratios_batch);
    int forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors);
    float* get_cpu_data(bm_tensor_t* tensor, float scale);
    int post_process(const std::vector<bm_image>& input_images, 
                     std::vector<bm_tensor_t>& output_tensors, 
                     const std::vector<std::pair<int, int>>& txy_batch, 
                     const std::vector<std::pair<float, float>>& ratios_batch,
                     std::vector<YoloV8BoxVec>& boxes);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool* alignWidth);
    void xywh2xyxy(YoloV8BoxVec& xyxyboxes, std::vector<std::vector<float>> box);
    static bool YoloV8Box_cmp(YoloV8Box a, YoloV8Box b);
    void clip_boxes(YoloV8BoxVec& yolobox_vec, int src_w, int src_h);
    void get_mask(bm_image& mask_data,
                  const ImageInfo& para,
                  cv::Rect bound,
                  cv::Mat& mask_out);
public:
    int batch_size = -1;
    TimeStamp* m_ts = NULL;
    YoloV8SegFuse(std::string bmodel_file, std::string coco_names_file, int dev_id = 0){
        std::ifstream ifs(coco_names_file);
        if (ifs.is_open()) {
            std::string line;
            while (std::getline(ifs, line)) {
                line = line.substr(0, line.length() - 1);
                m_class_names.push_back(line);
            }
        }

        // get handle
        auto ret = bm_dev_request(&handle, dev_id);
        assert(BM_SUCCESS == ret);

        // judge now is pcie or soc
        ret = bm_get_misc_info(handle, &misc_info);
        assert(BM_SUCCESS == ret);

        // create bmrt
        bmrt = bmrt_create(handle);
        if (!bmrt_load_bmodel(bmrt, bmodel_file.c_str())) {
            std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
        }

        // get network names from bmodel
        const char **names;
        int num = bmrt_get_network_number(bmrt);
        if (num > 1){
            std::cout << "This bmodel have " << num << " networks, and this program will only take network 0." << std::endl;
        }
        bmrt_get_network_names(bmrt, &names);
        for(int i = 0; i < num; ++i) {
            network_names.push_back(names[i]);
        }
        free(names);

        // get netinfo by netname
        netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
        if (netinfo->stage_num > 1){
            std::cout << "This bmodel have " << netinfo->stage_num << " stages, and this program will only take stage 0." << std::endl;
        }
        batch_size = netinfo->stages[0].input_shapes[0].dims[0];
        if (batch_size != 1){
            throw std::runtime_error("Only support batch_size == 1");
        }
        m_net_h = netinfo->stages[0].input_shapes[0].dims[1];
        m_net_w = netinfo->stages[0].input_shapes[0].dims[2];
        
        for (int i = 0; i < netinfo->output_num; i++) {
            auto& shape = netinfo->stages[0].output_shapes[i];
            if (shape.num_dims == 3 && netinfo->output_dtypes[i] != BM_UINT8) {
                throw std::runtime_error("Mask output must be uint8.");
            }
        }

        // set temp timestamp
        m_ts = &tmp_ts;
    }
    ~YoloV8SegFuse(){
        if (bmrt!=NULL) {
            bmrt_destroy(bmrt);
            bmrt = NULL;
        }  
        bm_dev_free(handle);
    };
    int Detect(const std::vector<bm_image>& images, std::vector<YoloV8BoxVec>& boxes);
    void draw_result(cv::Mat& img, YoloV8BoxVec& result, float draw_thresh=0.25);
};

#endif  //! YOLOV8_SEG_H
