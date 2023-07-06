#ifndef PPOCR_DET_HPP
#define PPOCR_DET_HPP

#include <iostream>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include "opencv2/opencv.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0

#include "bmnn_utils.h"
#include "bm_wrapper.hpp"
#include "utils.hpp"
struct OCRBox {
  int x1, y1, x2, y2, x3, y3, x4, y4;
  std::string rec_res;
  float score;
  void printInfo() {
        printf("Box info: (%d, %d); (%d, %d); (%d, %d); (%d, %d) \n", x1, y1, x2, y2, x3, y3, x4, y4);
    }
};

using OCRBoxVec = std::vector<OCRBox>;

class PPOCR_Detector
{   
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    TimeStamp* m_ts;
    public:
        PPOCR_Detector(std::shared_ptr<BMNNContext> context);
        virtual ~PPOCR_Detector();
        int Init();
        int postForward(const std::vector<bm_image> &input, const std::vector<std::vector<int>> img_resize, std::vector<OCRBoxVec>& batch_boxes);
        int run(const std::vector<bm_image> &input_image, std::vector<OCRBoxVec>& batch_boxes);
        int batch_size();

        void enableProfile(TimeStamp *ts){
            m_ts = ts;
        };

    private:
        std::vector<std::vector<int>> preprocess_bmcv(const std::vector<bm_image> &input);
        std::vector<int> resize_padding_op_(bm_image src_img, bm_image &dst_img, int max_size_len);

        // model info 
        const bm_net_info_t *net_info_;
        const char **net_names_;
        bool batch1_flag;

        // indicate current bmodel type INT8 or FP32
        bool input_is_int8_;
        bool output_is_int8_;

        // buffer of inference results
        float* output_fp32;
        int8_t *output_int8;

        float input_scale;
        float output_scale;
        int max_batch;
        int batch_size_;
        int det_limit_len_;
        int net_h_;
        int net_w_;
        int out_net_h_;
        int out_net_w_;

        std::vector<float> mean_ = {0.485, 0.456, 0.406};
        std::vector<float> scale_ = {1 / 0.229, 1 / 0.224, 1 / 0.225};

        // pre-process

        // bm image objects for storing intermediate results
        std::vector<bm_image> resize_bmcv_;
        std::vector<bm_image> linear_trans_bmcv_;
        std::vector<bm_image> padding_bmcv_;
        std::vector<bm_image> crop_bmcv_;

        // crop arguments of BMCV
        bmcv_rect_t crop_rect_;

        // linear transformation arguments of BMCV
        bmcv_convert_to_attr linear_trans_param_;

        bm_image_data_format_ext data_type;

        int stage;

};

#endif //!PPOCR_DET_HPP