#ifndef PPOCR_CLS_HPP
#define PPOCR_CLS_HPP

#include <iostream>
#include <vector>
#include <string.h>

#include "opencv2/opencv.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#include "bmnn_utils.h"
#include "ppocr_det.hpp"

class PPOCR_Cls
{   
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    public:
        PPOCR_Cls(std::shared_ptr<BMNNContext> context);
        virtual ~PPOCR_Cls();
        int Init(int batch_size);
        std::vector<std::vector<float>> run(std::vector<bm_image> &input);
        std::vector<std::vector<float>> postForward();
        void preForward(std::vector<bm_image> &input);
        void forward();
        int batch_size();
        int get_cls_thresh();
    
    private:
        bool input_is_int8_;
        bool output_is_int8_;

        int preprocess_bmcv(std::vector<bm_image> &input);
        std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
        std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
        std::vector<int> label_list = {0, 180};
        bool is_scale_ = true;

        double cls_thresh = 0.9;
        // bm image objects for storing intermediate results
        std::vector<bm_image> resize_bmcv_;
        std::vector<bm_image> linear_trans_bmcv_;
        std::vector<bm_image> padding_bmcv_;
        std::vector<bm_image> crop_bmcv_;

        int max_batch;
        int net_h_;
        int net_w_;
        int out_net_h_;
        int out_net_w_;

        // crop arguments of BMCV
        bmcv_rect_t crop_rect_;

        // linear transformation arguments of BMCV
        bmcv_convert_to_attr linear_trans_param_;

        int stage;
};



#endif //!PPOCR_CLS_HPP