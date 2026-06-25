#ifndef UNET
#define UNET

#include<iostream>
#include<vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"

#define USE_OPENCV 1
#define DEBUG 0

class UNet
{
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;

    float m_outThreshold = 0.5;

    int m_nclasses = 2;
    int m_net_h, m_net_w;
    int max_batch = 1;
    int output_num;
    int min_dim;
    
    bmcv_convert_to_attr input_converto_attr;
    bmcv_convert_to_attr output_converto_attr;

    TimeStamp * m_ts;

    private:
    int pre_process(const std::vector<bm_image> & images);
    int post_process(const std::vector<bm_image> & images, std::vector<bm_image> & masks);
    static float sigmoid(float x);
    static int argmax(float* x, int idx, int nclasses, int feature_size);
    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);


    public:
    UNet(std::shared_ptr<BMNNContext> context);
    virtual ~UNet();
    int Init(float outThresh=0.5, int nclasses=2);
    void enableProfile(TimeStamp * ts);
    int batch_size();
    int Segment(const std::vector<bm_image> & images, std::vector<bm_image> & masks);
};

#endif