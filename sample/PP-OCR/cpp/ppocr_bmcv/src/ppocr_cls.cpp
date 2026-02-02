#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <string>

#include "ppocr_cls.hpp"

PPOCR_Cls::PPOCR_Cls(std::shared_ptr<BMNNContext> context):m_bmContext(context)
{
    std::cout << "PPOCR_Cls ..." << std::endl;
}
PPOCR_Cls::~PPOCR_Cls()
{
    bm_image_free_contiguous_mem(max_batch, resize_bmcv_.data());
    bm_image_free_contiguous_mem(max_batch, linear_trans_bmcv_.data());
    bm_image_free_contiguous_mem(max_batch, padding_bmcv_.data());
    bm_image_free_contiguous_mem(max_batch, crop_bmcv_.data());
    for(int i=0; i<max_batch; i++){
        bm_image_destroy(resize_bmcv_[i]);
        bm_image_destroy(linear_trans_bmcv_[i]);
        bm_image_destroy(padding_bmcv_[i]);
        bm_image_destroy(crop_bmcv_[i]);
    }
}     

int PPOCR_Cls::Init(int batch_size)
{
    //1. get network
    m_bmNetwork = m_bmContext->network(0);

    max_batch = batch_size;
    resize_bmcv_.resize(max_batch);
    linear_trans_bmcv_.resize(max_batch);
    padding_bmcv_.resize(max_batch);
    crop_bmcv_.resize(max_batch);

    auto tensor = m_bmNetwork->inputTensor(0,0);
    net_h_ = tensor->get_shape()->dims[2];
    net_w_ = tensor->get_shape()->dims[3];

    if (tensor->get_dtype() == BM_FLOAT32)
        input_is_int8_ = false;
    else
        input_is_int8_ = true;
        
    // bm images for storing inference inputs
    bm_image_data_format_ext data_type;
    if (input_is_int8_)
    { // INT8
        data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    else
    { // FP32
        data_type = DATA_TYPE_EXT_FLOAT32;
    }

    // init bm images for storing results of combined operation of resize & crop & split+
    int aligned_net_w = FFALIGN(net_w_, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for(int i=0; i<max_batch; i++){
        auto ret = bm_image_create(m_bmContext->handle(), net_h_, net_w_,
            FORMAT_BGR_PLANAR,
            DATA_TYPE_EXT_1N_BYTE,
            &resize_bmcv_[i], strides);
        assert(BM_SUCCESS == ret);

        ret = bm_image_create(m_bmContext->handle(),
                                net_h_,
                                net_w_,
                                FORMAT_BGR_PLANAR,
                                data_type,
                                &linear_trans_bmcv_[i]);
        assert(BM_SUCCESS == ret);

        ret = bm_image_create(m_bmContext->handle(),
                                net_h_,
                                net_w_,
                                FORMAT_BGR_PLANAR,
                                data_type,
                                &padding_bmcv_[i]);
        assert(BM_SUCCESS == ret);
    }

    bm_image_alloc_contiguous_mem (max_batch, padding_bmcv_.data());

    linear_trans_param_.alpha_0 = 0.0078125;
    linear_trans_param_.alpha_1 = 0.0078125;
    linear_trans_param_.alpha_2 = 0.0078125;
    linear_trans_param_.beta_0 = -127.5 * 0.0078125;
    linear_trans_param_.beta_1 = -127.5 * 0.0078125;
    linear_trans_param_.beta_2 = -127.5 * 0.0078125;

    return 0;
}

std::vector<std::vector<float>> PPOCR_Cls::run(std::vector<bm_image> &input)
{
    preForward(input);
    forward();
    std::vector<std::vector<float>> cls_ret = postForward();

    return cls_ret;
}

std::vector<std::vector<float>> PPOCR_Cls::postForward()
{
    int output_num = m_bmNetwork->outputTensorNum();
    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
    std::vector<std::vector<float>> outputs_labels;
    // std::vector<int> labels;

    for(int i=0; i<output_num; i++){
        outputTensors[i] = m_bmNetwork->outputTensor(i, stage);
        auto output_shape = m_bmNetwork->outputTensor(i, stage)->get_shape();
        auto output_dims = output_shape->num_dims;
        int batch_num = output_shape->dims[0];
        int outputdim_1 = output_shape->dims[1];
        assert(output_dims == 2);

        float* output_data = nullptr;
        output_data = (float*)outputTensors[i]->get_cpu_data();
        // 获取数据
        for(int i = 0; i < batch_num; i++)
        {   
            std::vector<float> out;
            float score = 0;
            int label = 0;
            for (int j = i*outputdim_1; j < (i+1)*outputdim_1; j++) {
                if (output_data[j] > score) {
                    score = output_data[j];
                    label = j - i*outputdim_1;
                }
            }
            out.push_back(label_list[label]);
            out.push_back(score);
            outputs_labels.push_back(out);
        }
    }
    
    return outputs_labels;
}

void PPOCR_Cls::forward()
{
    m_bmNetwork->forward();
}

void PPOCR_Cls::preForward(std::vector<bm_image> &input)
{
    int ret = preprocess_bmcv(input);
}

int PPOCR_Cls::preprocess_bmcv(std::vector<bm_image> &input)
{   
    if(max_batch == 1){
        stage = 0;
    }else{
        stage = 1;
    }
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0, stage);
    int image_n = input.size();
    if (image_n > input_tensor->get_shape()->dims[0]) {
        std::cout << "input image size > input_shape.batch(" << input_tensor->get_shape()->dims[0] << ")." << std::endl;
        return -1;
    }
    //1. preprocess
    for(int i = 0; i < input.size(); i ++){
        bm_image image1 = input[i];
        bm_image image_aligned;
        bool need_copy = image1.width & (64-1);

        if(need_copy){
            int stride1[3], stride2[3];
            bm_image_get_stride(image1, stride1);
            stride2[0] = FFALIGN(stride1[0], 64);
            stride2[1] = FFALIGN(stride1[1], 64);
            stride2[2] = FFALIGN(stride1[2], 64);
            bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                image1.image_format, image1.data_type, &image_aligned, stride2);

            bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;

            bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned);
        } else {
            image_aligned = image1;
        }

        int h = image_aligned.height;
        int w = image_aligned.width;
        float ratio = w / float(h);
        int resize_h = input_tensor->get_shape()->dims[2];
        int new_w = ceilf(resize_h * ratio);
        int resize_w = input_tensor->get_shape()->dims[3];
        if(new_w < input_tensor->get_shape()->dims[3])
            resize_w = new_w;

        // resize + padding
        bmcv_padding_atrr_t padding_attr;
        memset(&padding_attr, 0, sizeof(padding_attr));
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        padding_attr.dst_crop_w = resize_w;
        padding_attr.dst_crop_h = resize_h;
        padding_attr.padding_b = 0;
        padding_attr.padding_g = 0;
        padding_attr.padding_r = 0;
        padding_attr.if_memset = 1;
        bmcv_rect_t crop_rect{0, 0, image_aligned.width, image_aligned.height};
        bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &resize_bmcv_[i], &padding_attr, &crop_rect);
        bmcv_image_convert_to(m_bmContext->handle(), 1, linear_trans_param_, &resize_bmcv_[i], &linear_trans_bmcv_[i]);
        bmcv_rect_t rect_t = {0,0,resize_w, resize_h};
        auto ret = bm_image_create(m_bmContext->handle(),
                                            resize_h,
                                            resize_w,
                                            FORMAT_BGR_PLANAR,
                                            DATA_TYPE_EXT_FLOAT32,
                                            &crop_bmcv_[i]);
        // bmcv_image_crop(m_bmContext->handle(), 1, &rect_t, linear_trans_bmcv_[i], &crop_bmcv_[i]);
        bmcv_image_vpp_convert(m_bmContext->handle(), 1, linear_trans_bmcv_[i], &crop_bmcv_[i], &rect_t);
        bmcv_copy_to_atrr_t  atrr_t = {0, 0, 0, 0, 0, 1};
        bmcv_image_copy_to(m_bmContext->handle(), atrr_t, crop_bmcv_[i], padding_bmcv_[i]);
    }
    //2. attach to tensor
    if(image_n != max_batch) 
        image_n = m_bmNetwork->get_nearest_batch(image_n); 
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, padding_bmcv_.data(), &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number

    return 0;
}

int PPOCR_Cls::batch_size()
{
    return max_batch;
};

int PPOCR_Cls::get_cls_thresh()
{
    return cls_thresh;
};

