#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <string>
// #include "cnpy.h" //only for debug, and only x86.
#include "ppocr_det.hpp"
#include "postprocess.hpp"
#define USE_ZERO_PADDING 0 //Only 1684
using namespace std;

// void dump_bmimage(bm_image input, int length){
//     int* size;
//     bm_image_get_byte_size(input, size);
// }
double pythonRound(double number) {
    double integer_part = 0.0;
    double fractional_part = std::modf(number, &integer_part);

    if (fractional_part > 0.5 || (fractional_part == 0.5 && fmod(integer_part, 2.0) == 1.0)) {
        integer_part += 1.0;
    }

    return integer_part;
}

PPOCR_Detector::PPOCR_Detector(std::shared_ptr<BMNNContext> context):m_bmContext(context)
{
    std::cout << "PPOCR_Detector ..." << std::endl;
}

PPOCR_Detector::~PPOCR_Detector()
{   
    bm_image_free_contiguous_mem(max_batch, resize_bmcv_.data());
    bm_image_free_contiguous_mem(max_batch, linear_trans_bmcv_.data());
#if USE_ZERO_PADDING
    bm_image_free_contiguous_mem(max_batch, padding_bmcv_.data());
#endif
    for(int i=0; i<max_batch; i++){
        bm_image_destroy(resize_bmcv_[i]);
        bm_image_destroy(linear_trans_bmcv_[i]);
    #if USE_ZERO_PADDING
        bm_image_destroy(padding_bmcv_[i]);
    #endif
    }
}

//Donot support multi hw shape.
int PPOCR_Detector::Init()
{
    //1. get network
    m_bmNetwork = m_bmContext->network(0);

    batch1_flag = false;
    for(int i = 0; i < m_bmNetwork->m_netinfo->stage_num; i++){
        auto tensor = m_bmNetwork->inputTensor(0, i);
        if(tensor->get_shape()->dims[0] == 1){
            batch1_flag = true;
            break;
        }
    }
    if(batch1_flag == false){
        std::cerr << "Warning: A batch4 stage cannot exist alone without a batch1 stage!" << std::endl;
    }
    
    max_batch = m_bmNetwork->maxBatch();
    // max_batch = batchsize;
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
    
    if (input_is_int8_)
    { // INT8
        data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    else
    { // FP32
        data_type = DATA_TYPE_EXT_FLOAT32;
    }

    det_limit_len_ = net_w_;

    int aligned_net_w = FFALIGN(net_w_, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for(int i=0; i < max_batch; i++){
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
    #if USE_ZERO_PADDING
        ret = bm_image_create(m_bmContext->handle(),
                                net_h_,
                                net_w_,
                                FORMAT_BGR_PLANAR,
                                data_type,
                                &padding_bmcv_[i]);
        assert(BM_SUCCESS == ret);
    #endif
    }

    bm_image_alloc_contiguous_mem (max_batch, linear_trans_bmcv_.data());
#if USE_ZERO_PADDING
    bm_image_alloc_contiguous_mem (max_batch, padding_bmcv_.data());
#endif
    linear_trans_param_.alpha_0 = scale_[0] / 255.0;
    linear_trans_param_.alpha_1 = scale_[1] / 255.0;
    linear_trans_param_.alpha_2 = scale_[2] / 255.0;
    linear_trans_param_.beta_0 = (0.0 - mean_[0]) * (scale_[0]);
    linear_trans_param_.beta_1 = (0.0 - mean_[1]) * (scale_[1]);
    linear_trans_param_.beta_2 = (0.0 - mean_[2]) * (scale_[2]);

    return 0;
}

//input: any batchsize vector<bm_image>, output: responding batchsize output_boxes
int PPOCR_Detector::run(const std::vector<bm_image> &input_images, std::vector<OCRBoxVec>& output_boxes)
{   
    int ret = 0;
    std::vector<bm_image> batch_images;
    std::vector<OCRBoxVec> batch_boxes;
    for(int i = 0; i < input_images.size(); i++){
        batch_images.push_back(input_images[i]);
        if(batch_images.size() == max_batch){
            vector<vector<int>> resize_vector = preprocess_bmcv(batch_images);
            m_ts->save("(per image)Det inference", batch_images.size());
            m_bmNetwork->forward();
            m_ts->save("(per image)Det inference", batch_images.size());
            ret = postForward(batch_images, resize_vector, batch_boxes);
            output_boxes.insert(output_boxes.end(), batch_boxes.begin(), batch_boxes.end());
            batch_images.clear();
            batch_boxes.clear();
        }
    }
    // Last incomplete batch, use single batch model stage.
    if(!batch1_flag){
        int batch_size_tmp = batch_images.size();
        for(int i = batch_images.size(); i < max_batch; i++){
            batch_images.push_back(batch_images[0]);
        }
        vector<vector<int>> resize_vector = preprocess_bmcv(batch_images);
        m_ts->save("(per image)Det inference", batch_images.size());
        m_bmNetwork->forward();
        m_ts->save("(per image)Det inference", batch_images.size());
        ret = postForward(batch_images, resize_vector, batch_boxes);
        for(int i = 0; i < batch_size_tmp; i++){
            output_boxes.push_back(batch_boxes[i]);
        }
    }
    else for(int i = 0; i < batch_images.size(); i++){
        vector<vector<int>> resize_vector = preprocess_bmcv({batch_images[i]});
        m_ts->save("(per image)Det inference", 1);
        m_bmNetwork->forward();
        m_ts->save("(per image)Det inference", 1);
        ret = postForward({batch_images[i]}, resize_vector, batch_boxes);
        output_boxes.insert(output_boxes.end(), batch_boxes.begin(), batch_boxes.end());
        batch_boxes.clear();
    }
    batch_images.clear();
    batch_boxes.clear();
    return ret;
}

std::vector<std::vector<int>> PPOCR_Detector::preprocess_bmcv(const std::vector<bm_image> &input)
{
    m_ts->save("(per image)Det preprocess", input.size()); 
    std::vector<bm_image> processed_imgs;
    std::vector<std::vector<int>> resize_vector;
    int ret = 0;
    for (size_t i = 0; i < input.size(); i++) 
    {   
        bm_image image1 = input[i];
        bm_image image_aligned;
        bool need_copy = image1.width & (64-1);

        if(need_copy){
            int stride1[3], stride2[3];
            assert(BM_SUCCESS == bm_image_get_stride(image1, stride1));
            stride2[0] = FFALIGN(stride1[0], 64);
            stride2[1] = FFALIGN(stride1[1], 64);
            stride2[2] = FFALIGN(stride1[2], 64);
            assert(BM_SUCCESS == bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                image1.image_format, image1.data_type, &image_aligned, stride2));
            assert(BM_SUCCESS == bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN));
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;
            assert(BM_SUCCESS == bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1, image_aligned));
        } else {
            image_aligned = image1;
        }

        resize_vector.push_back(resize_padding_op_(image_aligned, resize_bmcv_[i], det_limit_len_));

    #if 0
        cv::Mat test;
        cv::bmcv::toMAT(&resize_bmcv_[i], test);
        cv::imwrite("resized.jpg", test);
    #endif

        assert(BM_SUCCESS == bmcv_image_convert_to(m_bmContext->handle(), 1, linear_trans_param_, &resize_bmcv_[i], &linear_trans_bmcv_[i]));
    #if USE_ZERO_PADDING
        bmcv_rect_t rect_t = {0,0,resize_vector[i][1], resize_vector[i][0]};
        bmcv_copy_to_atrr_t  atrr_t = {0, 0, 0, 0, 0, 1};
        assert(BM_SUCCESS == bm_image_create(m_bmContext->handle(),
                                    resize_vector[i][0],
                                    resize_vector[i][1],
                                    FORMAT_BGR_PLANAR,
                                    DATA_TYPE_EXT_FLOAT32,
                                    &crop_bmcv_[i]));
        assert(BM_SUCCESS == bmcv_image_crop(m_bmContext->handle(), 1, &rect_t, linear_trans_bmcv_[i], &crop_bmcv_[i]));
        assert(BM_SUCCESS == bmcv_image_copy_to(m_bmContext->handle(), atrr_t, crop_bmcv_[i], padding_bmcv_[i]));    
        bm_image_destroy(crop_bmcv_[i]);
    #endif
        if(need_copy){
            bm_image_destroy(image_aligned);
        }
    }
    
    if (input.size() == 1){
        stage = 0; //batchsize=1
    }else {
        stage = 1; //batchsize=4
    }
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0, stage);
    int image_n = input.size();

    bm_device_mem_t input_dev_mem;

#if 0 //debug code
    // std::ifstream in("../../python/bin/det_preprocessed_{}.npy", std::ios::binary);
    // float* python_det_input = new float[1*3*640*640];
    // in.read(reinterpret_cast<char*>(python_det_input), 1*3*640*640*sizeof(float));
    // assert(BM_SUCCESS == bm_malloc_device_byte(m_bmContext->handle(), &input_dev_mem, 1*3*640*640*sizeof(float)));
    // assert(BM_SUCCESS == bm_memcpy_s2d(m_bmContext->handle(), input_dev_mem, python_det_input));
    // in.close();
    static int i = 0;
    cnpy::NpyArray arr = cnpy::npy_load(cv::format("../../python/bin/det_preprocessed_%d.npy", i));
    i++;
    float* python_det_input = arr.data<float>();
    assert(BM_SUCCESS == bm_malloc_device_byte(m_bmContext->handle(), &input_dev_mem, arr.shape[0]*3*640*640*sizeof(float)));
    assert(BM_SUCCESS == bm_memcpy_s2d(m_bmContext->handle(), input_dev_mem, python_det_input));
#else
    #if USE_ZERO_PADDING
        assert(BM_SUCCESS == bm_image_get_contiguous_device_mem(input.size(), padding_bmcv_.data(), &input_dev_mem));
    #else
        assert(BM_SUCCESS == bm_image_get_contiguous_device_mem(input.size(), linear_trans_bmcv_.data(), &input_dev_mem));
    #endif
#endif
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
    
    m_ts->save("(per image)Det preprocess", input.size()); 
    return resize_vector;
}


std::vector<int> PPOCR_Detector::resize_padding_op_(bm_image src_img, bm_image &dst_img, int max_size_len)
{
    int w = src_img.width;
    int h = src_img.height;

    float ratio = 1.;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        ratio = float(max_size_len) / max_wh;
    }else{
        ratio = 1;
    }
    int resize_h = int(h * ratio);
    int resize_w = int(w * ratio);

    resize_h = max(int(pythonRound((float)resize_h / 32) * 32), 32);
    resize_w = max(int(pythonRound((float)resize_w / 32) * 32), 32);

    std::vector<int> resize_hw;
    resize_hw.push_back(resize_h);
    resize_hw.push_back(resize_w);

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
    bmcv_rect_t crop_rect{0, 0, src_img.width, src_img.height};
    assert(BM_SUCCESS == bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, src_img, &dst_img, &padding_attr, &crop_rect));
    return resize_hw;
}

int PPOCR_Detector::batch_size()
{
    return max_batch;
};

int PPOCR_Detector::postForward(const std::vector<bm_image> &batch_input_bmimg, const std::vector<std::vector<int>> resize_vector, std::vector<OCRBoxVec>& batch_boxes){
    m_ts->save("(per image)Det postprocess", batch_input_bmimg.size()); 
    float min_score_thresh = 0.3;
    double det_db_box_thresh = 0.6;
    double det_db_unclip_ratio = 1.5;

    const double threshold = min_score_thresh * 255;
    const double maxvalue = 255;
    bool use_polygon_score = false;
    
    int output_num = m_bmNetwork->outputTensorNum();
    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
    for(int i=0; i<output_num; i++){
        outputTensors[i] = m_bmNetwork->outputTensor(i, stage);
        auto output_shape = outputTensors[i]->get_shape();
        auto output_dims = output_shape->num_dims;
        int batch_num = output_shape->dims[0];
        out_net_h_ = output_shape->dims[2];
        out_net_w_ = output_shape->dims[3];
    #if 0
        std::ifstream in("python_det_output.dat", std::ios::binary);
        float* predict_batch = new float[1*1*640*640];
        in.read(reinterpret_cast<char*>(predict_batch), 1*1*640*640*sizeof(float));
        in.close();
    #else
        float* predict_batch = (float*)outputTensors[i]->get_cpu_data();
    #endif
        for(int i = 0; i < batch_num; i++)
        {   
            int resize_h = resize_vector[i][0];
            int resize_w = resize_vector[i][1];
            float ratio_h = float(resize_h) / float(batch_input_bmimg[i].height);
            float ratio_w = float(resize_w) / float(batch_input_bmimg[i].width);

            PostProcessor post_processor;

            int n = out_net_h_ * out_net_w_;
            std::vector<float> pred(n, 0.0);
            std::vector<unsigned char> cbuf(n, ' ');


            for (int j = i*n; j < (i+1)*n; j++) {
                pred[j-i*n] = float(predict_batch[j]);
                cbuf[j-i*n] = (unsigned char)((predict_batch[j]) * 255);
            }

            cv::Mat cbuf_map_(out_net_h_, out_net_w_, CV_8UC1, (unsigned char *)cbuf.data());
            cv::Mat pred_map_(out_net_h_, out_net_w_, CV_32F, (float *)pred.data()); 
            
            cv::Rect crop_region(0, 0, resize_w, resize_h);
            cv::Mat cbuf_map = cbuf_map_(crop_region);
            cv::Mat pred_map = pred_map_(crop_region);
            
            cv::Mat bit_map;
            cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

            std::vector<std::vector<std::vector<int>>> boxes = post_processor.BoxesFromBitmap(
                    pred_map, bit_map, det_db_box_thresh, det_db_unclip_ratio, use_polygon_score, batch_input_bmimg[i].width, batch_input_bmimg[i].height);

            OCRBoxVec ocrboxes = post_processor.FilterTagDetRes(boxes, batch_input_bmimg[i]);
            batch_boxes.push_back(ocrboxes);
        }
    }

    m_ts->save("(per image)Det postprocess", batch_input_bmimg.size()); 
    return 0;
}

