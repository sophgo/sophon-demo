//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "c3d.hpp"


template<class ForwardIterator>
size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}
 
template<class ForwardIterator>
size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}


//public:
C3D::C3D(std::shared_ptr<BMNNContext> context, int step_len, int dev_id):
        m_bmContext(context), m_step(step_len), m_dev_id(dev_id){
    std::cout << "C3D create bm_context" << std::endl;
}

C3D::~C3D(){
    std::cout << "C3D delete bm_context" << std::endl;   
    bm_free_device(m_bmContext->handle(), input_tensor.device_mem);
    if(m_input_tensor->get_dtype() == BM_INT8){
        delete [] m_input_int8;
    }else{
        delete [] m_input_f32;
    }
}

void C3D::Init(){
    //1. Get network.
    m_bmNetwork = m_bmContext->network(0);
    //2. Malloc host memory
    m_input_tensor = m_bmNetwork->inputTensor(0);
    m_input_count = bmrt_shape_count(m_input_tensor->get_shape());
    if(m_input_tensor->get_dtype() == BM_INT8){
        m_input_int8 = new int8_t[m_input_count];
    }else{
        m_input_f32 = new float[m_input_count];
    }
    //3. Set parameters.
    max_batch = m_bmNetwork->maxBatch();
    m_num_channels = m_input_tensor->get_shape()->dims[1] 
                   * m_input_tensor->get_shape()->dims[2]; //(3*16,112,112)
    m_clip_len = m_input_tensor->get_shape()->dims[2];
    m_net_h = m_input_tensor->get_shape()->dims[3];
    m_net_w = m_input_tensor->get_shape()->dims[4];
    std::vector<float> mean_values;
    mean_values.push_back(104.0);//ImageNet channel B mean
    mean_values.push_back(117.0);//ImageNet channel G mean
    mean_values.push_back(123.0);//ImageNet channel R mean
    setMean(mean_values);
    //4. Set device mem
    bmrt_tensor(&input_tensor, 
                m_bmContext->bmrt(), 
                m_input_tensor->get_dtype(), 
                *m_input_tensor->get_shape());
    m_input_tensor->set_device_mem(&input_tensor.device_mem);
}

int C3D::batch_size(){
    return max_batch;
}

int C3D::detect(const std::vector<std::string> &batch_videos, std::vector<int> &preds){
    int ret = 0;
    
    std::vector<cv::Mat> m_decoded_input;
    m_decoded_input.resize(max_batch * m_clip_len);
    //0. Decode videos and get frame list.
    m_ts->save("C3D decode_time");
    for(int i = 0; i < max_batch; i++){
        if(i < batch_videos.size())
            decode_video(batch_videos[i], m_decoded_input, i);
        else{
            decode_video(batch_videos[0], m_decoded_input, i); //useless data
        }
    }
    m_ts->save("C3D decode_time");

    //1. Preprocess, convert raw images to format which fits C3D network.
    m_ts->save("C3D preprocess_time");
    ret = pre_process(m_decoded_input);
    m_ts->save("C3D preprocess_time");

    CV_Assert(ret == 0);
    //2. Run C3D inference.
    m_ts->save("C3D infer_time");
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("C3D infer_time");
    
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    auto output_shape = outputTensor->get_shape();
    auto output_dims = output_shape->num_dims;
#if DEBUG
    std::cout<<"Output dims:"<<output_dims<<std::endl;
    std::cout<<"Output shape infos: "<<output_shape->dims[0]<<" "
             <<output_shape->dims[1]<<std::endl;
#endif
    assert(m_bmNetwork->outputTensorNum() == 1); 
    int class_num = output_shape->dims[1];
    auto output_data = outputTensor->get_cpu_data();
    m_ts->save("C3D postprocess_time");
    for(int i = 0; i < batch_videos.size(); i++){
        auto max_index = argmax(output_data + i * class_num, 
                                output_data + (i + 1) * class_num);
        preds.push_back(max_index);
    }
    m_ts->save("C3D postprocess_time");
    return 0;
}

void C3D::enableProfile(TimeStamp *ts){
    m_ts = ts;
}

void C3D::setMean(std::vector<float> &values) {
    std::vector<cv::Mat> channels;
    for(int i = 0; i < m_num_channels / m_clip_len; i++) {
        /* Extract an individual channel. */
        if(m_input_tensor->get_dtype() == BM_INT8){
            cv::Mat channel(m_net_h, m_net_w, CV_8SC1,cv::Scalar(0), cv::SophonDevice(m_dev_id));
            channels.push_back(channel);
        }else{
            cv::Mat channel(m_net_h, m_net_w, CV_32FC1,cv::Scalar(0), cv::SophonDevice(m_dev_id));
            channels.push_back(channel); 
        }
    }
    //init mat mean_
    std::vector<cv::Mat> channels_;
    for (int i = 0; i < m_num_channels / m_clip_len; i++) {
        /* Extract an individual channel. */
        cv::Mat channel_(m_net_h, m_net_w, CV_32FC1, cv::Scalar((float)values[i]), cv::SophonDevice(m_dev_id));
        channels_.push_back(channel_);
    }
    if (m_input_tensor->get_dtype() == BM_INT8) {
        m_mean.create(m_net_h, m_net_w, CV_8SC3, m_dev_id);
    }else{
        m_mean.create(m_net_h, m_net_w, CV_32FC3, m_dev_id);
    }

    cv::merge(channels_, m_mean);
}

void C3D::decode_video(const std::string video_path, std::vector<cv::Mat> &decoded_frames, int video_id){
    int channel_base = video_id * m_clip_len;
    auto handle = m_bmContext->handle();

    cv::VideoCapture cap(video_path, cv::CAP_ANY, m_dev_id);
    if(!cap.isOpened()){
        std::cout << "open video stream failed!" << std::endl;
        exit(1);
    }
    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
#if DEBUG
    std::cout << "Frame num: " << frame_num << std::endl;
    std::cout << "resolution of input stream: " << h << ", " << w << std::endl;
#endif
    int frame_count = 0;

    for(int i = 0; i < frame_num; i++){
        cv::Mat img(h, w, CV_8UC3, cv::SophonDevice(m_dev_id));
        cap.read(img);
        if(img.empty()) continue;
        if(frame_count >= m_clip_len)
            break;
        if(i % m_step != 0)
            continue;
        decoded_frames[channel_base + frame_count]=img;
        frame_count++;
    }
    while(frame_count < m_clip_len){
        decoded_frames[channel_base + frame_count - 1].copyTo(decoded_frames[channel_base + frame_count]);
        frame_count++;
    }
}

void print_array(float array[], int i0, int i1, int i2, int i3){
    for(int i = 0; i < 112; i++){
        std::cout<<array[i3*112 + i2 * 112 * 112 + 
                        i1 * 16 * 112 * 112 + i0 * 3 * 16 * 112 * 112 + i]<<" ";
    }
    std::cout<<std::endl;
}

void C3D::wrapInputLayer(std::vector<cv::Mat>* input_channels, int batch_id) {
    int h = m_net_h;
    int w = m_net_w;

    //init input_channels
    if(m_input_tensor->get_dtype() == BM_INT8) {
        int8_t *channel_base = m_input_int8;
        channel_base += h * w * m_num_channels * batch_id;
        for (int i = 0; i < m_num_channels; i++) {
            cv::Mat channel(h, w, CV_8SC1, channel_base);
            input_channels->push_back(channel);
            channel_base += h * w;
        }
    } else {
        float *channel_base = m_input_f32;
        channel_base += h * w * m_num_channels * batch_id;
        for (int i = 0; i < m_num_channels; i++) {
            cv::Mat channel(h, w, CV_32FC1, channel_base);
            input_channels->push_back(channel);
            channel_base += h * w;
        }
    }
}

int C3D::pre_process(const std::vector<cv::Mat> &decoded_frames){
    //Safety check.
    /* Convert the input frame to the format of the network. */   
    
    //1. Preprocess input videos in host memory.
    int ret = 0;
    for(int batch_id = 0; batch_id < max_batch; batch_id++){
        std::vector<cv::Mat> input_channels;
        wrapInputLayer(&input_channels, batch_id);
        cv::Mat tmp_channels[m_num_channels / m_clip_len];
        int channel_base = batch_id * m_clip_len;
        for(int i = channel_base; i < channel_base + m_clip_len; i++){
            cv::Mat sample_resized(171, 128, CV_8UC3, cv::SophonDevice(m_dev_id));//resize output default CV_8UC3;
            cv::resize(decoded_frames[i], sample_resized, cv::Size(171, 128)); //171 128, preprocess of UCF101.        
            cv::Mat sample_croped = sample_resized(cv::Rect(30, 8, m_net_w, m_net_h));
            cv::Mat sample_float(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(m_dev_id));
            sample_croped.convertTo(sample_float, CV_32FC3);
            cv::Mat sample_normalized(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(m_dev_id));
            cv::subtract(sample_float, m_mean, sample_normalized);
            /*note: int8 in convert need mul input_scale*/
            if (m_input_tensor->get_dtype() == BM_INT8) {
                cv::Mat sample_int8(m_net_h, m_net_w, CV_8UC3, cv::SophonDevice(m_dev_id));
                sample_normalized.convertTo(sample_int8, CV_8SC1, m_input_tensor->get_scale()); 
                cv::split(sample_int8, tmp_channels);
            } else {
                cv::split(sample_normalized, tmp_channels);
            }
            for(int j = 0; j < m_num_channels / m_clip_len; j++){
                tmp_channels[j].copyTo(input_channels[i + j * m_clip_len - channel_base]);
            }
        }
    }
    

    //2. Attach to input tensor.
    if(m_input_tensor->get_dtype() == BM_INT8){
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_int8);
    }else{
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_f32);
    }

    return 0;
}
