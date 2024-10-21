//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "slowfast.hpp"

struct Element {
    float value;
    size_t index;

    // To sort values in std::set in descending order
    bool operator<(const Element& other) const {
        return value < other.value; // The less-than symbol represents sorting in ascending order.
    }
};

// Custom comparator for sorting values in descending order
struct CompareDescending {
    bool operator()(const Element& lhs, const Element& rhs) const {
        return lhs.value > rhs.value; // The greater-than symbol implements descending order sorting.
    }
};

template<class ForwardIterator>
std::vector<size_t> argmax_n(ForwardIterator first, ForwardIterator last, size_t n) {
    std::set<Element, CompareDescending> max_set; // Use a set to keep the top n largest values.

    size_t index = 0;
    for (auto it = first; it != last; ++it, ++index) {
        max_set.insert({*it, index}); // insert element

        // If the size of the set exceeds n, remove the smallest element.
        if (max_set.size() > n) {
            max_set.erase(--max_set.end()); // Remove the last element, which is the smallest element.
        }
    }

    // Extract the indices of the top n largest values.
    std::vector<size_t> indices;
    for (const auto& elem : max_set) {
        indices.push_back(elem.index);
    }

    return indices;
}

template<class ForwardIterator>
static void post_softmax(ForwardIterator first, ForwardIterator last) {
    float sum = 0.0;
    for (auto it = first; it != last; ++it) {
        *it = exp(*it);
        sum += *it;
    }
    for (auto it = first; it != last; ++it) {
        *it /= sum;
    }
}

//public:
SlowFast::SlowFast(std::shared_ptr<BMNNContext> context, int step_len, int dev_id):
        m_bmContext(context), m_step(step_len), m_dev_id(dev_id){
    std::cout << "SlowFast create bm_context" << std::endl;
}

SlowFast::~SlowFast(){
    std::cout << "SlowFast delete bm_context" << std::endl;   
    bm_free_device(m_bmContext->handle(), input_tensor.device_mem);
    bm_free_device(m_bmContext->handle(), input_tensor_fast.device_mem);
    if(m_input_tensor->get_dtype() == BM_INT8){
        delete [] m_input_int8;
        delete [] m_input_int8_fast;
    }else{
        delete [] m_input_f32;
        delete [] m_input_f32_fast;
    }
}

void SlowFast::Init(){
    //1. Get network.
    m_bmNetwork = m_bmContext->network(0);
    //2. Malloc host memory
    m_input_tensor = m_bmNetwork->inputTensor(0);
    m_input_tensor_fast = m_bmNetwork->inputTensor(1);
    m_input_count = bmrt_shape_count(m_input_tensor->get_shape());
    m_input_count_fast = bmrt_shape_count(m_input_tensor_fast->get_shape());
    if(m_input_tensor->get_dtype() == BM_INT8){
        m_input_int8 = new int8_t[m_input_count];
        m_input_int8_fast = new int8_t[m_input_count_fast];
    }else{
        m_input_f32 = new float[m_input_count];
        m_input_f32_fast = new float[m_input_count_fast];
    }
    //3. Set parameters.
    max_batch = m_bmNetwork->maxBatch();
    m_num_channels = m_input_tensor->get_shape()->dims[1] 
                   * m_input_tensor->get_shape()->dims[2]; //(3*16,112,112)
    m_num_channels_fast = m_input_tensor_fast->get_shape()->dims[1] 
                   * m_input_tensor_fast->get_shape()->dims[2]; //(3*16,112,112)
    m_clip_len = m_input_tensor->get_shape()->dims[2];
    m_clip_len_fast = m_input_tensor_fast->get_shape()->dims[2];
    m_net_h = m_input_tensor->get_shape()->dims[3];
    m_net_w = m_input_tensor->get_shape()->dims[4];
    std::vector<float> mean_values;
    mean_values.push_back(114.75);//ImageNet channel B mean
    mean_values.push_back(114.75);//ImageNet channel G mean
    mean_values.push_back(114.75);//ImageNet channel R mean
    setMean(mean_values);
    //4. Set device mem
    bmrt_tensor(&input_tensor, m_bmContext->bmrt(), m_input_tensor->get_dtype(), *m_input_tensor->get_shape());
    bmrt_tensor(&input_tensor_fast, m_bmContext->bmrt(), m_input_tensor_fast->get_dtype(), *m_input_tensor_fast->get_shape());
    m_input_tensor->set_device_mem(&input_tensor.device_mem);
    m_input_tensor_fast->set_device_mem(&input_tensor_fast.device_mem);
}

int SlowFast::batch_size(){
    return max_batch;
}

int SlowFast::detect(const std::vector<std::string> &batch_videos, std::vector<int> &preds){
    int ret = 0;
    
    std::vector<cv::Mat> m_decoded_input;
    m_decoded_input.resize(max_batch * m_clip_len_fast);
    //0. Decode videos and get frame list.
    m_ts->save("SlowFast decode_time", max_batch);
    for(int i = 0; i < max_batch; i++){
        if(i < batch_videos.size()) {
            decode_video(batch_videos[i], m_decoded_input, i);
        }else{
            decode_video(batch_videos[0], m_decoded_input, i); //useless data
        }
    }
    m_ts->save("SlowFast decode_time", max_batch);

    //1. Preprocess, convert raw images to format which fits SlowFast network.
    m_ts->save("SlowFast preprocess_time", max_batch);
    ret = pre_process(m_decoded_input);
    m_ts->save("SlowFast preprocess_time", max_batch);

    CV_Assert(ret == 0);
    //2. Run SlowFast inference.
    m_ts->save("SlowFast inference", max_batch);
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("SlowFast inference", max_batch);
    
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
    m_ts->save("SlowFast postprocess_time", max_batch);
    for(int i = 0; i < batch_videos.size(); i++){
        post_softmax(output_data + i * class_num, output_data + (i + 1) * class_num);
        auto max_indices = argmax_n(output_data + i * class_num,
                                     output_data + (i + 1) * class_num,
                                     5); // 获取前五个最大值的索引
        /*
        std::cout << "pred = " << "";
        for (int i=0;i<5;i++)
            std::cout << max_indices[i] << ", ";
        std::cout << std::endl;
        */
        preds.push_back(max_indices[0]);
    }
    m_ts->save("SlowFast postprocess_time", max_batch);
    return 0;
}

void SlowFast::enableProfile(TimeStamp *ts){
    m_ts = ts;
}

void SlowFast::setMean(std::vector<float> &values) {
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

void SlowFast::decode_video(const std::string video_path, std::vector<cv::Mat> &decoded_frames, int video_id){
    int channel_base = video_id * m_clip_len_fast;
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
        if(frame_count >= m_clip_len_fast)
            break;
        if(i % m_step != 0)
            continue;
        decoded_frames[channel_base + frame_count]=img;
        frame_count++;
    }
    while(frame_count < m_clip_len_fast){
        decoded_frames[channel_base + frame_count - 1].copyTo(decoded_frames[channel_base + frame_count]);
        frame_count++;
    }
}

void SlowFast::wrapInputLayer(std::vector<cv::Mat>* input_channels, std::vector<cv::Mat>* input_channels_fast, int batch_id) {
    int h = m_net_h;
    int w = m_net_w;

    //init input_channels
    if(m_input_tensor->get_dtype() == BM_INT8) {
        int8_t *channel_base = m_input_int8;
        channel_base += h * w * m_num_channels* batch_id;
        for (int i = 0; i < m_num_channels; i++) {
            cv::Mat channel(h, w, CV_8SC1, channel_base);
            input_channels->push_back(channel);
            channel_base += h * w;
        }
        int8_t *channel_base_fast = m_input_int8_fast;
        channel_base_fast += h * w * m_num_channels_fast * batch_id;
        for (int i = 0; i < m_num_channels_fast; i++) {
            cv::Mat channel(h, w, CV_8SC1, channel_base_fast);
            input_channels_fast->push_back(channel);
            channel_base_fast += h * w;
        }
    } else {
        float *channel_base = m_input_f32;
        channel_base += h * w * m_num_channels * batch_id;
        for (int i = 0; i < m_num_channels; i++) {
            cv::Mat channel(h, w, CV_32FC1, channel_base);
            input_channels->push_back(channel);
            channel_base += h * w;
        }
        float *channel_base_fast = m_input_f32_fast;
        channel_base_fast += h * w * m_num_channels_fast * batch_id;
        for (int i = 0; i < m_num_channels_fast; i++) {
            cv::Mat channel(h, w, CV_32FC1, channel_base_fast);
            input_channels_fast->push_back(channel);
            channel_base_fast += h * w;
        }
    }
}

int SlowFast::pre_process(const std::vector<cv::Mat> &decoded_frames){
    //1. Preprocess input videos in host memory.
    int ret = 0;
    for(int batch_id = 0; batch_id < max_batch; batch_id++){
        std::vector<cv::Mat> input_channels;
        std::vector<cv::Mat> input_channels_fast;
        wrapInputLayer(&input_channels, &input_channels_fast, batch_id);
        cv::Mat tmp_channels[m_num_channels / m_clip_len];
        int channel_base = batch_id * m_clip_len;
        int channel_base_fast = batch_id * m_clip_len_fast;

        int ori_w = decoded_frames[channel_base_fast].cols;
        int ori_h = decoded_frames[channel_base_fast].rows;
        int scale_h, scale_w, start_h = 0, start_w = 0;
        if (ori_w > ori_h) {
            scale_h = m_net_h;
            scale_w = static_cast<int>( static_cast<double>(ori_w) / ori_h * m_net_h );
            start_w = static_cast<int>( (static_cast<double>(scale_w)-m_net_w)/2 );
        } else {
            scale_w = m_net_w;
            scale_h = static_cast<int>( static_cast<double>(ori_h) / ori_w * m_net_w );
            start_h = static_cast<int>( (static_cast<double>(scale_h)-m_net_h)/2 );
        }

        for(int i = channel_base_fast; i < channel_base_fast + m_clip_len_fast; i++){
            // RGB -> BGR
            cv::Mat sample_bgr(ori_w, ori_h, CV_8UC3, cv::SophonDevice(m_dev_id));
            cv::cvtColor(decoded_frames[i], sample_bgr, cv::COLOR_RGB2BGR);
            // ShortSideScale
            cv::Mat sample_resized(scale_w, scale_h, CV_8UC3, cv::SophonDevice(m_dev_id));
            cv::resize(sample_bgr, sample_resized, cv::Size(scale_w, scale_h));
            // CenterCropVideo
            cv::Mat sample_croped = sample_resized(cv::Rect(start_w, start_h, m_net_w, m_net_h));
            // Conver to float
            cv::Mat sample_float(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(m_dev_id));
            sample_croped.convertTo(sample_float, CV_32FC3);
            // NormalizeVideo
            cv::Mat sample_normalized(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(m_dev_id));
            cv::subtract(sample_float, m_mean, sample_normalized);
            // mean = [0.45, 0.45, 0.45]
            // std = [0.225, 0.225, 0.225]
            // Then img_std = (img_ori/255.0 - mean)/std = (img_ori-114.75)/57.375
            sample_normalized = sample_normalized / 57.375;
            /*note: int8 in convert need mul input_scale*/
            if (m_input_tensor->get_dtype() == BM_INT8) {
                cv::Mat sample_int8(m_net_h, m_net_w, CV_8UC3, cv::SophonDevice(m_dev_id));
                sample_normalized.convertTo(sample_int8, CV_8SC1, m_input_tensor->get_scale()); 
                cv::split(sample_int8, tmp_channels);
            } else {
                cv::split(sample_normalized, tmp_channels);
            }
            for(int j = 0; j < m_num_channels / m_clip_len; j++){
                // Here, the fast channel has 32 frames, and the slow channel is obtained by
                // evenly extracting 8 frames from the fast channel.
                // Therefore, we need to divide i by 4 to extract from the slow channel.
                if (i % 4 == 0) {
                    tmp_channels[j].copyTo(input_channels[i/4 + j * m_clip_len - channel_base]);
                }
                tmp_channels[j].copyTo(input_channels_fast[i + j * m_clip_len_fast - channel_base_fast]);
            }
        }
    }

    //2. Attach to input tensor.
    if(m_input_tensor->get_dtype() == BM_INT8){
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_int8);
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor_fast.device_mem, (void *)m_input_int8_fast);
    }else{
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_f32);
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor_fast.device_mem, (void *)m_input_f32_fast);
    }

    return 0;
}
