//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "ssd.hpp"
extern bool IS_DIR;
extern bool CONFIDENCE;
extern bool NMS;
float overlap_FM(float x1, float w1, float x2, float w2)
{
	float l1 = x1;
	float l2 = x2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1;
	float r2 = x2 + w2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection_FM(SSDObjRect a, SSDObjRect b)
{
	float w = overlap_FM(a.x1, a.x2 - a.x1, b.x1, b.x2 - b.x1);
	float h = overlap_FM(a.y1, a.y2 - a.y1, b.y2, b.y2 - b.y1);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float box_union_FM(SSDObjRect a, SSDObjRect b)
{
	float i = box_intersection_FM(a, b);
	float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i;
	return u;
}

float box_iou_FM(SSDObjRect a, SSDObjRect b)
{
	return box_intersection_FM(a, b) / box_union_FM(a, b);
}

static bool sort_ObjRect(SSDObjRect a, SSDObjRect b)
{
    return a.score > b.score;
}

static void nms_sorted_bboxes(const std::vector<SSDObjRect>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();

    for (int i = 0; i < n; i++)    {
        const SSDObjRect& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const SSDObjRect& b = objects[picked[j]];

            float iou = box_iou_FM(a, b);
            if (iou > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

//public:
SSD::SSD(std::shared_ptr<BMNNContext> context, float conf_thre, float nms_thre, int dev_id):
        m_bmContext(context), m_conf_thre(conf_thre), m_nms_thre(nms_thre) ,m_dev_id(dev_id){
    std::cout << "SSD create bm_context" << std::endl;
}

SSD::~SSD(){
    std::cout << "SSD delete bm_context" << std::endl;   
    if(m_input_tensor->get_dtype() == BM_INT8){
        delete [] m_input_int8;
    }else{
        delete [] m_input_f32;
    }
}

void SSD::Init(){
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
    m_num_channels = m_input_tensor->get_shape()->dims[1];
    m_net_h = m_input_tensor->get_shape()->dims[2];
    m_net_w = m_input_tensor->get_shape()->dims[3];
    std::vector<float> mean_values;
    mean_values.push_back(104.0);//B
    mean_values.push_back(117.0);//G
    mean_values.push_back(123.0);//R
    setMean(mean_values);
}

int SSD::batch_size(){
    return max_batch;
}

int SSD::detect(const std::vector<cv::Mat> &input_images, const std::vector<std::string> &input_names,
                  std::vector<std::vector<SSDObjRect> > &results, cv::VideoWriter *VideoWriter){
    int ret = 0;
    //1. Preprocess, convert raw images to format which fits SSD network.
    m_ts->save("SSD preprocess");
    ret = pre_process(input_images);
    m_ts->save("SSD preprocess");

    CV_Assert(ret == 0);
    //2. Run SSD inference.
    m_ts->save("SSD inference");
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    m_ts->save("SSD inference");

    //3. Postprocess, get detected boxes and draw them on original images.
    m_ts->save("SSD postprocess");
    ret = post_process(input_images, input_names, results, VideoWriter);
    CV_Assert(ret == 0);
    m_ts->save("SSD postprocess");
    return ret;
}

void SSD::enableProfile(TimeStamp *ts){
    m_ts = ts;
}

//private:
void SSD::setMean(std::vector<float> &values) {
    std::vector<cv::Mat> channels;
    for(int i = 0; i < m_num_channels; i++) {
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
    for (int i = 0; i < m_num_channels; i++) {
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

void SSD::wrapInputLayer(std::vector<cv::Mat>* input_channels, int batch_id) {
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

void SSD::pre_process_image(const cv::Mat& img, std::vector<cv::Mat> *input_channels){
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample = img;
    cv::Mat sample_resized(m_net_h, m_net_w, CV_8UC3, cv::SophonDevice(m_dev_id));//resize output default CV_8UC3;
    if(sample.size() != cv::Size(m_net_h, m_net_w)){
        cv::resize(sample, sample_resized, cv::Size(m_net_h, m_net_w));
    }else{
        sample_resized = sample;
    }
    cv::Mat sample_float(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(m_dev_id));
    sample_resized.convertTo(sample_float, CV_32FC3);
    cv::Mat sample_normalized(m_net_h, m_net_w, CV_32FC3, cv::SophonDevice(m_dev_id));
    cv::subtract(sample_float, m_mean, sample_normalized);
    /*note: int8 in convert need mul input_scale*/
    if (m_input_tensor->get_dtype() == BM_INT8) {
        cv::Mat sample_int8(m_net_h, m_net_w, CV_8UC3, cv::SophonDevice(m_dev_id));
        sample_normalized.convertTo(sample_int8, CV_8SC1, m_input_tensor->get_scale()); 
        cv::split(sample_int8, *input_channels);
    } else {
        cv::split(sample_normalized, *input_channels);
    }
}
int SSD::pre_process(const std::vector<cv::Mat> &images){
    //Safety check.
    assert(images.size() <= max_batch);
    
    //1. Preprocess input images in host memory.
    int ret = 0;
    for(int i = 0; i < max_batch; i++){
        std::vector<cv::Mat> input_channels;
        wrapInputLayer(&input_channels, i);
        if(i < images.size())
            pre_process_image(images[i], &input_channels);
        else{
            cv::Mat tmp = cv::Mat::zeros(m_net_h, m_net_w, CV_32FC3);
            pre_process_image(tmp, &input_channels);
        }
    }
    //2. Attach to input tensor.
    bm_tensor_t input_tensor;
    bmrt_tensor(&input_tensor, 
                m_bmContext->bmrt(), 
                m_input_tensor->get_dtype(), 
                *m_input_tensor->get_shape());
    if(m_input_tensor->get_dtype() == BM_INT8){
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_int8);
    }else{
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem, (void *)m_input_f32);
    }
    m_input_tensor->set_device_mem(&input_tensor.device_mem);
    bm_free_device(m_bmContext->handle(), input_tensor.device_mem);
    return 0;
}

int SSD::post_process(const std::vector<cv::Mat> &images, const std::vector<std::string> &input_names,
                            std::vector<std::vector<SSDObjRect> > &results, cv::VideoWriter *VideoWriter){
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    auto output_shape = outputTensor->get_shape();
    auto output_dims = output_shape->num_dims; //[1 1 200 7]
    assert(m_bmNetwork->outputTensorNum() == 1);// [1 1 200/800 7]
#if DEBUG
    std::cout<<"Output shape infos: "<<output_shape->dims[0]<<" "
             <<output_shape->dims[1]<<" "<<output_shape->dims[2]<<" "
             <<output_shape->dims[3]<<std::endl;
#endif
    assert(output_dims == 4 &&
           output_shape->dims[0] == 1 &&
           output_shape->dims[1] == 1 &&
           output_shape->dims[3] == 7 );
    auto output_data = outputTensor->get_cpu_data();
    //1. Get output bounding boxes.
    int box_num_raw = output_shape->dims[0] * 
                      output_shape->dims[1] * 
                      output_shape->dims[2] ;
    for(int bid = 0; bid < box_num_raw; bid++){ //bid: box id
        SSDObjRect temp_bbox;
        temp_bbox.class_id = *(output_data + 7 * bid + 1);
        temp_bbox.score = *(output_data + 7 * bid + 2);
        int i = *(output_data + 7 * bid);
        if(i >= results.size())
            continue;
        if(CONFIDENCE || !IS_DIR){
            if(temp_bbox.score < m_conf_thre || i >= images.size()){
                continue;
            }        
        }
        temp_bbox.x1 = *(output_data + 7 * bid + 3) * m_net_w;
        temp_bbox.y1 = *(output_data + 7 * bid + 4) * m_net_h;
        temp_bbox.x2 = *(output_data + 7 * bid + 5) * m_net_w;
        temp_bbox.y2 = *(output_data + 7 * bid + 6) * m_net_h;
        results[i].push_back(temp_bbox);
    }
    for(int i = 0; i < results.size(); i++){        
        #if DEBUG
            std::cout << "width: " << images[i].cols 
                      << " height: " << images[i].rows << std::endl;
        #endif
        for(int j = 0; j < results[i].size(); j++){
            results[i][j].x1 *= (float)images[i].cols / (float)m_net_w;
            results[i][j].x2 *= (float)images[i].cols / (float)m_net_w;
            results[i][j].y1 *= (float)images[i].rows / (float)m_net_h;
            results[i][j].y2 *= (float)images[i].rows / (float)m_net_h;
        }
    }
    //2. NMS.
    if(NMS || !IS_DIR){
        for(int i = 0; i < results.size(); i++){
            std::vector<int> class_ids;
            std::vector<SSDObjRect> bboxes_per_class;
            std::vector<SSDObjRect> nmsed_results;
            for(int j = 0; j < results[i].size(); j++){
                if(std::find(class_ids.begin(), class_ids.end(), results[i][j].class_id) 
                    == class_ids.end()){
                    class_ids.push_back(results[i][j].class_id);
                }
            }
            for(int j = 0; j < class_ids.size(); j++){
                bboxes_per_class.clear();
                for(int k = 0; k < results[i].size(); k++){
                    if(results[i][k].class_id == class_ids[j])
                        bboxes_per_class.push_back(results[i][k]);
                }
                std::sort(bboxes_per_class.begin(), bboxes_per_class.end(), sort_ObjRect);
                std::vector<int> picked;
                nms_sorted_bboxes(bboxes_per_class, picked, m_nms_thre);
                for(int k = 0; k < picked.size(); k++){
                    nmsed_results.push_back(bboxes_per_class[picked[k]]);
                }
            }
            results[i].clear();
            results[i] = nmsed_results;
        }
    }
    

    for(int i = 0; i < results.size(); i++){
        auto &frame = images[i];
        int frame_width = frame.cols;
        int frame_height = frame.rows;

        for(int j = 0; j < results[i].size(); j++){
            SSDObjRect rect = results[i][j];
            cv::rectangle(frame, cv::Rect(rect.x1, rect.y1,
                                          rect.x2 - rect.x1, rect.y2 - rect.y1),
                                          cv::Scalar(255, 0, 0), 2);
        }
        if(IS_DIR){
            //save result images.
            if(access("results", 0) != F_OK)
                mkdir("results", S_IRWXU);
            if(m_bmNetwork->inputTensor(0)->get_dtype() == BM_FLOAT32){
                if(batch_size() == 1){
                    if(access("results/fp32-b1", 0) != F_OK)
                        mkdir("results/fp32-b1", S_IRWXU);
                    cv::imwrite("results/fp32-b1/" + input_names[i], frame);
                }else{
                    if(access("results/fp32-b4", 0) != F_OK)
                        mkdir("results/fp32-b4", S_IRWXU);
                    cv::imwrite("results/fp32-b4/" + input_names[i], frame);
                }
            }else{
                if(batch_size() == 1){
                    if(access("results/int8-b1", 0) != F_OK)
                        mkdir("results/int8-b1", S_IRWXU);
                    cv::imwrite("results/int8-b1/" + input_names[i], frame);
                }else{
                    if(access("results/int8-b4", 0) != F_OK)
                        mkdir("results/int8-b4", S_IRWXU);
                    cv::imwrite("results/int8-b4/" + input_names[i], frame);
                }
            }
        }else{
            //save video.
            (*VideoWriter) << frame;
        }

    }
    return 0;
}