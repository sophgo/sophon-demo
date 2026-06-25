//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "lprnet.hpp"
#include <fstream>
#include "utils.hpp"

using namespace std;

static char const* arr_chars[] = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",  "2",  "3",  "4",
    "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",
    "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",
    "V",  "W",  "X",  "Y",  "Z",  "I",  "O",  "-"};

// public:
LPRNET::LPRNET(std::shared_ptr<BMNNContext> context, int dev_id)
    : m_bmContext(context), m_dev_id(dev_id) {
    std::cout << "LPRNET create bm_context" << std::endl;
}

LPRNET::~LPRNET() {
    std::cout << "LPRNET delete bm_context" << std::endl;
    bm_free_device(m_bmContext->handle(), input_tensor.device_mem);
    if (m_input_tensor->get_dtype() == BM_INT8) {
        delete[] m_input_int8;
    } else {
        delete[] m_input_f32;
    }
}

int LPRNET::Init() {
    // 1. Get network.
    m_bmNetwork = m_bmContext->network(0);

    // 2. Malloc host memory
    m_input_tensor = m_bmNetwork->inputTensor(0);

    m_input_count = bmrt_shape_count(m_input_tensor->get_shape());
    if (m_input_tensor->get_dtype() == BM_INT8) {
        m_input_int8 = new int8_t[m_input_count];
    } else {
        m_input_f32 = new float[m_input_count];
    }
    // 3. Set parameters.
    max_batch = m_bmNetwork->maxBatch();
    m_num_channels = m_input_tensor->get_shape()->dims[1];
    m_net_h = m_input_tensor->get_shape()->dims[2];
    m_net_w = m_input_tensor->get_shape()->dims[3];

    output_num = m_bmNetwork->outputTensorNum();
    assert(output_num > 0);
    auto output_tensor = m_bmNetwork->outputTensor(0);
    auto output_shape = output_tensor->get_shape();
    auto output_dims = output_shape->num_dims;
    clas_char = output_shape->dims[1];
    len_char = output_shape->dims[2];

    vector<float> scale_values;
    scale_values.push_back(0.0078125);
    scale_values.push_back(0.0078125);
    scale_values.push_back(0.0078125);

    std::vector<float> mean_values;
    mean_values.push_back(127.5);  // B
    mean_values.push_back(127.5);  // G
    mean_values.push_back(127.5);  // R
    setStdMean(scale_values, mean_values);

    //4. set device mem
    bmrt_tensor(&input_tensor, m_bmContext->bmrt(), m_input_tensor->get_dtype(),
                *m_input_tensor->get_shape());
    m_input_tensor->set_device_mem(&input_tensor.device_mem);
    return 0;
}

int LPRNET::batch_size() {
    return max_batch;
};

void LPRNET::enableProfile(TimeStamp* ts) {
    ts_ = ts;
}

int LPRNET::Detect(const std::vector<cv::Mat>& input_images,
                   vector<string>& results) {
    int ret = 0;
    // 3. preprocess
    ts_->save( "lprnet preprocess", input_images.size());
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    ts_->save( "lprnet preprocess", input_images.size());

    // 4. forward
    ts_->save( "lprnet inference", input_images.size());
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    ts_->save( "lprnet inference", input_images.size());

    // 5. post process
    ts_->save( "lprnet postprocess", input_images.size());
    ret = post_process(input_images, results);
    CV_Assert(ret == 0);
    ts_->save( "lprnet postprocess", input_images.size());
    return ret;
}

// private:
void LPRNET::setStdMean(std::vector<float>& std, std::vector<float>& mean) {
    // init mat mean_
    std::vector<cv::Mat> channels_mean;
    std::vector<cv::Mat> channels_std;
    for (int i = 0; i < m_num_channels; i++) {
        /* Extract an individual channel. */
        cv::Mat channel_mean(m_net_h, m_net_w, CV_32FC1,
                             cv::Scalar((float)mean[i]),
                             cv::SophonDevice(m_dev_id));
        cv::Mat channel_std(m_net_h, m_net_w, CV_32FC1,
                            cv::Scalar((float)std[i]),
                            cv::SophonDevice(m_dev_id));
        channels_mean.push_back(channel_mean);
        channels_std.push_back(channel_std);
    }
    if (m_input_tensor->get_dtype() == BM_INT8) {
        m_mean.create(m_net_h, m_net_w, CV_8SC3, m_dev_id);
        m_std.create(m_net_h, m_net_w, CV_8SC3, m_dev_id);
    } else {
        m_mean.create(m_net_h, m_net_w, CV_32FC3, m_dev_id);
        m_std.create(m_net_h, m_net_w, CV_32FC3, m_dev_id);
    }

    cv::merge(channels_mean, m_mean);
    cv::merge(channels_std, m_std);
}

void LPRNET::wrapInputLayer(std::vector<cv::Mat>* input_channels,
                            int batch_id) {
    int h = m_net_h;
    int w = m_net_w;

    // init input_channels
    if (m_input_tensor->get_dtype() == BM_INT8) {
        int8_t* channel_base = m_input_int8;
        channel_base += h * w * m_num_channels * batch_id;
        for (int i = 0; i < m_num_channels; i++) {
            cv::Mat channel(h, w, CV_8SC1, channel_base);
            input_channels->push_back(channel);
            channel_base += h * w;
        }
    } else {
        float* channel_base = m_input_f32;
        channel_base += h * w * m_num_channels * batch_id;
        for (int i = 0; i < m_num_channels; i++) {
            cv::Mat channel(h, w, CV_32FC1, channel_base);
            input_channels->push_back(channel);
            channel_base += h * w;
        }
    }
}

void LPRNET::pre_process_image(const cv::Mat& img,
                               std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample = img;
    cv::Mat sample_resized(
        m_net_h, m_net_w, CV_8UC3,
        cv::SophonDevice(m_dev_id));  // resize output default CV_8UC3;
    if (sample.size() != cv::Size(m_net_w, m_net_h)) {
        cv::resize(sample, sample_resized, cv::Size(m_net_w, m_net_h));
    } else {
        sample_resized = sample;
    }
    cv::Mat sample_float(m_net_h, m_net_w, CV_32FC3,
                         cv::SophonDevice(m_dev_id));
    sample_resized.convertTo(sample_float, CV_32FC3);
    cv::Mat sample_subtracted(m_net_h, m_net_w, CV_32FC3,
                              cv::SophonDevice(m_dev_id));
    cv::Mat sample_normalized(m_net_h, m_net_w, CV_32FC3,
                              cv::SophonDevice(m_dev_id));
    cv::subtract(sample_float, m_mean, sample_subtracted);
    cv::multiply(sample_subtracted, m_std, sample_normalized);
    /*note: int8 in convert need mul input_scale*/
    if (m_input_tensor->get_dtype() == BM_INT8) {
        cv::Mat sample_int8(m_net_h, m_net_w, CV_8UC3,
                            cv::SophonDevice(m_dev_id));
        sample_normalized.convertTo(sample_int8, CV_8SC1,
                                    m_input_tensor->get_scale());
        cv::split(sample_int8, *input_channels);
    } else {
        cv::split(sample_normalized, *input_channels);
    }
}

int LPRNET::pre_process(const std::vector<cv::Mat>& images) {
    // Safety check.
    assert(images.size() <= max_batch);
    // 1. Preprocess input images in host memory.
    int ret = 0;
    for (int i = 0; i < max_batch; i++) {
        std::vector<cv::Mat> input_channels;
        wrapInputLayer(&input_channels, i);
        if (i < images.size())
            pre_process_image(images[i], &input_channels);
        else {
            cv::Mat tmp = cv::Mat::zeros(m_net_h, m_net_w, CV_32FC3);
            pre_process_image(tmp, &input_channels);
        }
    }
    // 2. Attach to input tensor.
    if (m_input_tensor->get_dtype() == BM_INT8) {
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem,
                      (void*)m_input_int8);
    } else {
        bm_memcpy_s2d(m_bmContext->handle(), input_tensor.device_mem,
                      (void*)m_input_f32);
    }

    return 0;
}

int LPRNET::post_process(const std::vector<cv::Mat>& images,
                         vector<string>& results) {
    vector<shared_ptr<BMNNTensor>> outputTensors(output_num);
    for (int i = 0; i < output_num; i++) {
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }
    auto shape = outputTensors[0]->get_shape();

    float* output_data = nullptr;
    float ptr[clas_char];
    int pred_num[len_char];
    for (int idx = 0; idx < images.size(); ++idx) {
        for (int i = 0; i < output_num; i++) {
            auto out_tensor = outputTensors[i];
            output_data =
                (float*)out_tensor->get_cpu_data() + idx * len_char * clas_char;
            for (int j = 0; j < len_char; j++) {
                for (int k = 0; k < clas_char; k++) {
                    ptr[k] = *(output_data + k * len_char + j);
#ifdef DEBUG
                    // cout << "j = " << j << ", k = " << k << ", ptr[k] = " <<
                    // ptr[k] << endl;
#endif
                }
                int class_id = argmax(&ptr[0], clas_char);
                float confidence = ptr[class_id];
#ifdef DEBUG
                // cout << "class_id = " << class_id << ",confidence = " <<
                // confidence << endl;
#endif
                pred_num[j] = class_id;
            }
            string res = get_res(pred_num, len_char, clas_char);
#ifdef DEBUG
            cout << "res = " << res << endl;
#endif
            results.push_back(res);
        }
    }
    return 0;
}

int LPRNET::argmax(float* data, int num) {
    float max_value = -1e10;
    int max_index = 0;
    for (int i = 0; i < num; ++i) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }
    return max_index;
}

string LPRNET::get_res(int pred_num[], int len_char, int clas_char) {
    int no_repeat_blank[20];
    // int num_chars = sizeof(CHARS) / sizeof(CHARS[0]);
    int cn_no_repeat_blank = 0;
    int pre_c = pred_num[0];
    if (pre_c != clas_char - 1) {
        no_repeat_blank[0] = pre_c;
        cn_no_repeat_blank++;
    }
    for (int i = 0; i < len_char; i++) {
        if (pred_num[i] == pre_c)
            continue;
        if (pred_num[i] == clas_char - 1) {
            pre_c = pred_num[i];
            continue;
        }
        no_repeat_blank[cn_no_repeat_blank] = pred_num[i];
        pre_c = pred_num[i];
        cn_no_repeat_blank++;
    }

    string res = "";
    for (int j = 0; j < cn_no_repeat_blank; j++) {
        res = res + arr_chars[no_repeat_blank[j]];
    }

    return res;
}