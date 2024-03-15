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
// #define DEBUG
using namespace std;

static char const* arr_chars[] = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",  "2",  "3",  "4",
    "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",
    "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",
    "V",  "W",  "X",  "Y",  "Z",  "I",  "O",  "-"};

LPRNET::LPRNET(shared_ptr<BMNNContext> context) : m_bmContext(context) {
    cout << "LPRNET ctor .." << endl;
}

LPRNET::~LPRNET() {
    cout << "LPRNET dtor ..." << endl;
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for (int i = 0; i < max_batch; i++) {
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
}

int LPRNET::Init() {
    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    // 2. get input
    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];
    // 3. get output
    output_num = m_bmNetwork->outputTensorNum();
    assert(output_num > 0);
    auto output_tensor = m_bmNetwork->outputTensor(0);
    auto output_shape = output_tensor->get_shape();
    auto output_dims = output_shape->num_dims;
    clas_char = output_shape->dims[1];
    len_char = output_shape->dims[2];

#ifdef DEBUG
    cout << "net_batch = " << max_batch << endl;
    cout << "output_num = " << output_num << endl;
    cout << "output_shape = " << output_shape->dims[0] << ","
         << output_shape->dims[1] << "," << output_shape->dims[2] << endl;
#endif

    // 2. initialize bmimages
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);
    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for (int i = 0; i < max_batch; i++) {
        auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w,
                                   FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE,
                                   &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());

    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }

    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w,
                                     FORMAT_BGR_PLANAR, img_dtype,
                                     m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);
    // 3.converto
    float input_scale = tensor->get_scale();
    input_scale = input_scale * 0.0078125;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = -127.5 * input_scale;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = -127.5 * input_scale;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = -127.5 * input_scale;
    return 0;
}

void LPRNET::enableProfile(TimeStamp* ts) {
    ts_ = ts;
}

int LPRNET::batch_size() {
    return max_batch;
};

int LPRNET::Detect(const vector<bm_image>& input_images,
                   vector<string>& results) {
    int ret = 0;
    // 3. preprocess
    ts_->save("lprnet preprocess", input_images.size());
    ret = pre_process(input_images);
    CV_Assert(ret == 0);
    ts_->save("lprnet preprocess", input_images.size());

    // 4. forward
    ts_->save("lprnet inference", input_images.size());
    ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);
    ts_->save("lprnet inference", input_images.size());

    // 5. post process
    ts_->save("lprnet postprocess", input_images.size());
    ret = post_process(input_images, results);
    CV_Assert(ret == 0);
    ts_->save("lprnet postprocess", input_images.size());
    return ret;
}

int LPRNET::pre_process(const std::vector<bm_image>& images) {
    if (images.empty()) {
        cout << "pre_process input empty!!!" << endl;
        return -1;
    }

    shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    int image_n = images.size();
    if (image_n > max_batch) {
        std::cout << "input image size > input_shape.batch(" << max_batch
                  << ")." << std::endl;
        return -1;
    }
    // 1. resize image
    int ret = 0;
    for (int i = 0; i < image_n; ++i) {
        bm_image image1 = images[i];
        bm_image image_aligned;
        // src_img
        // CV_Assert(0 == cv::bmcv::toBMI((cv::Mat&)images[i], &image1, true));
        bool need_copy = image1.width & (64 - 1);

        if (need_copy) {
            int stride1[3], stride2[3];
            bm_image_get_stride(image1, stride1);
            stride2[0] = FFALIGN(stride1[0], 64);
            stride2[1] = FFALIGN(stride1[1], 64);
            stride2[2] = FFALIGN(stride1[2], 64);
            bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                            image1.image_format, image1.data_type,
                            &image_aligned, stride2);

            bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
            bmcv_copy_to_atrr_t copyToAttr;
            memset(&copyToAttr, 0, sizeof(copyToAttr));
            copyToAttr.start_x = 0;
            copyToAttr.start_y = 0;
            copyToAttr.if_padding = 1;

            bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1,
                               image_aligned);
        } else {
            image_aligned = image1;
        }
        auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1,
                                          image_aligned, &m_resized_imgs[i]);
        assert(BM_SUCCESS == ret);
        bm_image_destroy(image1);
        if (need_copy)
            bm_image_destroy(image_aligned);
    }
    // 2. converto
    ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr,
                                m_resized_imgs.data(), m_converto_imgs.data());
    CV_Assert(ret == 0);

    // 3. attach to tensor
    if (image_n != max_batch)
        image_n = m_bmNetwork->get_nearest_batch(image_n);
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(),
                                       &input_dev_mem);
    input_tensor->set_device_mem(&input_dev_mem);
    input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
    return 0;
}

int LPRNET::post_process(const vector<bm_image>& images,
                         vector<string>& results) {
    vector<shared_ptr<BMNNTensor>> outputTensors(output_num);
    for (int i = 0; i < output_num; i++) {
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }

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
